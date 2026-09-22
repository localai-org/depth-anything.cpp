#include "cli.hpp"
#include "engine.hpp"
#include "depth_export.hpp"
#include "pose_export.hpp"
#include "ply_export.hpp"
#include "quantize.hpp"
#include "preprocess.hpp"
#include "glb_export.hpp"
#include "colmap_export.hpp"
#ifdef DA_ENABLE_TIFF
#include "depth_upscale.hpp"
#include "tiff_io.hpp"
#endif
#include <cstdio>
#include <algorithm>
#include <array>
#include <vector>
#include <string>
#include <utility>
#include <chrono>
static int cmd_info(const std::string& model){
    auto eng = da::Engine::load(model, 1);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    const auto& c = eng->config();
    std::printf("checkpoint: %s\nembed_dim: %u\ndepth: %u\nnum_heads: %u\nstatus: loaded ok\n",
                c.checkpoint_name.c_str(), c.embed_dim, c.depth, c.num_heads);
    return 0;
}
static int cmd_depth_multi(const da::cli::Parsed& p, da::Engine& eng){
    // Load all input images.
    std::vector<da::Image> imgs(p.inputs.size());
    for (size_t i = 0; i < p.inputs.size(); ++i){
        if (!da::load_image_rgb(p.inputs[i], imgs[i])){
            std::fprintf(stderr, "error: load image failed: %s\n", p.inputs[i].c_str()); return 1;
        }
    }
    std::vector<da::ViewResult> views; int H, W;
    if (!eng.depth_pose_multi(imgs, views, H, W)){ std::fprintf(stderr, "error: depth_pose_multi failed\n"); return 1; }
    // Output prefix: --out-prefix, else --pfm, else --png, else "out".
    std::string prefix = !p.out_prefix.empty() ? p.out_prefix
                       : !p.output_pfm.empty() ? p.output_pfm
                       : !p.output_png.empty() ? p.output_png : std::string("out");
    for (size_t i = 0; i < views.size(); ++i){
        const auto& r = views[i];
        float dmin = r.depth[0], dmax = r.depth[0];
        for (float v : r.depth){ dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
        std::printf("view %zu: depth %dx%d min=%.4f max=%.4f fx=%.4f fy=%.4f\n",
                    i, W, H, dmin, dmax, r.intr[0], r.intr[4]);
        std::string base = prefix + "_view" + std::to_string(i);
        da::write_pfm(base + ".pfm", r.depth, H, W);
        da::write_depth_png(base + ".png", r.depth, H, W, p.invert);
        da::write_pose_json(base + ".json", r.ext, r.intr);
    }
    return 0;
}
static int cmd_depth_metric(const da::cli::Parsed& p){
    auto eng = da::Engine::load_nested(p.model, p.metric_model, 0);
    if (!eng){ std::fprintf(stderr, "error: load_nested failed\n"); return 1; }
    da::NestedOut out; int H, W;
    if (!eng->depth_metric_path(p.input, out, H, W)){ std::fprintf(stderr, "error: depth_metric failed\n"); return 1; }
    float dmin=out.depth[0], dmax=out.depth[0]; for(float v:out.depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("metric depth %dx%d min=%.4f max=%.4f scale_factor=%.6f\n", W, H, dmin, dmax, out.scale_factor);
    if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, out.depth, H, W);
    if(!p.output_png.empty()) da::write_depth_png(p.output_png, out.depth, H, W, p.invert);
    if(!p.output_pose.empty()){
        da::write_pose_json(p.output_pose, out.extrinsics, out.intrinsics);
    }
    return 0;
}
// Bench hook: load once, run depth p.repeat times, print load + per-iter timing,
// and exit. Single-image depth only (no pose/metric/multiview). Gives clean
// inference-only timing without the per-subprocess model reload overhead.
static int cmd_depth_bench(const da::cli::Parsed& p){
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto eng = da::Engine::load(p.model, p.n_threads);
    auto t1 = clk::now();
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    const bool native = !p.legacy_resize;
    const bool with_pose = !p.output_pose.empty();
    std::vector<double> ms; ms.reserve(p.repeat);
    std::vector<float> depth, conf; int H = 0, W = 0;
    std::array<float,12> ext; std::array<float,9> intr;
    for (int it = 0; it < p.repeat; ++it){
        auto a = clk::now();
        bool ok;
        if (with_pose){
            ok = native ? eng->depth_pose_native_path(p.input, depth, conf, ext, intr, H, W)
                        : eng->depth_pose_path(p.input, depth, conf, ext, intr, H, W);
        } else {
            ok = native ? eng->depth_native(p.input, depth, conf, H, W)
                        : eng->depth(p.input, depth, conf, H, W);
        }
        auto b = clk::now();
        if (!ok){ std::fprintf(stderr, "error: %s failed\n", with_pose ? "depth_pose" : "depth"); return 1; }
        ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<double> sorted = ms; std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size()/2];
    double p90 = sorted[(size_t)(sorted.size()*0.9) < sorted.size() ? (size_t)(sorted.size()*0.9) : sorted.size()-1];
    double minv = sorted.front(), maxv = sorted.back();
    std::printf("bench: out=%dx%d threads=%d load=%.1fms infer=%.1fms/iter "
                "(median over %d, min=%.1f max=%.1f p90=%.1f)\n",
                W, H, p.n_threads > 0 ? p.n_threads : 1, load_ms, median, p.repeat,
                minv, maxv, p90);
    return 0;
}
static std::string basename_of(const std::string& path){
    size_t s = path.find_last_of("/\\");
    return s == std::string::npos ? path : path.substr(s + 1);
}
// Single-image .glb / COLMAP export. Runs the native depth+pose pipeline (the
// deterministic, downsample-free export path), captures the processed-resolution
// RGB uint8 colors, builds the 4x4 extrinsic, and writes the requested artifacts.
static int cmd_depth_export(const da::cli::Parsed& p, da::Engine& eng){
    da::Image img;
    if (!da::load_image_rgb(p.input, img)){
        std::fprintf(stderr, "error: load image failed: %s\n", p.input.c_str()); return 1; }
    std::vector<float> depth, conf; std::array<float,12> ext; std::array<float,9> intr; int H=0, W=0;
    if (!eng.depth_pose_native(img, depth, conf, ext, intr, H, W)){
        std::fprintf(stderr, "error: depth+pose failed (does this model produce camera pose?)\n"); return 1; }
    // Processed-resolution RGB uint8 — identical resize to the model input.
    da::Preprocessed pp; std::vector<uint8_t> rgb_u8;
    if (!da::preprocess_real(img, eng.config(), pp, &rgb_u8) || pp.H != H || pp.W != W){
        std::fprintf(stderr, "error: failed to capture processed image colors\n"); return 1; }
    // 3x4 extrinsic -> 4x4 (append [0,0,0,1]).
    std::array<float,16> ext4{};
    for (int i = 0; i < 12; ++i) ext4[i] = ext[i];
    ext4[12] = 0.f; ext4[13] = 0.f; ext4[14] = 0.f; ext4[15] = 1.f;
    std::vector<std::array<float,9>>  K { intr };
    std::vector<std::array<float,16>> E { ext4 };
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    float dmin=depth[0], dmax=depth[0]; for(float v:depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("depth %dx%d min=%.4f max=%.4f fx=%.4f fy=%.4f\n", W, H, dmin, dmax, intr[0], intr[4]);
    if (!p.output_glb.empty()){
        if (!da::write_glb(p.output_glb, depth, conf, K, E, imgs_u8, H, W, 1, da::GlbOptions{})){
            std::fprintf(stderr, "error: write_glb failed\n"); return 1; }
        std::printf("wrote %s\n", p.output_glb.c_str());
    }
    if (!p.output_colmap.empty()){
        std::vector<std::string> names { basename_of(p.input) };
        std::vector<std::pair<int,int>> orig_wh { { img.w, img.h } };
        if (!da::write_colmap(p.output_colmap, depth, conf, K, E, imgs_u8, names, orig_wh, H, W, 1, p.colmap_binary)){
            std::fprintf(stderr, "error: write_colmap failed\n"); return 1; }
        std::printf("wrote COLMAP model (%s) to %s\n", p.colmap_binary ? "bin" : "txt", p.output_colmap.c_str());
    }
    // Pose is already computed for the glb/colmap export above; honor --pose here
    // too (previously silently dropped when --glb/--colmap were present).
    if (!p.output_pose.empty()){
        if (!da::write_pose_json(p.output_pose, ext, intr)){
            std::fprintf(stderr, "error: write pose json failed\n"); return 1; }
        std::printf("wrote %s\n", p.output_pose.c_str());
    }
    if (!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
    if (!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
    return 0;
}
#ifdef DA_ENABLE_TIFF
static int cmd_depth_upscale(const da::cli::Parsed& p){
    std::string error;
    da::Image image;
    if (!da::load_tiff_rgb(p.input, image, &error)){
        std::fprintf(stderr, "error: load RGB TIFF failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<uint16_t> sensor_depth; int sensor_h = 0, sensor_w = 0;
    if (!da::load_packed_depth_tiff(p.sensor_depth_tiff, sensor_depth, sensor_h, sensor_w, &error)){
        std::fprintf(stderr, "error: load packed sensor depth TIFF failed: %s\n", error.c_str());
        return 1;
    }
    auto eng = da::Engine::load(p.model, p.n_threads);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    std::vector<float> predicted; int prediction_h = 0, prediction_w = 0;
    if (!da::predict_depth_for_upscale(*eng, image, predicted, prediction_h, prediction_w, &error)){
        std::fprintf(stderr, "error: model inference failed: %s\n", error.c_str());
        return 1;
    }
    da::DepthUpscaleOptions options;
    options.degree = p.upscale_degree;
    std::vector<uint16_t> output;
    if (!da::upscale_depth_map(sensor_depth, sensor_h, sensor_w, predicted, prediction_h, prediction_w,
                               output, options, &error)){
        std::fprintf(stderr, "error: depth calibration failed: %s\n", error.c_str());
        return 1;
    }
    if (!da::write_depth_tiff_u16(p.output_depth_tiff, output, sensor_h, sensor_w, &error)){
        std::fprintf(stderr, "error: write depth TIFF failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("upscaled depth %dx%d from model %dx%d -> %s\n", sensor_w, sensor_h, prediction_w,
                prediction_h, p.output_depth_tiff.c_str());
    return 0;
}
#endif

static int cmd_depth(const da::cli::Parsed& p){
    if (!p.metric_model.empty()) return cmd_depth_metric(p);
    if (p.repeat > 0 && p.inputs.size() <= 1) return cmd_depth_bench(p);
    auto eng = da::Engine::load(p.model, p.n_threads);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    if (!p.output_glb.empty() || !p.output_colmap.empty()){
        if (p.inputs.size() > 1){ std::fprintf(stderr, "error: --glb/--colmap support a single --input only\n"); return 1; }
        return cmd_depth_export(p, *eng);
    }
    if (p.inputs.size() > 1) return cmd_depth_multi(p, *eng);
    // Depth Anything V2 (arch=="depthanything2"): relative/metric depth only.
    if (eng->is_da2()){
        std::vector<float> depth; int H, W;
        if (!eng->depth_relative_path(p.input, depth, H, W)){ std::fprintf(stderr, "error: depth_relative failed\n"); return 1; }
        float dmin=depth.empty()?0:depth[0], dmax=dmin;
        for (float v : depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v); }
        std::printf("depth %dx%d min=%.4f max=%.4f (da2)\n", W, H, dmin, dmax);
        if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
        if(!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
        return 0;
    }
    // Standalone monocular checkpoint (output_dim==1 + sky head): depth + sky,
    // no conf / pose. Auto-detected from the gguf head metadata.
    if (eng->is_mono()){
        std::vector<float> depth, sky; int H, W;
        if (!eng->depth_mono_path(p.input, depth, sky, H, W)){ std::fprintf(stderr, "error: depth_mono failed\n"); return 1; }
        float dmin=depth[0], dmax=depth[0]; for(float v:depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
        std::printf("depth %dx%d min=%.4f max=%.4f (mono+sky)\n", W, H, dmin, dmax);
        if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
        if(!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
        if(!p.output_sky.empty()) da::write_pfm(p.output_sky, sky, H, W);
        return 0;
    }
    std::vector<float> depth, conf; int H,W;
    // Default: native-resolution real DA3 resize. --legacy-resize forces the old floor path.
    const bool native = !p.legacy_resize;
    if (p.ray_pose && p.output_pose.empty()){
        std::fprintf(stderr, "error: --ray-pose requires --pose <out.json>\n"); return 1;
    }
    if (!p.output_pose.empty()){
        std::array<float,12> ext; std::array<float,9> intr;
        bool ok;
        if (p.ray_pose){
            if (!eng->has_aux()){
                std::fprintf(stderr, "error: --ray-pose needs an aux-bearing GGUF (convert with --with-aux)\n");
                return 1;
            }
            ok = eng->depth_pose_rays_native_path(p.input, depth, conf, ext, intr, H, W);
        } else {
            ok = native ? eng->depth_pose_native_path(p.input, depth, conf, ext, intr, H, W)
                        : eng->depth_pose_path(p.input, depth, conf, ext, intr, H, W);
        }
        if (!ok){ std::fprintf(stderr, "error: depth_pose failed\n"); return 1; }
        std::printf("pose%s: fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n", p.ray_pose ? " (ray)" : "",
                    intr[0], intr[4], intr[2], intr[5]);
        if (!da::write_pose_json(p.output_pose, ext, intr)){ std::fprintf(stderr, "error: write pose json failed\n"); return 1; }
    } else {
        bool ok = native ? eng->depth_native(p.input, depth, conf, H, W)
                         : eng->depth(p.input, depth, conf, H, W);
        if (!ok){ std::fprintf(stderr, "error: depth failed\n"); return 1; }
    }
    float dmin=depth[0], dmax=depth[0]; for(float v:depth){ dmin=std::min(dmin,v); dmax=std::max(dmax,v);}
    std::printf("depth %dx%d min=%.4f max=%.4f\n", W, H, dmin, dmax);
    if(!p.output_pfm.empty()) da::write_pfm(p.output_pfm, depth, H, W);
    if(!p.output_png.empty()) da::write_depth_png(p.output_png, depth, H, W, p.invert);
    return 0;
}
static int cmd_reconstruct(const da::cli::Parsed& p){
    auto eng = da::Engine::load(p.model, 0);
    if (!eng){ std::fprintf(stderr, "error: load failed\n"); return 1; }
    da::Gaussians g; int H, W;
    if (!eng->reconstruct_path(p.input, g, H, W)){ std::fprintf(stderr, "error: reconstruct failed\n"); return 1; }
    std::printf("reconstructed %d gaussians (%dx%d)\n", g.N, W, H);
    if (!da::write_gaussian_ply(p.output_ply, g)){ std::fprintf(stderr, "error: write ply failed\n"); return 1; }
    std::printf("wrote %s\n", p.output_ply.c_str());
    return 0;
}
static int cmd_quantize(const da::cli::Parsed& p){
    if(!da::quantize_gguf(p.q_in, p.q_out, p.q_type)){ std::fprintf(stderr,"error: quantize failed\n"); return 1; }
    std::printf("wrote %s (%s)\n", p.q_out.c_str(), p.q_type.c_str());
    return 0;
}
int main(int argc, char** argv){
    auto p = da::cli::parse(argc, argv);
    if (!p.error.empty()){ std::fprintf(stderr, "error: %s\n", p.error.c_str()); da::cli::print_help(); return 1; }
    using S = da::cli::Sub;
    switch (p.sub){
        case S::Info: return cmd_info(p.model);
        case S::Depth: return cmd_depth(p);
        case S::Reconstruct: return cmd_reconstruct(p);
        case S::Quantize: return cmd_quantize(p);
        case S::DepthUpscale:
#ifdef DA_ENABLE_TIFF
            return cmd_depth_upscale(p);
#else
            std::fprintf(stderr, "error: depth-upscale requires a build with -DDA_ENABLE_TIFF=ON\n");
            return 1;
#endif
        case S::Help: da::cli::print_help(); return 0;
        default: da::cli::print_help(); return 1;
    }
}
