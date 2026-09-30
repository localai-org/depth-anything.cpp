#pragma once
#include <string>
#include <vector>
namespace da { namespace cli {
enum class Sub { Info, Depth, DepthUpscale, Reconstruct, Quantize, Help, None };
struct Parsed {
    Sub sub = Sub::None;
    std::string model;
    std::string metric_model;          // depth: --metric-model => nested metric-scale depth
    std::string input, output_pfm, output_png;
    std::string output_sky;            // depth (mono): --sky out.pfm => dump the sky map
    std::string output_pose;
    std::string output_ply;            // reconstruct: --ply out.ply
    std::string output_glb;            // depth: --glb out.glb (glTF-2.0 point cloud)
    std::string output_colmap;         // depth: --colmap <dir> (binary COLMAP model)
    bool colmap_binary = true;         // --colmap => bin; --colmap-txt => txt variant
    std::vector<std::string> inputs;   // accumulates repeated --input; >1 => multi-view mode
    std::string out_prefix;            // --out-prefix for multi-view outputs
    std::string sensor_depth_tiff;     // depth-upscale: packed two-channel sensor TIFF
    std::string output_depth_tiff;     // depth-upscale: calibrated uint16 TIFF
    std::string q_in, q_out, q_type;   // quantize: <in.gguf> <out.gguf> <type>
    int n_threads = 0;                 // --threads N (0 => engine default of 1)
    int repeat    = 0;                 // --repeat N  (bench hook: time N inferences, then exit)
    int upscale_degree = 2;            // depth-upscale: polynomial calibration degree
    bool invert = true;
    // Single-image depth uses the REAL DA3 native-resolution resize by default.
    // --legacy-resize forces the old floor-to-patch path (fixture parity only).
    bool legacy_resize = false;
    // --ray-pose: solve camera pose from the aux ray field (use_ray_pose) instead of
    // the cam_dec MLP. Requires an aux-bearing GGUF (converted with --with-aux).
    bool ray_pose = false;
    std::string error;
};
Parsed parse(int argc, char** argv);
void print_help();
}}
