// Verifies the C-API routes a Depth Anything V2 (relative ViT-L) GGUF through the
// DA2 depth-only path: da_capi_depth_dense returns depth only (no conf/sky/pose),
// is_metric==0 for the relative model, and da_capi_pose_path returns -1 (no pose).
#include "da_capi.h"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <cmath>

static bool finite_all(const float* p, int n){
    for (int i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}

int main(){
    const char* gguf = std::getenv("DA_TEST_GGUF_DA2"); // relative ViT-L DA2 GGUF
    if (!gguf) return 77;                               // skip if fixture absent
    const char* png = "assets/samples/desk.jpg";        // WORKING_DIRECTORY = DA_ROOT
    FILE* f = std::fopen(png, "rb");
    if (!f){ std::fprintf(stderr, "sample image %s absent, skipping\n", png); return 77; }
    std::fclose(f);

    da_ctx* c = da_capi_load(gguf, 1);
    if (!c){ std::fprintf(stderr, "da2: load failed\n"); return 1; }

    const uint16_t projected_range[] = {
        1000, 1100, 1200, 1300,
        1150, 1250, 1350, 1450,
        1300, 1400, 1500, 1600,
        1450, 1550, 1650, 1750,
    };
    uint16_t upscaled[16] = {};
    const int invalid_sigma = da_capi_depth_upscale(
        c, png, projected_range, 4, 4, 1, std::numeric_limits<float>::max(), upscaled);
    const int invalid_degree = da_capi_depth_upscale(
        c, png, projected_range, 4, 4, 9, 0.0f, upscaled);
    const int invalid_image = da_capi_depth_upscale(
        c, "missing-image.jpg", projected_range, 4, 4, 1, 0.0f, upscaled);
    const int upscale = da_capi_depth_upscale(
        c, png, projected_range, 4, 4, 1, 0.0f, upscaled);
    bool upscale_ok = invalid_sigma == -1 && invalid_degree == -1 && invalid_image == -1 && upscale == 0;
    for (uint16_t value : upscaled) upscale_ok = upscale_ok && value > 0;
    std::fprintf(stderr, "da2 upscale: invalid_sigma=%d invalid_degree=%d invalid_image=%d result=%d -> %s\n",
                 invalid_sigma, invalid_degree, invalid_image, upscale, upscale_ok ? "OK" : "FAIL");


    int H=0, W=0, is_metric=-1; float *depth=nullptr, *conf=nullptr, *sky=nullptr;
    float ext[12], intr[9];
    int r = da_capi_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext, intr, &is_metric);
    bool ok = (r == 0) && H>0 && W>0 && depth && !conf && !sky;
    if (ok) ok = (H*W > 0) && finite_all(depth, H*W);
    if (ok) ok = (is_metric == 0); // relative DA2 -> non-metric
    std::fprintf(stderr, "da2 dense: r=%d %dx%d depth=%p conf=%p sky=%p is_metric=%d -> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, is_metric, ok?"OK":"FAIL");
    da_capi_free_floats(depth); da_capi_free_floats(conf); da_capi_free_floats(sky);

    // DA2 has no camera pose: pose_path must fail.
    int rp = da_capi_pose_path(c, png, ext, intr);
    bool okp = (rp == -1);
    std::fprintf(stderr, "da2 pose: r=%d (expect -1) err=\"%s\" -> %s\n",
                 rp, da_capi_last_error(c), okp?"OK":"FAIL");

    da_capi_free(c);
    return (ok && okp && upscale_ok) ? 0 : 1;
}
