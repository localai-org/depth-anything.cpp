#include "da_capi.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Fixture {
    const char* label;
    const char* gguf_var;
    const char* image_var;
};

static bool exists(const char* path) {
    if (!path) return false;
    FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    std::fclose(file);
    return true;
}

static bool run_upscale(const char* label, const char* gguf_var, const char* image_var) {
    const char* gguf = std::getenv(gguf_var);
    const char* image = std::getenv(image_var);
    if (!exists(gguf) || !exists(image)) return false;

    da_ctx* ctx = da_capi_load(gguf, 1);
    if (!ctx) {
        std::fprintf(stderr, "%s: failed to load model\n", label);
        return false;
    }
    const uint16_t projected_range[] = {
        1000, 1100, 1200, 1300,
        1150, 1250, 1350, 1450,
        1300, 1400, 1500, 1600,
        1450, 1550, 1650, 1750,
    };
    uint16_t output[16] = {};
    const int result = da_capi_depth_upscale(ctx, image, projected_range, 4, 4, 1, 0.0f, output);
    bool ok = result == 0;
    for (uint16_t value : output) ok = ok && value > 0;
    std::fprintf(stderr, "%s: upscale=%d -> %s\n", label, result, ok ? "OK" : "FAIL");
    da_capi_free(ctx);
    return ok;
}

int main() {
    int available = 0;
    bool ok = true;
    const Fixture fixtures[] = {
        {"DualDPT DA3", "DA_TEST_GGUF", "DA_TEST_NATIVE_PNG"},
        {"DA3 mono", "DA_TEST_GGUF_MONO", "DA_TEST_MONO_PNG"},
    };
    for (const Fixture& fixture : fixtures) {
        if (!exists(std::getenv(fixture.gguf_var)) || !exists(std::getenv(fixture.image_var))) continue;
        ++available;
        ok = run_upscale(fixture.label, fixture.gguf_var, fixture.image_var) && ok;
    }
    if (available == 0) {
        std::fprintf(stderr, "DA3 upscale fixtures absent, skipping\n");
        return 77;
    }
    return ok ? 0 : 1;
}
