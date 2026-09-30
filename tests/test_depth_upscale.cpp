#include "depth_upscale.hpp"
#include "da_capi.h"
#ifdef DA_ENABLE_TIFF
#include "tiff_io.hpp"
#include <tiffio.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

using namespace da;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::fprintf(stderr, "%s: %s\n", ok ? "ok" : "FAIL", message);
    if (!ok) ++failures;
}

#ifdef DA_ENABLE_TIFF
static std::string temporary_path(const char* suffix) {
    return std::string("/tmp/da-depth-upscale-") + std::to_string(getpid()) + suffix;
}

static bool write_packed_sensor(const std::string& path, const std::vector<uint16_t>& values, int h, int w) {
    TIFF* tiff = TIFFOpen(path.c_str(), "w");
    if (!tiff) return false;
    uint16_t extra_sample = EXTRASAMPLE_UNASSALPHA;
    const bool configured = TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(w)) &&
                            TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(h)) &&
                            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8) &&
                            TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 2) &&
                            TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
                            TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
                            TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE) &&
                            TIFFSetField(tiff, TIFFTAG_EXTRASAMPLES, 1, &extra_sample);
    std::vector<uint8_t> row(static_cast<size_t>(w) * 2);
    bool written = configured;
    for (int y = 0; written && y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint16_t value = values[static_cast<size_t>(y) * w + x];
            row[2 * x] = static_cast<uint8_t>(value >> 8);
            row[2 * x + 1] = static_cast<uint8_t>(value);
        }
        written = TIFFWriteScanline(tiff, row.data(), y, 0) >= 0;
    }
    TIFFClose(tiff);
    return written;
}

static bool write_rgb_fixture(const std::string& path) {
    TIFF* tiff = TIFFOpen(path.c_str(), "w");
    if (!tiff) return false;
    const bool configured = TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 2) &&
                            TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 1) &&
                            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8) &&
                            TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3) &&
                            TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
                            TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB) &&
                            TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    const std::vector<uint8_t> pixels { 1, 2, 3, 250, 251, 252 };
    const bool written = configured && TIFFWriteScanline(tiff, const_cast<uint8_t*>(pixels.data()), 0, 0) >= 0;
    TIFFClose(tiff);
    return written;
}
#endif

int main() {

    check(da_capi_abi_version() == 12, "reports C API version 12");
    uint16_t one = 1;
    check(da_capi_depth_upscale(nullptr, "missing.tiff", &one, 1, 1, 2, 1.0f, &one) == -1,
          "C API rejects a null context");

    std::vector<uint8_t> normalized;
    std::string error;
    check(normalize_depth_u8({-2.0f, 0.0f, 2.0f}, normalized, &error), "normalizes finite model depth");
    check(normalized == std::vector<uint8_t>({0, 127, 255}), "normalization matches uint8 truncation semantics");
    check(!normalize_depth_u8({1.0f, 1.0f}, normalized, &error), "rejects constant model depth");


    constexpr int h = 4, w = 5;
    std::vector<float> relative(h * w);
    std::vector<uint16_t> sensor(h * w);
    for (int i = 0; i < h * w; ++i) {
        relative[i] = static_cast<float>(i + 1);
        sensor[i] = static_cast<uint16_t>(1000 + 200 * relative[i]);
    }
    DepthUpscaleOptions options;
    options.degree = 1;
    options.gaussian_sigma = 0.0f;
    std::vector<uint16_t> output;
    check(upscale_depth_map(sensor, h, w, relative, h, w, output, options, &error), "fits linear overlapping depth");
    check(output.front() == sensor.front() && output.back() == sensor.back(),
          "preserves calibrated sensor units without full-range stretching");
    bool monotonic = true;
    for (size_t i = 1; i < output.size(); ++i) monotonic = monotonic && output[i - 1] <= output[i];
    check(monotonic, "preserves relative-depth ordering after calibration");
    check(!upscale_depth_map(sensor, h, w, std::vector<float>(h * w, 1), h, w, output, options, &error),
          "rejects degenerate predictor");

    std::vector<uint16_t> zero_sensor {1000, 2000, 3000};
    std::vector<float> zero_relative {1.0f, 2.0f, 3.0f};
    options.degree = 2;
    check(upscale_depth_map(zero_sensor, 1, 3, zero_relative, 1, 3, output, options, &error),
          "fits normalized zero-valued predictor samples");
    check(output == zero_sensor, "calibrates the full normalized predictor range");

    // The uint8 compatibility overload must retain the supplied nonzero range:
    // normalizing it first would turn 10 into zero and leave only three samples
    // for this cubic fit.
    std::vector<uint16_t> legacy_sensor {1000, 2000, 3000, 4000};
    std::vector<uint8_t> legacy_relative {10, 20, 30, 40};
    std::vector<uint16_t> legacy_output;
    options.degree = 3;
    check(upscale_depth_map(legacy_sensor, 2, 2, legacy_relative, 2, 2, legacy_output, options, &error),
          "preserves legacy uint8 predictor values during calibration");
    options.degree = 1;

#ifdef DA_ENABLE_TIFF
    const std::string packed_path = temporary_path("-packed.tiff");
    const std::string output_path = temporary_path("-output.tiff");
    const std::string rgb_path = temporary_path("-rgb.tiff");
    check(write_rgb_fixture(rgb_path), "writes RGB TIFF fixture");
    Image rgb;
    check(load_tiff_rgb(rgb_path, rgb, &error), "reads RGB TIFF");
    check(rgb.w == 2 && rgb.h == 1 && rgb.rgb == std::vector<uint8_t>({1, 2, 3, 250, 251, 252}),
          "preserves RGB TIFF pixels");

    check(write_packed_sensor(packed_path, sensor, h, w), "writes packed sensor fixture");
    std::vector<uint16_t> loaded; int loaded_h = 0, loaded_w = 0;
    check(load_packed_depth_tiff(packed_path, loaded, loaded_h, loaded_w, &error), "reads packed sensor TIFF");
    check(loaded_h == h && loaded_w == w && loaded == sensor, "decodes big-endian two-channel sensor depth");
    check(write_depth_tiff_u16(output_path, sensor, h, w, &error), "writes uint16 depth TIFF");

    TIFF* tiff = TIFFOpen(output_path.c_str(), "r");
    uint16_t bits = 0, samples = 0, format = 0;
    const bool metadata_ok = tiff && TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits) &&
                             TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples) &&
                             TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &format);
    check(metadata_ok && bits == 16 && samples == 1 && format == SAMPLEFORMAT_UINT,
          "output TIFF has uint16 single-channel metadata");
    if (tiff) TIFFClose(tiff);
    std::remove(packed_path.c_str());
    std::remove(output_path.c_str());
    std::remove(rgb_path.c_str());
#endif

    std::fprintf(stderr, "%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
