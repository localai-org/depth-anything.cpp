#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace da {

class Engine;
struct Image;
inline constexpr float kMaxDepthUpscaleGaussianSigma = 64.0f;


struct DepthUpscaleOptions {
    int degree = 2;
    float gaussian_sigma = 1.0f;
};

// Runs the depth-only inference path suitable for range calibration. Supports
// DA2, DA3MONO, and the single-file DA3 DualDPT models without computing pose.
bool predict_depth_for_upscale(Engine& engine, const Image& image,
                               std::vector<float>& depth, int& h, int& w,
                               std::string* error = nullptr);

// Calibrates a Depth Anything map against projected sensor ranges. The
// predictor is normalized, resized, Gaussian-smoothed, and polynomial-fitted
// over pixels where sensor samples exist. Output remains calibrated uint16
// sensor units (normally millimetres); it is not stretched to the full range.
bool upscale_depth_map(const std::vector<uint16_t>& sensor_depth, int sensor_h, int sensor_w,
                       const std::vector<float>& relative_depth, int relative_h, int relative_w,
                       std::vector<uint16_t>& output, const DepthUpscaleOptions& options = {},
                       std::string* error = nullptr);

// Compatibility overload for the former normalized-uint8 service boundary.
bool upscale_depth_map(const std::vector<uint16_t>& sensor_depth, int sensor_h, int sensor_w,
                       const std::vector<uint8_t>& relative_depth, int relative_h, int relative_w,
                       std::vector<uint16_t>& output, const DepthUpscaleOptions& options = {},
                       std::string* error = nullptr);

// Matches the former service boundary: render a model depth field to the
// normalized 8-bit image that feeds upscale_depth_map.
bool normalize_depth_u8(const std::vector<float>& depth, std::vector<uint8_t>& normalized,
                        std::string* error = nullptr);

} // namespace da
