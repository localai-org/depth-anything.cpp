#pragma once

#include "image_io.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace da {

// Reads the first TIFF image directory as an 8-bit RGB image. Supports the
// compressed RGB TIFF captures used by the depth-upscale workflow.
bool load_tiff_rgb(const std::string& path, Image& out, std::string* error = nullptr);

// Reads a depth sensor TIFF whose first two 8-bit interleaved samples encode a
// big-endian unsigned 16-bit depth value.
bool load_packed_depth_tiff(const std::string& path, std::vector<uint16_t>& depth,
                            int& h, int& w, std::string* error = nullptr);

// Writes one uncompressed, single-channel, unsigned 16-bit depth TIFF.
bool write_depth_tiff_u16(const std::string& path, const std::vector<uint16_t>& depth,
                           int h, int w, std::string* error = nullptr);

} // namespace da
