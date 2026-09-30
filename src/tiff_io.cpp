#include "tiff_io.hpp"

#include <tiffio.h>
#include <limits>

namespace da {
namespace {

void set_error(std::string* error, const char* message) {
    if (error) *error = message;
}

bool dimensions_fit_int(uint32_t width, uint32_t height) {
    return width > 0 && height > 0 &&
           width <= static_cast<uint32_t>(std::numeric_limits<int>::max()) &&
           height <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

} // namespace

bool load_tiff_rgb(const std::string& path, Image& out, std::string* error) {
    TIFF* tiff = TIFFOpen(path.c_str(), "r");
    if (!tiff) {
        set_error(error, "cannot open TIFF");
        return false;
    }

    uint32_t width = 0, height = 0;
    const bool valid = TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) &&
                       TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) &&
                       dimensions_fit_int(width, height);
    if (!valid || static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height) {
        TIFFClose(tiff);
        set_error(error, "invalid TIFF dimensions");
        return false;
    }

    std::vector<uint32_t> rgba(static_cast<size_t>(width) * height);
    if (!TIFFReadRGBAImageOriented(tiff, width, height, rgba.data(), ORIENTATION_TOPLEFT, 0)) {
        TIFFClose(tiff);
        set_error(error, "failed to decode TIFF pixels");
        return false;
    }
    TIFFClose(tiff);

    out.w = static_cast<int>(width);
    out.h = static_cast<int>(height);
    out.rgb.resize(rgba.size() * 3);
    for (size_t i = 0; i < rgba.size(); ++i) {
        out.rgb[3 * i + 0] = TIFFGetR(rgba[i]);
        out.rgb[3 * i + 1] = TIFFGetG(rgba[i]);
        out.rgb[3 * i + 2] = TIFFGetB(rgba[i]);
    }
    return true;
}

bool load_packed_depth_tiff(const std::string& path, std::vector<uint16_t>& depth,
                            int& h, int& w, std::string* error) {
    TIFF* tiff = TIFFOpen(path.c_str(), "r");
    if (!tiff) {
        set_error(error, "cannot open TIFF");
        return false;
    }

    uint32_t width = 0, height = 0;
    uint16_t bits = 0, samples = 0, planar = PLANARCONFIG_CONTIG;
    const bool valid = TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) &&
                       TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) &&
                       TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits) &&
                       TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples) &&
                       TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar) &&
                       dimensions_fit_int(width, height) && bits == 8 && samples >= 2 &&
                       planar == PLANARCONFIG_CONTIG;
    if (!valid || static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height) {
        TIFFClose(tiff);
        set_error(error, "expected an interleaved two-channel 8-bit depth TIFF");
        return false;
    }

    const tsize_t scanline_size = TIFFScanlineSize(tiff);
    if (scanline_size < static_cast<tsize_t>(width) * samples) {
        TIFFClose(tiff);
        set_error(error, "invalid TIFF scanline size");
        return false;
    }
    std::vector<uint8_t> row(static_cast<size_t>(scanline_size));
    depth.resize(static_cast<size_t>(width) * height);
    for (uint32_t y = 0; y < height; ++y) {
        if (TIFFReadScanline(tiff, row.data(), y, 0) < 0) {
            TIFFClose(tiff);
            depth.clear();
            set_error(error, "failed to decode depth TIFF scanline");
            return false;
        }
        for (uint32_t x = 0; x < width; ++x) {
            const size_t src = static_cast<size_t>(x) * samples;
            depth[static_cast<size_t>(y) * width + x] =
                static_cast<uint16_t>((static_cast<uint16_t>(row[src]) << 8) | row[src + 1]);
        }
    }
    TIFFClose(tiff);
    h = static_cast<int>(height);
    w = static_cast<int>(width);
    return true;
}

bool write_depth_tiff_u16(const std::string& path, const std::vector<uint16_t>& depth,
                           int h, int w, std::string* error) {
    if (h <= 0 || w <= 0 || depth.size() != static_cast<size_t>(h) * w) {
        set_error(error, "depth dimensions do not match pixel count");
        return false;
    }
    TIFF* tiff = TIFFOpen(path.c_str(), "w");
    if (!tiff) {
        set_error(error, "cannot create TIFF");
        return false;
    }
    const bool configured = TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(w)) &&
                            TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(h)) &&
                            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16) &&
                            TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1) &&
                            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT) &&
                            TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
                            TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
                            TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE) &&
                            TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, static_cast<uint32_t>(h));
    if (!configured) {
        TIFFClose(tiff);
        set_error(error, "failed to configure output TIFF");
        return false;
    }
    for (int y = 0; y < h; ++y) {
        if (TIFFWriteScanline(tiff, const_cast<uint16_t*>(depth.data() + static_cast<size_t>(y) * w), y, 0) < 0) {
            TIFFClose(tiff);
            set_error(error, "failed to write depth TIFF scanline");
            return false;
        }
    }
    TIFFClose(tiff);
    return true;
}

} // namespace da
