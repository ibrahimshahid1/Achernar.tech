#pragma once

// Shared application-level LibRaw unpack contract. See the A04 protocol and
// local Collins (2017) PDF pp5/11 and Stetson (1987) p7. LibRaw/camera-level
// transformations may precede this buffer; we add no calibration or scaling.
#include <libraw/libraw.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ols {
namespace untouched_raw {
constexpr const char *version = "achernar-libraw-unpack-v1";

struct Crop {
    int width, height, x, y;
    size_t source_pitch;
};

inline Crop crop(int raw_width, int raw_height, int visible_width,
                 int visible_height, int left, int top, size_t pitch,
                 int width = -1, int height = -1, int bin = 1) {
    if (bin != 1)
        throw std::runtime_error("Untouched RAW unpack supports bin=1 only; bin after calibration");
    if (raw_width <= 0 || raw_height <= 0 || visible_width <= 0 || visible_height <= 0 ||
        left < 0 || top < 0 || int64_t(left) + visible_width > raw_width || int64_t(top) + visible_height > raw_height)
        throw std::runtime_error("Invalid LibRaw raw/visible geometry");
    if (width == -1 && height == -1) {
        width = visible_width / 2 * 2;
        height = visible_height / 2 * 2;
    }
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1) ||
        width > visible_width || height > visible_height)
        throw std::runtime_error("Requested RAW region must be positive, even and inside the visible area");
    const int x = left + (visible_width - width) / 4 * 2;
    const int y = top + (visible_height - height) / 4 * 2;
    const size_t row_bytes = static_cast<size_t>(raw_width) * sizeof(uint16_t);
    if (!pitch) pitch = row_bytes;
    if (pitch < row_bytes || static_cast<size_t>(raw_height) > std::numeric_limits<size_t>::max() / pitch)
        throw std::runtime_error("Invalid LibRaw raw byte stride");
    return {width, height, x, y, pitch};
}

inline std::vector<uint8_t> copy(const uint8_t *source, const Crop &region) {
    if (!source || region.width <= 0 || region.height <= 0 || region.x < 0 || region.y < 0)
        throw std::runtime_error("Invalid RAW source/crop");
    const size_t row_bytes = static_cast<size_t>(region.width) * sizeof(uint16_t);
    if (static_cast<size_t>(region.height) > std::numeric_limits<size_t>::max() / row_bytes ||
        (static_cast<size_t>(region.x) * 2 + row_bytes) > region.source_pitch)
        throw std::runtime_error("Invalid RAW copy dimensions");
    std::vector<uint8_t> pixels(row_bytes * static_cast<size_t>(region.height));
    for (int y = 0; y < region.height; ++y)
        std::memcpy(pixels.data() + static_cast<size_t>(y) * row_bytes,
                    source + (static_cast<size_t>(region.y) + y) * region.source_pitch + static_cast<size_t>(region.x) * 2,
                    row_bytes);
    return pixels;
}

inline int channel(unsigned filters, int y, int x) {
    return (filters >> ((((y << 1) & 14) | (x & 1)) << 1)) & 3;
}

// Standard 2x2 RGB Bayer only. Do not invent RGB identities for CMYG,
// X-Trans, Leaf's 16x16 array, or an unknown color layout.
inline std::string bayer(unsigned filters, const char *colors, int x, int y) {
    if (filters < 1000 || !colors || colors[0] != 'R' || colors[1] != 'G' || colors[2] != 'B')
        return "";
    auto color = [&](int row, int col) {
        const int c = channel(filters, row, col);
        return c == 3 ? (colors[3] == 'G' ? 'G' : '?') : colors[c];
    };
    std::string pattern;
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col)
            pattern += color(y + row, x + col);
    if (pattern != "RGGB" && pattern != "BGGR" && pattern != "GRBG" && pattern != "GBRG")
        return "";
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 2; ++col)
            if (color(y + row, x + col) != pattern[(row % 2) * 2 + col]) return "";
    return pattern;
}

struct Plane {
    Crop region;
    std::vector<uint8_t> pixels;
    std::string pattern;
};

inline std::vector<unsigned> black_table(const LibRaw &raw);

inline Plane unpack(LibRaw &raw, int width = -1, int height = -1, int bin = 1) {
    if (bin != 1) throw std::runtime_error("Untouched RAW unpack supports bin=1 only; bin after calibration");
    const int code = raw.unpack();
    if (code != LIBRAW_SUCCESS)
        throw std::runtime_error(std::string("LibRaw unpack failed: ") + libraw_strerror(code));
    if (!raw.imgdata.rawdata.raw_image)
        throw std::runtime_error("File has no supported unpacked 16-bit single-component RAW plane");
    if (raw.imgdata.idata.filters && raw.imgdata.idata.filters < 1000)
        throw std::runtime_error("Nonstandard CFA (including X-Trans and Leaf) is unsupported");
    (void)black_table(raw);
    const auto &s = raw.imgdata.sizes;
    const Crop region = crop(s.raw_width, s.raw_height, s.width, s.height,
                             s.left_margin, s.top_margin, s.raw_pitch, width, height, bin);
    Plane result{region, copy(reinterpret_cast<const uint8_t *>(raw.imgdata.rawdata.raw_image), region),
                 bayer(raw.imgdata.idata.filters, raw.imgdata.idata.cdesc, region.x, region.y)};
    return result;
}

inline std::vector<unsigned> black_table(const LibRaw &raw) {
    const auto &black = raw.imgdata.color.cblack;
    const size_t rows = black[4], cols = black[5];
    const size_t capacity = sizeof(black) / sizeof(black[0]) - 6;
    if ((!rows != !cols) || (rows && cols > capacity / rows))
        throw std::runtime_error("Malformed LibRaw spatial black table dimensions");
    return std::vector<unsigned>(black + 6, black + 6 + rows * cols);
}
} // namespace untouched_raw
} // namespace ols
