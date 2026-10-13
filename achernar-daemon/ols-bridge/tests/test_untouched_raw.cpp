// Production wrapper and bridge, with no camera opened or triggered.
#include "../src/ols_bridge.cpp"
#include "libraw_wrapper.h"
#include <iostream>

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> static void refuses(F f) {
  bool failed = false;
  try { f(); } catch (const std::exception &) { failed = true; }
  require(failed, "Expected an explicit refusal");
}

static void synthetic() {
  using namespace ols::untouched_raw;
  const uint16_t values[] = {0, 2047, 2048, 2049, 16382, 16383, 16384, 65535};
  std::vector<uint8_t> source(20 * 8, 0xcc);
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      std::memcpy(source.data() + y * 20 + x * 2, &values[(x + y) % 8], 2);
  for (int left = 0; left < 2; ++left)
    for (int top = 0; top < 2; ++top) {
      const auto region = crop(8, 8, 6, 6, left, top, 20, 4, 4, 1);
      const auto pixels = copy(source.data(), region);
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
          require(!std::memcmp(pixels.data() + (y * 4 + x) * 2,
                    &values[(x + left + y + top) % 8], 2), "Unpack copy changed a detector code");
    }
  for (const std::string pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}) {
    unsigned filters = 0;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 2; ++x) {
        const char c = pattern[(y % 2) * 2 + x];
        filters |= unsigned(c == 'R' ? 0 : c == 'B' ? 2 : 1) << ((y * 2 + x) * 2);
      }
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        const auto shifted = bayer(filters, "RGBG", x, y);
        require(shifted.size() == 4, "Missing standard CFA phase");
        for (int yy = 0; yy < 2; ++yy)
          for (int xx = 0; xx < 2; ++xx)
            require(shifted[yy * 2 + xx] == pattern[((yy + y) % 2) * 2 + (xx + x) % 2], "Odd origin changed CFA phase");
      }
  }
  require(bayer(9, "RGBG", 0, 0).empty(), "X-Trans cannot be Bayer");
  refuses([] { crop(8, 8, 6, 6, 1, 1, 20, 4, 4, 2); });
  refuses([] { crop(8, 8, 6, 6, 1, 1, 8); });
  refuses([] { crop(8, 8, 6, 6, 1, 1, 20, 8, 8); });
  refuses([] { crop(8, 8, 6, 6, 1, 1, 20, 3, 4); });
  refuses([] { crop(8, 8, 6, 6, 1, 1, 20, -1, 4); });
  LibRaw raw;
  raw.imgdata.color.cblack[4] = 2;
  raw.imgdata.color.cblack[5] = 2;
  for (int i = 0; i < 4; ++i) raw.imgdata.color.cblack[6 + i] = 10 + i;
  require(black_table(raw) == std::vector<unsigned>({10, 11, 12, 13}), "Spatial black table was lost");
  raw.imgdata.color.cblack[4] = 10000;
  refuses([&] { black_table(raw); });
  OlsRawImage *image = nullptr;
  const uint8_t junk[] = {1, 2, 3, 4};
  require(ols_decode_raw(junk, sizeof(junk), -1, -1, 1, &image) != OLS_OK && !image, "Corrupt RAW was accepted");
  require(ols_decode_raw(junk, sizeof(junk), -1, -1, 2, &image) != OLS_OK && !image, "RAW binning was accepted");
}

static void real_file(const char *path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "Cannot read requested RAW fixture");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  for (int size : {-1, 128}) {
    auto legacy = ols::load_libraw(reinterpret_cast<char *>(bytes.data()), bytes.size(), size, size, 1);
    OlsRawImage *handle = nullptr;
    require(ols_decode_raw(bytes.data(), bytes.size(), size, size, 1, &handle) == OLS_OK, ols_last_error());
    std::unique_ptr<OlsRawImage_t> image(handle);
    OlsRawImageInfo info{};
    require(ols_rawimage_info(handle, &info) == OLS_OK, "Missing RAW info");
    size_t length = 0;
    const auto *pixels = ols_rawimage_pixels(handle, &length);
    require(legacy.first.isContinuous() && legacy.first.cols == info.width && legacy.first.rows == info.height, "Route dimensions differ");
    require(length == legacy.first.total() * 2 && !std::memcmp(pixels, legacy.first.data, length), "Legacy and dedicated RAW pixels differ");
    auto metadata = nlohmann::json::parse(ols_rawimage_metadata_json(handle));
    require(metadata["schema_version"] == 1 && metadata["black_table_complete"] == true, "Incomplete RAW metadata");
    require(metadata["width"] == info.width && metadata["height"] == info.height, "Metadata dimensions differ");
    require(metadata["black_table"].size() == size_t(info.cblack_pattern[0]) * size_t(info.cblack_pattern[1]), "Incomplete black table");
    const auto pattern = metadata["bayer_at_crop_origin"].get<std::string>();
    require(legacy.second == (pattern.empty() ? ols::bayer_na : ols::bayer_type_from_str(pattern)), "Route CFA phases differ");
    std::cout << "real RAW byte equality " << info.width << "x" << info.height << ", " << length << " bytes\n";
  }
}

int main(int argc, char **argv) {
  try {
    synthetic();
    if (argc > 1) real_file(argv[1]);
    std::cout << "Untouched RAW native checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
