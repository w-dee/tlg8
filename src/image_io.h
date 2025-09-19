// Minimal image I/O interfaces for tlgconv
#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class ImageFormat {
  Auto,
  R8G8B8,      // 24-bit RGB
  A8R8G8B8     // 32-bit ARGB (byte order: A R G B per pixel)
};

struct PixelBuffer {
  uint32_t width = 0;
  uint32_t height = 0;
  // channels = 3 (RGB) or 4 (ARGB)
  uint32_t channels = 0;
  std::vector<uint8_t> data; // row-major, tightly packed

  bool has_alpha() const {
    if (channels != 4) return false;
    const uint8_t* p = data.data();
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
      if (p[i*4 + 0] != 255) return true; // A != 255
    }
    return false;
  }
};

// Dispatch by extension helpers
bool has_ext(const std::string &path, const char* extLowerNoDot);

// PNG I/O (system libpng)
bool load_png(const std::string &path, PixelBuffer &out, std::string &err);
bool save_png(const std::string &path, const PixelBuffer &src, std::string &err);

// BMP I/O (uncompressed 24/32-bit)
bool load_bmp(const std::string &path, PixelBuffer &out, std::string &err);
bool save_bmp(const std::string &path, const PixelBuffer &src, std::string &err);

// TLG I/O (TLG5/6) — implemented in later steps
struct TlgOptions {
  int version = 6; // 5 or 6
  ImageFormat fmt = ImageFormat::Auto; // decided by input if Auto
};

bool load_tlg(const std::string &path, PixelBuffer &out, std::string &err);
bool save_tlg(const std::string &path, const PixelBuffer &src, const TlgOptions &opt, std::string &err);

