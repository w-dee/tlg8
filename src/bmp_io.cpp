// Minimal BMP (24/32-bit BI_RGB) I/O
#include "image_io.h"
#include <cstdint>
#include <cstdio>
#include <vector>

#pragma pack(push, 1)
struct BMPFILEHEADER {
  uint16_t bfType;      // 'BM'
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
};

struct BMPINFOHEADER {
  uint32_t biSize;      // 40
  int32_t  biWidth;
  int32_t  biHeight;    // positive: bottom-up, negative: top-down
  uint16_t biPlanes;    // 1
  uint16_t biBitCount;  // 24 or 32
  uint32_t biCompression; // 0 = BI_RGB
  uint32_t biSizeImage; // may be 0 for BI_RGB
  int32_t  biXPelsPerMeter;
  int32_t  biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
};
#pragma pack(pop)

static inline uint32_t pad4(uint32_t x) { return (x + 3u) & ~3u; }

bool load_bmp(const std::string &path, PixelBuffer &out, std::string &err) {
  err.clear();
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) { err = "cannot open file"; return false; }

  BMPFILEHEADER fh{}; BMPINFOHEADER ih{};
  if (fread(&fh, sizeof(fh), 1, fp) != 1) { fclose(fp); err = "read error"; return false; }
  if (fh.bfType != 0x4D42) { fclose(fp); err = "not a BMP"; return false; }
  if (fread(&ih, sizeof(ih), 1, fp) != 1) { fclose(fp); err = "read error"; return false; }
  if (ih.biSize < 40) { fclose(fp); err = "unsupported DIB header"; return false; }
  if (ih.biCompression != 0) { fclose(fp); err = "compressed BMP not supported"; return false; }
  if (!(ih.biBitCount == 24 || ih.biBitCount == 32)) { fclose(fp); err = "only 24/32-bit BMP supported"; return false; }

  const bool top_down = ih.biHeight < 0;
  const uint32_t width = static_cast<uint32_t>(ih.biWidth);
  const uint32_t height = static_cast<uint32_t>(top_down ? -ih.biHeight : ih.biHeight);

  if (fseek(fp, static_cast<long>(fh.bfOffBits), SEEK_SET) != 0) { fclose(fp); err = "seek error"; return false; }

  const uint32_t src_bpp = ih.biBitCount / 8;
  const uint32_t src_stride = (ih.biBitCount == 24) ? pad4(width * 3) : width * 4;
  std::vector<uint8_t> row(src_stride);

  out.width = width;
  out.height = height;
  if (ih.biBitCount == 32) out.channels = 4; else out.channels = 3;
  out.data.resize(static_cast<size_t>(width) * height * out.channels);

  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t src_y = top_down ? y : (height - 1 - y);
    if (fread(row.data(), 1, src_stride, fp) != src_stride) { fclose(fp); err = "read error"; return false; }
    uint8_t* d = &out.data[static_cast<size_t>(src_y) * width * out.channels];
    for (uint32_t x = 0; x < width; ++x) {
      const uint8_t b = row[x*src_bpp + 0];
      const uint8_t g = row[x*src_bpp + 1];
      const uint8_t r = row[x*src_bpp + 2];
      if (out.channels == 4) {
        const uint8_t a = row[x*src_bpp + 3];
        d[x*4 + 0] = a; // ARGB
        d[x*4 + 1] = r;
        d[x*4 + 2] = g;
        d[x*4 + 3] = b;
      } else {
        d[x*3 + 0] = r; // RGB
        d[x*3 + 1] = g;
        d[x*3 + 2] = b;
      }
    }
    (void)src_y; // we always read sequentially due to fseek to bfOffBits + stride*y would be more random; row already read linearly
  }

  fclose(fp);
  return true;
}

bool save_bmp(const std::string &path, const PixelBuffer &src, std::string &err) {
  err.clear();
  if (!(src.channels == 3 || src.channels == 4)) { err = "unsupported pixel channels"; return false; }
  if (src.data.size() < static_cast<size_t>(src.width) * src.height * src.channels) { err = "pixel buffer too small"; return false; }

  const bool has_alpha = (src.channels == 4) && src.has_alpha();
  const uint16_t bits = has_alpha ? 32 : 24;
  const uint32_t stride = (bits == 24) ? pad4(src.width * 3) : src.width * 4;
  const uint32_t image_size = stride * src.height;

  BMPFILEHEADER fh{}; BMPINFOHEADER ih{};
  fh.bfType = 0x4D42; // 'BM'
  fh.bfOffBits = sizeof(BMPFILEHEADER) + sizeof(BMPINFOHEADER);
  fh.bfSize = fh.bfOffBits + image_size;
  ih.biSize = sizeof(BMPINFOHEADER);
  ih.biWidth = static_cast<int32_t>(src.width);
  ih.biHeight = static_cast<int32_t>(src.height); // bottom-up
  ih.biPlanes = 1;
  ih.biBitCount = bits;
  ih.biCompression = 0; // BI_RGB
  ih.biSizeImage = image_size;

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) { err = "cannot open file"; return false; }
  if (fwrite(&fh, sizeof(fh), 1, fp) != 1 || fwrite(&ih, sizeof(ih), 1, fp) != 1) { fclose(fp); err = "write error"; return false; }

  std::vector<uint8_t> row(stride, 0);
  for (int y = static_cast<int>(src.height) - 1; y >= 0; --y) {
    const uint8_t* s = &src.data[static_cast<size_t>(y) * src.width * src.channels];
    if (bits == 24) {
      for (uint32_t x = 0; x < src.width; ++x) {
        // write BGR from RGB or ARGB
        uint8_t r, g, b;
        if (src.channels == 4) {
          r = s[x*4 + 1]; g = s[x*4 + 2]; b = s[x*4 + 3];
        } else {
          r = s[x*3 + 0]; g = s[x*3 + 1]; b = s[x*3 + 2];
        }
        row[x*3 + 0] = b;
        row[x*3 + 1] = g;
        row[x*3 + 2] = r;
      }
    } else {
      for (uint32_t x = 0; x < src.width; ++x) {
        // our buffer may be ARGB or RGB
        uint8_t r, g, b, a;
        if (src.channels == 4) {
          a = s[x*4 + 0]; r = s[x*4 + 1]; g = s[x*4 + 2]; b = s[x*4 + 3];
        } else {
          r = s[x*3 + 0]; g = s[x*3 + 1]; b = s[x*3 + 2]; a = 255;
        }
        row[x*4 + 0] = b; row[x*4 + 1] = g; row[x*4 + 2] = r; row[x*4 + 3] = a; // BGRA
      }
    }
    if (fwrite(row.data(), 1, stride, fp) != stride) { fclose(fp); err = "write error"; return false; }
  }

  fclose(fp);
  return true;
}
