// PNG I/O using system libpng
#include "image_io.h"
#include <png.h>
#include <cstdio>
#include <vector>

static void png_read_fn(png_structp png_ptr, png_bytep data, png_size_t length)
{
  FILE *fp = static_cast<FILE *>(png_get_io_ptr(png_ptr));
  if (fread(data, 1, length, fp) != length)
    png_error(png_ptr, "read error");
}

static void png_write_fn(png_structp png_ptr, png_bytep data, png_size_t length)
{
  FILE *fp = static_cast<FILE *>(png_get_io_ptr(png_ptr));
  if (fwrite(data, 1, length, fp) != length)
    png_error(png_ptr, "write error");
}

static void png_flush_fn(png_structp png_ptr)
{
  FILE *fp = static_cast<FILE *>(png_get_io_ptr(png_ptr));
  fflush(fp);
}

bool load_png(const std::string &path, PixelBuffer &out, std::string &err)
{
  err.clear();
  FILE *fp = fopen(path.c_str(), "rb");
  if (!fp)
  {
    err = "cannot open file";
    return false;
  }

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr)
  {
    fclose(fp);
    err = "png_create_read_struct failed";
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    fclose(fp);
    err = "png_create_info_struct failed";
    return false;
  }
  if (setjmp(png_jmpbuf(png_ptr)))
  {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);
    if (err.empty())
      err = "libpng error";
    return false;
  }

  png_set_read_fn(png_ptr, fp, png_read_fn);
  png_read_info(png_ptr, info_ptr);

  png_uint_32 w, h;
  int bit_depth, color_type;
  png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, nullptr, nullptr, nullptr);

  // Transforms to 8-bit RGB/RGBA
  if (bit_depth == 16)
    png_set_strip_16(png_ptr);
  if (color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png_ptr);
  // Ensure we have alpha channel if any transparency
  if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
  {
    png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER); // RGB -> RGBA
  }
  if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
  {
    png_set_gray_to_rgb(png_ptr);
  }

  png_read_update_info(png_ptr, info_ptr);

  png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  std::vector<png_bytep> rows(h);
  std::vector<uint8_t> buffer(rowbytes * h);
  for (png_uint_32 y = 0; y < h; ++y)
    rows[y] = buffer.data() + y * rowbytes;
  png_read_image(png_ptr, rows.data());
  png_read_end(png_ptr, nullptr);

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  fclose(fp);

  // buffer is in RGBA; convert to our ARGB (A,R,G,B) per pixel
  out.width = w;
  out.height = h;
  out.channels = 4;
  out.data.resize(static_cast<size_t>(w) * h * 4);
  const uint8_t *s = buffer.data();
  uint8_t *d = out.data.data();
  const size_t pixels = static_cast<size_t>(w) * h;
  for (size_t i = 0; i < pixels; ++i)
  {
    uint8_t r = s[i * 4 + 0];
    uint8_t g = s[i * 4 + 1];
    uint8_t b = s[i * 4 + 2];
    uint8_t a = s[i * 4 + 3];
    d[i * 4 + 0] = a;
    d[i * 4 + 1] = r;
    d[i * 4 + 2] = g;
    d[i * 4 + 3] = b;
  }
  return true;
}

bool save_png(const std::string &path, const PixelBuffer &src, std::string &err)
{
  err.clear();
  if (!(src.channels == 3 || src.channels == 4))
  {
    err = "unsupported pixel channels";
    return false;
  }
  if (src.data.size() < static_cast<size_t>(src.width) * src.height * src.channels)
  {
    err = "pixel buffer too small";
    return false;
  }

  FILE *fp = fopen(path.c_str(), "wb");
  if (!fp)
  {
    err = "cannot open file";
    return false;
  }

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr)
  {
    fclose(fp);
    err = "png_create_write_struct failed";
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    fclose(fp);
    err = "png_create_info_struct failed";
    return false;
  }
  if (setjmp(png_jmpbuf(png_ptr)))
  {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    if (err.empty())
      err = "libpng error";
    return false;
  }

  png_set_write_fn(png_ptr, fp, png_write_fn, png_flush_fn);

  int color_type = (src.channels == 4) ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB;
  png_set_IHDR(png_ptr, info_ptr, src.width, src.height, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);

  // Prepare rows as RGBA/RGB; convert from our ARGB if needed
  const size_t pixels = static_cast<size_t>(src.width) * src.height;
  std::vector<uint8_t> buffer;
  std::vector<png_bytep> rows(src.height);
  if (src.channels == 4)
  {
    buffer.resize(pixels * 4);
    for (size_t i = 0; i < pixels; ++i)
    {
      const uint8_t a = src.data[i * 4 + 0];
      const uint8_t r = src.data[i * 4 + 1];
      const uint8_t g = src.data[i * 4 + 2];
      const uint8_t b = src.data[i * 4 + 3];
      buffer[i * 4 + 0] = r;
      buffer[i * 4 + 1] = g;
      buffer[i * 4 + 2] = b;
      buffer[i * 4 + 3] = a;
    }
    for (uint32_t y = 0; y < src.height; ++y)
      rows[y] = buffer.data() + y * (src.width * 4);
  }
  else
  {
    // channels == 3, already tight RGB? Our buffer is RGB; ensure it's in that layout
    buffer = src.data; // copy
    for (uint32_t y = 0; y < src.height; ++y)
      rows[y] = buffer.data() + y * (src.width * 3);
  }

  png_write_image(png_ptr, rows.data());
  png_write_end(png_ptr, nullptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  return true;
}
