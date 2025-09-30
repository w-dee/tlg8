#include "tlg8_io.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "tlg_io_common.h"
#include "tlg8_bit_io.h"

namespace
{
  using tlg::detail::read_exact;

  bool write_bytes(FILE *fp, const void *data, size_t size)
  {
    return std::fwrite(data, 1, size, fp) == size;
  }

  bool write_u32le(FILE *fp, uint32_t value)
  {
    uint8_t buf[4];
    for (int i = 0; i < 4; ++i)
      buf[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xffu);
    return write_bytes(fp, buf, sizeof(buf));
  }

  bool write_u64le(FILE *fp, uint64_t value)
  {
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i)
      buf[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xffu);
    return write_bytes(fp, buf, sizeof(buf));
  }

  bool read_u32le(FILE *fp, uint32_t &value)
  {
    uint8_t buf[4];
    if (!read_exact(fp, buf, sizeof(buf)))
      return false;
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i)
    {
      v <<= 8;
      v |= static_cast<uint32_t>(buf[i]);
    }
    value = v;
    return true;
  }

  bool read_u64le(FILE *fp, uint64_t &value)
  {
    uint8_t buf[8];
    if (!read_exact(fp, buf, sizeof(buf)))
      return false;
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i)
    {
      v <<= 8;
      v |= static_cast<uint64_t>(buf[i]);
    }
    value = v;
    return true;
  }
}

namespace tlg::v8
{
  namespace
  {
    constexpr uint64_t TILE_SIZE = 8192; // virtually non tiling

    bool copy_pixels_to_buffer(const PixelBuffer &src,
                               int desired_colors,
                               std::vector<uint8_t> &out_data,
                               std::string &err)
    {
      if (!(src.channels == 3 || src.channels == 4))
      {
        err = "tlg8: unsupported source channels";
        return false;
      }
      const uint64_t pixel_count = static_cast<uint64_t>(src.width) * src.height;
      if (pixel_count == 0)
      {
        err = "tlg8: empty image";
        return false;
      }
      if (pixel_count > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: image too large";
        return false;
      }
      if (!(desired_colors == 3 || desired_colors == 4))
      {
        err = "tlg8: unsupported desired colors";
        return false;
      }
      if (src.channels != 0 && pixel_count > std::numeric_limits<size_t>::max() / src.channels)
      {
        err = "tlg8: image too large";
        return false;
      }
      const size_t src_min = static_cast<size_t>(pixel_count) * src.channels;
      if (src.data.size() < src_min)
      {
        err = "tlg8: source buffer too small";
        return false;
      }
      const uint64_t total_bytes = pixel_count * static_cast<uint64_t>(desired_colors);
      if (total_bytes > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: image too large";
        return false;
      }
      out_data.resize(static_cast<size_t>(total_bytes));
      size_t dst_index = 0;
      const uint8_t *src_ptr = src.data.data();
      for (uint64_t i = 0; i < pixel_count; ++i)
      {
        if (desired_colors == 3)
        {
          uint8_t r, g, b;
          if (src.channels == 4)
          {
            r = src_ptr[i * 4 + 1];
            g = src_ptr[i * 4 + 2];
            b = src_ptr[i * 4 + 3];
          }
          else
          {
            r = src_ptr[i * 3 + 0];
            g = src_ptr[i * 3 + 1];
            b = src_ptr[i * 3 + 2];
          }
          out_data[dst_index++] = r;
          out_data[dst_index++] = g;
          out_data[dst_index++] = b;
        }
        else
        {
          uint8_t a, r, g, b;
          if (src.channels == 4)
          {
            a = src_ptr[i * 4 + 0];
            r = src_ptr[i * 4 + 1];
            g = src_ptr[i * 4 + 2];
            b = src_ptr[i * 4 + 3];
          }
          else
          {
            a = 255;
            r = src_ptr[i * 3 + 0];
            g = src_ptr[i * 3 + 1];
            b = src_ptr[i * 3 + 2];
          }
          out_data[dst_index++] = a;
          out_data[dst_index++] = r;
          out_data[dst_index++] = g;
          out_data[dst_index++] = b;
        }
      }
      return true;
    }
  }

  bool decode_for_tile(detail::bitio::BitReader &reader,
                       uint32_t tile_w,
                       uint32_t tile_h,
                       uint32_t components,
                       uint32_t origin_x,
                       uint32_t origin_y,
                       uint32_t image_width,
                       std::vector<uint8_t> &decoded,
                       std::string &err);

  bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err)
  {
    err.clear();

    uint8_t colors = 0;
    if (!read_exact(fp, &colors, 1))
    {
      err = "tlg8: failed to read color count";
      return false;
    }

    uint8_t header_bytes[5];
    if (!read_exact(fp, header_bytes, sizeof(header_bytes)))
    {
      err = "tlg8: failed to read header flags";
      return false;
    }

    uint64_t tile_w = 0;
    uint64_t tile_h = 0;
    uint64_t width64 = 0;
    uint64_t height64 = 0;
    if (!read_u64le(fp, tile_w) || !read_u64le(fp, tile_h) || !read_u64le(fp, width64) || !read_u64le(fp, height64))
    {
      err = "tlg8: failed to read dimensions";
      return false;
    }

    if (!(colors == 3 || colors == 4))
    {
      err = "tlg8: unsupported color count";
      return false;
    }
    if (width64 == 0 || height64 == 0)
    {
      err = "tlg8: invalid image size";
      return false;
    }
    if (width64 > std::numeric_limits<uint32_t>::max() || height64 > std::numeric_limits<uint32_t>::max())
    {
      err = "tlg8: image size exceeds 32-bit limit";
      return false;
    }

    const uint64_t pixel_count = width64 * height64;
    if (pixel_count > std::numeric_limits<size_t>::max())
    {
      err = "tlg8: pixel count too large";
      return false;
    }
    const uint64_t total_bytes = pixel_count * colors;
    if (total_bytes > std::numeric_limits<size_t>::max())
    {
      err = "tlg8: payload too large";
      return false;
    }

    if (tile_w == 0 || tile_h == 0)
    {
      err = "tlg8: invalid tile size";
      return false;
    }
    if (tile_w > std::numeric_limits<uint32_t>::max() || tile_h > std::numeric_limits<uint32_t>::max())
    {
      err = "tlg8: tile size exceeds 32-bit limit";
      return false;
    }

    const uint32_t width = static_cast<uint32_t>(width64);
    const uint32_t height = static_cast<uint32_t>(height64);
    const uint32_t tile_width = static_cast<uint32_t>(tile_w);
    const uint32_t tile_height = static_cast<uint32_t>(tile_h);
    const uint32_t components = colors;

    std::vector<uint8_t> decoded(static_cast<size_t>(total_bytes));
    std::vector<uint8_t> tile_buffer;

    for (uint32_t origin_y = 0; origin_y < height; origin_y += tile_height)
    {
      const uint32_t tile_h = std::min<uint32_t>(tile_height, height - origin_y);
      for (uint32_t origin_x = 0; origin_x < width; origin_x += tile_width)
      {
        const uint32_t tile_w = std::min<uint32_t>(tile_width, width - origin_x);
        uint32_t tile_size_u32 = 0;
        if (!read_u32le(fp, tile_size_u32))
        {
          err = "tlg8: failed to read tile size";
          return false;
        }
        const size_t tile_size = static_cast<size_t>(tile_size_u32);
        tile_buffer.resize(tile_size);
        if (tile_size > 0 && !read_exact(fp, tile_buffer.data(), tile_size))
        {
          err = "tlg8: failed to read tile payload";
          return false;
        }
        detail::bitio::BitReader reader(tile_buffer.data(), tile_size);
        if (!decode_for_tile(reader,
                             tile_w,
                             tile_h,
                             components,
                             origin_x,
                             origin_y,
                             width,
                             decoded,
                             err))
          return false;
      }
    }

    out.width = width;
    out.height = height;
    out.channels = components;
    out.data = std::move(decoded);

    (void)header_bytes;

    return true;
  }

  namespace enc
  {
  bool encode_for_tile(detail::bitio::BitWriter &writer,
                       const uint8_t *image_base,
                       uint32_t image_width,
                       uint32_t components,
                       uint32_t origin_x,
                       uint32_t origin_y,
                       uint32_t tile_w,
                       uint32_t tile_h,
                       FILE *dump_fp,
                       bool dump_before_hilbert,
                       bool dump_after_hilbert,
                       std::string &err);

    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int desired_colors,
                   const std::string &dump_residuals_path,
                   TlgOptions::DumpResidualsOrder dump_residuals_order,
                   std::string &err)
    {
      static_assert(std::numeric_limits<int8_t>::min() == -128 &&
                        std::numeric_limits<int8_t>::max() == 127,
                    "int8_t is not two's complement");

      err.clear();

      struct FileCloser
      {
        void operator()(FILE *p) const noexcept
        {
          if (p)
            std::fclose(p);
        }
      };

      std::unique_ptr<FILE, FileCloser> dump_file;
      if (!dump_residuals_path.empty())
      {
        FILE *dump_fp = std::fopen(dump_residuals_path.c_str(), "w");
        if (!dump_fp)
        {
          err = "tlg8: 残差ダンプファイルを開けません: " + dump_residuals_path;
          return false;
        }
        dump_file.reset(dump_fp);
      }

      if (src.width == 0 || src.height == 0)
      {
        err = "tlg8: empty source image";
        return false;
      }
      if (!(desired_colors == 3 || desired_colors == 4))
      {
        err = "tlg8: unsupported desired colors";
        return false;
      }

      std::vector<uint8_t> packed;
      if (!copy_pixels_to_buffer(src, desired_colors, packed, err))
        return false;

      const unsigned char mark[11] = {'T', 'L', 'G', '8', '.', '0', 0, 'r', 'a', 'w', 0x1a};
      if (!write_bytes(fp, mark, sizeof(mark)))
      {
        err = "tlg8: failed to write magic";
        return false;
      }

      uint8_t colors_u8 = static_cast<uint8_t>(desired_colors);
      if (!write_bytes(fp, &colors_u8, 1))
      {
        err = "tlg8: failed to write color count";
        return false;
      }

      const uint8_t meta[5] = {0x01, 0x00, 0x00, 0x00, 0x08};
      if (!write_bytes(fp, meta, sizeof(meta)))
      {
        err = "tlg8: failed to write header flags";
        return false;
      }

      const uint64_t tile_side = TILE_SIZE;
      if (tile_side == 0 || tile_side > std::numeric_limits<uint32_t>::max())
      {
        err = "tlg8: invalid tile size";
        return false;
      }
      if (!write_u64le(fp, tile_side) || !write_u64le(fp, tile_side) ||
          !write_u64le(fp, src.width) || !write_u64le(fp, src.height))
      {
        err = "tlg8: failed to write dimensions";
        return false;
      }

      const uint32_t width = src.width;
      const uint32_t height = src.height;
      const uint32_t tile_width = static_cast<uint32_t>(tile_side);
      const uint32_t tile_height = static_cast<uint32_t>(tile_side);
      const uint32_t components = static_cast<uint32_t>(desired_colors);
      const uint64_t tile_capacity_u64 = static_cast<uint64_t>(tile_width) * tile_height * 4u * 2u;
      if (tile_capacity_u64 == 0 || tile_capacity_u64 > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: tile buffer size overflow";
        return false;
      }
      std::vector<uint8_t> tile_buffer(static_cast<size_t>(tile_capacity_u64));
      const uint8_t *packed_ptr = packed.data();

      const bool dump_before_hilbert = dump_file &&
                                       (dump_residuals_order == TlgOptions::DumpResidualsOrder::BeforeHilbert);
      const bool dump_after_hilbert = dump_file &&
                                      (dump_residuals_order == TlgOptions::DumpResidualsOrder::AfterHilbert);

      for (uint32_t origin_y = 0; origin_y < height; origin_y += tile_height)
      {
        const uint32_t tile_h = std::min<uint32_t>(tile_height, height - origin_y);
        for (uint32_t origin_x = 0; origin_x < width; origin_x += tile_width)
        {
          const uint32_t tile_w = std::min<uint32_t>(tile_width, width - origin_x);
          detail::bitio::BitWriter writer(tile_buffer.data(), tile_buffer.size());
          if (!encode_for_tile(writer,
                               packed_ptr,
                               width,
                               components,
                               origin_x,
                               origin_y,
                               tile_w,
                               tile_h,
                               dump_file.get(),
                               dump_before_hilbert,
                               dump_after_hilbert,
                               err))
            return false;
          if (!writer.align_to_u32_zero() || !writer.finish())
          {
            err = "tlg8: failed to finalize tile payload";
            return false;
          }
          const size_t tile_bytes = writer.bytes_written();
          if (tile_bytes > std::numeric_limits<uint32_t>::max())
          {
            err = "tlg8: tile payload too large";
            return false;
          }
          if (!write_u32le(fp, static_cast<uint32_t>(tile_bytes)))
          {
            err = "tlg8: failed to write tile size";
            return false;
          }
          if (tile_bytes > tile_buffer.size())
          {
            err = "tlg8: tile size exceeds buffer";
            return false;
          }
          if (tile_bytes > 0 && !write_bytes(fp, tile_buffer.data(), tile_bytes))
          {
            err = "tlg8: failed to write tile payload";
            return false;
          }
        }
      }

      return true;
    }
  } // namespace enc
} // namespace tlg::v8
