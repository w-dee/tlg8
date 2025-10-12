#include "tlg8_io.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "tlg_io_common.h"
#include "tlg8_bit_io.h"
#include "tlg8_entropy.h"

namespace
{
  using tlg::detail::read_exact;

  constexpr uint8_t kHeaderFlagHasGolombTable = 0x01;
  constexpr std::size_t kSerializedGolombTableSize =
      static_cast<std::size_t>(tlg::v8::enc::kGolombRowCount) * tlg::v8::enc::kGolombColumnCount * sizeof(uint16_t);

  bool write_bytes(FILE *fp, const void *data, size_t size)
  {
    return std::fwrite(data, 1, size, fp) == size;
  }

  template <typename T>
  bool write_little_endian(FILE *fp, T value)
  {
    static_assert(std::is_integral_v<T>, "write_little_endian は整数型専用です");
    using UnsignedT = std::make_unsigned_t<T>;
    std::array<uint8_t, sizeof(T)> buf{};
    UnsignedT v = static_cast<UnsignedT>(value);
    for (std::size_t i = 0; i < buf.size(); ++i)
      buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xffu);
    return write_bytes(fp, buf.data(), buf.size());
  }

  template <typename T>
  bool read_little_endian(FILE *fp, T &value)
  {
    static_assert(std::is_integral_v<T>, "read_little_endian は整数型専用です");
    std::array<uint8_t, sizeof(T)> buf{};
    if (!read_exact(fp, buf.data(), buf.size()))
      return false;
    using UnsignedT = std::make_unsigned_t<T>;
    UnsignedT v = 0;
    for (std::size_t i = buf.size(); i-- > 0;)
    {
      v <<= 8;
      v |= static_cast<UnsignedT>(buf[i]);
    }
    value = static_cast<T>(v);
    return true;
  }

  bool write_u32le(FILE *fp, uint32_t value)
  {
    return write_little_endian(fp, value);
  }

  bool write_u64le(FILE *fp, uint64_t value)
  {
    return write_little_endian(fp, value);
  }

  bool read_u32le(FILE *fp, uint32_t &value)
  {
    return read_little_endian(fp, value);
  }

  bool read_u64le(FILE *fp, uint64_t &value)
  {
    return read_little_endian(fp, value);
  }
}

namespace tlg::v8
{
  namespace
  {
    constexpr uint64_t TILE_WIDTH = 8192;
    constexpr uint64_t TILE_HEIGHT = 80;

    // コピー処理の前提条件を検証するヘルパー。
    bool validate_copy_params(const PixelBuffer &src,
                              int desired_colors,
                              std::string &err,
                              uint64_t &pixel_count,
                              size_t &dst_required)
    {
      if (!(src.channels == 3 || src.channels == 4))
      {
        err = "tlg8: unsupported source channels";
        return false;
      }

      pixel_count = static_cast<uint64_t>(src.width) * src.height;
      if (pixel_count == 0)
      {
        err = "tlg8: empty image";
        return false;
      }

      if (!(desired_colors == 3 || desired_colors == 4))
      {
        err = "tlg8: unsupported desired colors";
        return false;
      }

      if (pixel_count > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: image too large";
        return false;
      }

      if (pixel_count > std::numeric_limits<size_t>::max() / src.channels)
      {
        err = "tlg8: image too large";
        return false;
      }

      const size_t src_required = static_cast<size_t>(pixel_count) * src.channels;
      if (src.data.size() < src_required)
      {
        err = "tlg8: source buffer too small";
        return false;
      }

      const uint64_t dst_bytes = pixel_count * static_cast<uint64_t>(desired_colors);
      if (dst_bytes > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: image too large";
        return false;
      }

      dst_required = static_cast<size_t>(dst_bytes);
      return true;
    }

    bool copy_pixels_to_buffer(const PixelBuffer &src,
                               int desired_colors,
                               std::vector<uint8_t> &out_data,
                               std::string &err)
    {
      uint64_t pixel_count = 0;
      size_t dst_required = 0;
      if (!validate_copy_params(src, desired_colors, err, pixel_count, dst_required))
        return false;

      out_data.resize(dst_required);

      const uint8_t *src_ptr = src.data.data();
      uint8_t *dst_ptr = out_data.data();
      const uint32_t src_channels = src.channels;
      const uint32_t desired = static_cast<uint32_t>(desired_colors);

      if (desired == src_channels)
      {
        std::copy_n(src_ptr, dst_required, dst_ptr);
        return true;
      }

      if (desired == 3)
      {
        for (uint64_t i = 0; i < pixel_count; ++i)
        {
          const uint8_t *rgba = src_ptr + i * src_channels;
          // ARGB から RGB へ変換 (アルファは破棄)
          dst_ptr[0] = rgba[1];
          dst_ptr[1] = rgba[2];
          dst_ptr[2] = rgba[3];
          dst_ptr += 3;
        }
        return true;
      }

      for (uint64_t i = 0; i < pixel_count; ++i)
      {
        const uint8_t *rgb = src_ptr + i * src_channels;
        // RGB から ARGB へ変換 (アルファは 255 固定)
        dst_ptr[0] = 255;
        dst_ptr[1] = rgb[0];
        dst_ptr[2] = rgb[1];
        dst_ptr[3] = rgb[2];
        dst_ptr += 4;
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

    if (enc::is_golomb_table_overridden())
    {
      err = "tlg8: --tlg8-golomb-table オプションはデコード時には使用できません";
      return false;
    }

    {
      std::string reset_err;
      if (!configure_golomb_table("", reset_err))
      {
        err = reset_err.empty() ? "tlg8: ゴロムテーブルを初期化できません" : reset_err;
        return false;
      }
    }

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

    const bool has_golomb_table = (header_bytes[0] & kHeaderFlagHasGolombTable) != 0;
    const uint32_t serialized_table_size = static_cast<uint32_t>(header_bytes[1]) |
                                           (static_cast<uint32_t>(header_bytes[2]) << 8) |
                                           (static_cast<uint32_t>(header_bytes[3]) << 16);

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

    if (!has_golomb_table)
    {
      if (serialized_table_size != 0)
      {
        err = "tlg8: ゴロムテーブル長が不正です";
        return false;
      }
    }
    else
    {
      if (serialized_table_size != static_cast<uint32_t>(kSerializedGolombTableSize))
      {
        err = "tlg8: ゴロムテーブルのサイズが不正です";
        return false;
      }
      std::array<uint8_t, kSerializedGolombTableSize> table_bytes{};
      if (!read_exact(fp, table_bytes.data(), table_bytes.size()))
      {
        err = "tlg8: ゴロムテーブルを読み取れません";
        return false;
      }
      enc::golomb_table_counts table{};
      std::size_t offset = 0;
      for (std::size_t row = 0; row < table.size(); ++row)
      {
        uint32_t row_sum = 0;
        for (std::size_t col = 0; col < table[row].size(); ++col)
        {
          const uint16_t low = table_bytes[offset];
          const uint16_t high = table_bytes[offset + 1];
          const uint16_t value = static_cast<uint16_t>(low | static_cast<uint16_t>(high << 8));
          table[row][col] = value;
          row_sum += value;
          offset += 2;
        }
        if (row_sum != enc::kGolombRowSum)
        {
          err = "tlg8: ゴロムテーブルの行合計が不正です";
          return false;
        }
      }
      (void)enc::apply_golomb_table(table);
    }

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
                       TlgOptions::DumpResidualsOrder dump_order,
                       PixelBuffer *residual_bitmap,
                       TlgOptions::DumpResidualsOrder residual_bitmap_order,
                       double residual_bitmap_emphasis,
                       std::string &err);

    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int desired_colors,
                   const std::string &dump_residuals_path,
                   TlgOptions::DumpResidualsOrder dump_residuals_order,
                   const std::string &dump_golomb_prediction_path,
                   const std::string &residual_bmp_path,
                   TlgOptions::DumpResidualsOrder residual_bmp_order,
                   double residual_bmp_emphasis,
                   std::string &err,
                   uint64_t *out_entropy_bits)
    {
      static_assert(std::numeric_limits<int8_t>::min() == -128 &&
                        std::numeric_limits<int8_t>::max() == 127,
                    "int8_t is not two's complement");

      err.clear();

      if (out_entropy_bits)
        *out_entropy_bits = 0;

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

      std::unique_ptr<FILE, FileCloser> golomb_prediction_file;
      if (!dump_golomb_prediction_path.empty())
      {
        FILE *prediction_fp = std::fopen(dump_golomb_prediction_path.c_str(), "w");
        if (!prediction_fp)
        {
          err = "tlg8: ゴロム予測ダンプファイルを開けません: " + dump_golomb_prediction_path;
          return false;
        }
        golomb_prediction_file.reset(prediction_fp);
      }

      tlg::v8::enc::set_golomb_prediction_dump_file(golomb_prediction_file.get());
      struct PredictionDumpGuard
      {
        ~PredictionDumpGuard()
        {
          tlg::v8::enc::set_golomb_prediction_dump_file(nullptr);
        }
      } prediction_dump_guard;

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

      PixelBuffer residual_bitmap;
      PixelBuffer *residual_bitmap_ptr = nullptr;
      TlgOptions::DumpResidualsOrder effective_bitmap_order = residual_bmp_order;
      const double residual_bitmap_emphasis = residual_bmp_emphasis;

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

      constexpr uint32_t kSerializedGolombTableSizeU32 = static_cast<uint32_t>(kSerializedGolombTableSize);
      uint8_t meta[5] = {
          kHeaderFlagHasGolombTable,
          static_cast<uint8_t>(kSerializedGolombTableSizeU32 & 0xFFu),
          static_cast<uint8_t>((kSerializedGolombTableSizeU32 >> 8) & 0xFFu),
          static_cast<uint8_t>((kSerializedGolombTableSizeU32 >> 16) & 0xFFu),
          0x08};
      if (!write_bytes(fp, meta, sizeof(meta)))
      {
        err = "tlg8: failed to write header flags";
        return false;
      }

      const uint64_t tile_width_u64 = TILE_WIDTH;
      const uint64_t tile_height_u64 = TILE_HEIGHT;
      if (tile_width_u64 == 0 || tile_width_u64 > std::numeric_limits<uint32_t>::max() ||
          tile_height_u64 == 0 || tile_height_u64 > std::numeric_limits<uint32_t>::max())
      {
        err = "tlg8: invalid tile size";
        return false;
      }
      if (!write_u64le(fp, tile_width_u64) || !write_u64le(fp, tile_height_u64) ||
          !write_u64le(fp, src.width) || !write_u64le(fp, src.height))
      {
        err = "tlg8: failed to write dimensions";
        return false;
      }

      const auto &golomb_table = current_golomb_table();
      std::array<uint8_t, kSerializedGolombTableSize> table_bytes{};
      std::size_t table_offset = 0;
      for (const auto &row : golomb_table)
      {
        for (uint16_t value : row)
        {
          table_bytes[table_offset++] = static_cast<uint8_t>(value & 0xFFu);
          table_bytes[table_offset++] = static_cast<uint8_t>((value >> 8) & 0xFFu);
        }
      }
      if (!write_bytes(fp, table_bytes.data(), table_bytes.size()))
      {
        err = "tlg8: ゴロムテーブルを書き出せません";
        return false;
      }

      const uint32_t width = src.width;
      const uint32_t height = src.height;
      const uint32_t tile_width = static_cast<uint32_t>(tile_width_u64);
      const uint32_t tile_height = static_cast<uint32_t>(tile_height_u64);
      const uint32_t components = static_cast<uint32_t>(desired_colors);

      if (!residual_bmp_path.empty())
      {
        if (effective_bitmap_order == TlgOptions::DumpResidualsOrder::BeforeHilbert)
          effective_bitmap_order = TlgOptions::DumpResidualsOrder::AfterColorFilter;
        residual_bitmap.width = width;
        residual_bitmap.height = height;
        residual_bitmap.channels = components;
        const size_t total_pixels = static_cast<size_t>(width) * height;
        const size_t buffer_size = total_pixels * components;
        residual_bitmap.data.resize(buffer_size);
        std::fill(residual_bitmap.data.begin(), residual_bitmap.data.end(), static_cast<uint8_t>(128));
        residual_bitmap_ptr = &residual_bitmap;
      }
      const uint64_t tile_capacity_u64 = static_cast<uint64_t>(tile_width) * tile_height * 4u * 2u;
      if (tile_capacity_u64 == 0 || tile_capacity_u64 > std::numeric_limits<size_t>::max())
      {
        err = "tlg8: tile buffer size overflow";
        return false;
      }
      std::vector<uint8_t> tile_buffer(static_cast<size_t>(tile_capacity_u64));
      const uint8_t *packed_ptr = packed.data();

      TlgOptions::DumpResidualsOrder effective_dump_order = dump_residuals_order;
      if (dump_file)
      {
        if (effective_dump_order == TlgOptions::DumpResidualsOrder::BeforeHilbert)
          effective_dump_order = TlgOptions::DumpResidualsOrder::AfterColorFilter;
      }

      uint64_t total_entropy_bits = 0;

      for (uint32_t origin_y = 0; origin_y < height; origin_y += tile_height)
      {
        const uint32_t tile_h = std::min<uint32_t>(tile_height, height - origin_y);
        for (uint32_t origin_x = 0; origin_x < width; origin_x += tile_width)
        {
          const uint32_t tile_w = std::min<uint32_t>(tile_width, width - origin_x);
          detail::bitio::BitWriter writer(tile_buffer.data(), tile_buffer.size());
          uint64_t tile_entropy_bits = 0;
          writer.set_bit_counter(&tile_entropy_bits);
          if (!encode_for_tile(writer,
                               packed_ptr,
                               width,
                               components,
                               origin_x,
                               origin_y,
                               tile_w,
                               tile_h,
                               dump_file.get(),
                               effective_dump_order,
                               residual_bitmap_ptr,
                               effective_bitmap_order,
                               residual_bitmap_emphasis,
                               err))
            return false;
          if (!writer.align_to_u32_zero() || !writer.finish())
          {
            err = "tlg8: failed to finalize tile payload";
            return false;
          }
          total_entropy_bits += tile_entropy_bits;
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

      if (out_entropy_bits)
        *out_entropy_bits = total_entropy_bits;

      if (residual_bitmap_ptr)
      {
        std::string bmp_err;
        if (!save_bmp(residual_bmp_path, *residual_bitmap_ptr, bmp_err))
        {
          err = "tlg8: 残差ビットマップを書き出せません: " + bmp_err;
          return false;
        }
      }

      return true;
    }
  } // namespace enc
} // namespace tlg::v8
