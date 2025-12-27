#include "tlg8_io.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "tlg_io_common.h"
#include "tlg8_bit_io.h"
#include "tlg8_entropy.h"
#include "tlg8_reorder.h"
#include "sha256.h"

namespace
{
  using tlg::detail::read_exact;
  using tlg::v8::DumpContext;

  constexpr uint8_t kHeaderFlagHasGolombTable = 0x01;
  constexpr std::size_t kGolombSerializedFieldCount = tlg::v8::enc::kGolombColumnCount + 2;
  constexpr std::size_t kSerializedGolombTableSize =
      static_cast<std::size_t>(tlg::v8::enc::kGolombRowCount) * kGolombSerializedFieldCount * sizeof(uint16_t);

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

  bool write_f64le(FILE *fp, double value)
  {
    static_assert(sizeof(double) == sizeof(uint64_t), "double のサイズが 64bit ではありません");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return write_u64le(fp, bits);
  }

  bool write_feature_stats_file(const tlg::v8::DumpContext::FeatureStatsState &state, std::string &err)
  {
    if (!state.enabled())
      return true;
    if (state.sum.size() != state.sumsq.size())
    {
      err = "tlg8: 特徴量統計の内部状態が不正です";
      return false;
    }

    std::filesystem::path out_path = std::filesystem::u8path(state.path);
    if (!out_path.parent_path().empty())
    {
      std::error_code ec;
      std::filesystem::create_directories(out_path.parent_path(), ec);
      if (ec)
      {
        err = "tlg8: 特徴量統計の出力ディレクトリを作成できません: " + out_path.parent_path().u8string();
        return false;
      }
    }

    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(state.path.c_str(), "wb"), &std::fclose);
    if (!file)
    {
      err = "tlg8: 特徴量統計を書き出すファイルを開けません: " + state.path;
      return false;
    }

    constexpr char magic[4] = {'F', 'S', 'C', '8'};
    if (!write_bytes(file.get(), magic, sizeof(magic)))
    {
      err = "tlg8: 特徴量統計のマジック書き込みに失敗しました";
      return false;
    }
    if (!write_u32le(file.get(), 1u))
    {
      err = "tlg8: 特徴量統計のバージョン書き込みに失敗しました";
      return false;
    }
    const uint32_t dimension = static_cast<uint32_t>(state.sum.size());
    if (!write_u32le(file.get(), dimension))
    {
      err = "tlg8: 特徴量統計の次元数書き込みに失敗しました";
      return false;
    }
    if (!write_u64le(file.get(), state.count))
    {
      err = "tlg8: 特徴量統計のサンプル数書き込みに失敗しました";
      return false;
    }
    for (double value : state.sum)
    {
      if (!write_f64le(file.get(), value))
      {
        err = "tlg8: 特徴量統計の総和書き込みに失敗しました";
        return false;
      }
    }
    for (double value : state.sumsq)
    {
      if (!write_f64le(file.get(), value))
      {
        err = "tlg8: 特徴量統計の二乗和書き込みに失敗しました";
        return false;
      }
    }
    return true;
  }

  bool read_u32le(FILE *fp, uint32_t &value)
  {
    return read_little_endian(fp, value);
  }

  bool read_u64le(FILE *fp, uint64_t &value)
  {
    return read_little_endian(fp, value);
  }

  std::string escape_json_string(const std::string &value)
  {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (unsigned char ch : value)
    {
      switch (ch)
      {
      case '\\':
      case '"':
        result.push_back('\\');
        result.push_back(static_cast<char>(ch));
        break;
      case '\b':
        result.append("\\b");
        break;
      case '\f':
        result.append("\\f");
        break;
      case '\n':
        result.append("\\n");
        break;
      case '\r':
        result.append("\\r");
        break;
      case '\t':
        result.append("\\t");
        break;
      default:
        if (ch < 0x20)
        {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(ch));
          result.append(buf);
        }
        else
        {
          result.push_back(static_cast<char>(ch));
        }
        break;
      }
    }
    result.push_back('"');
    return result;
  }

  double file_time_to_seconds(const std::filesystem::file_time_type &tp)
  {
    using clock = std::filesystem::file_time_type::clock;
    const auto system_now = std::chrono::system_clock::now();
    const auto file_now = clock::now();
    const auto adjusted = tp - file_now + system_now;
    return std::chrono::duration<double>(adjusted.time_since_epoch()).count();
  }

  struct LabelCacheInputMeta
  {
    std::filesystem::path resolved_path;
    std::uintmax_t size = 0;
    double mtime_seconds = 0.0;
    std::string sha256_hex;
  };

  bool compute_file_sha256(const std::filesystem::path &path,
                           std::string &out_hex,
                           std::string &err)
  {
    std::ifstream fp(path, std::ios::binary);
    if (!fp)
    {
      err = "tlg8: ファイルのハッシュ計算に失敗しました: " + path.u8string();
      return false;
    }
    std::array<char, 1024 * 1024> buffer{};
    Sha256 hasher;
    while (fp)
    {
      fp.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const std::streamsize got = fp.gcount();
      if (got > 0)
        hasher.update(buffer.data(), static_cast<std::size_t>(got));
    }
    if (fp.bad())
    {
      err = "tlg8: ファイルの読み取り中にエラーが発生しました: " + path.u8string();
      return false;
    }
    out_hex = hasher.hexdigest();
    return true;
  }

  bool gather_label_cache_inputs(const std::vector<std::string> &input_paths,
                                 std::vector<LabelCacheInputMeta> &out,
                                 std::string &err)
  {
    out.clear();
    out.reserve(input_paths.size());
    for (const auto &raw : input_paths)
    {
      std::filesystem::path source = std::filesystem::u8path(raw);
      std::error_code ec;
      if (!std::filesystem::exists(source, ec) || ec)
      {
        err = "tlg8: ラベルキャッシュ入力ファイルが見つかりません: " + source.u8string();
        return false;
      }
      auto resolved = std::filesystem::canonical(source, ec);
      if (ec)
      {
        resolved = std::filesystem::absolute(source, ec);
        if (ec)
        {
          err = "tlg8: ラベルキャッシュ入力の絶対パス取得に失敗しました: " + source.u8string();
          return false;
        }
      }
      const auto size = std::filesystem::file_size(resolved, ec);
      if (ec)
      {
        err = "tlg8: ラベルキャッシュ入力のファイルサイズ取得に失敗しました: " + resolved.u8string();
        return false;
      }
      const auto mtime = std::filesystem::last_write_time(resolved, ec);
      if (ec)
      {
        err = "tlg8: ラベルキャッシュ入力の更新時刻取得に失敗しました: " + resolved.u8string();
        return false;
      }
      std::string sha;
      if (!compute_file_sha256(resolved, sha, err))
        return false;

      LabelCacheInputMeta meta;
      meta.resolved_path = resolved;
      meta.size = size;
      meta.mtime_seconds = file_time_to_seconds(mtime);
      meta.sha256_hex = std::move(sha);
      out.push_back(std::move(meta));
    }
    return true;
  }

  bool hex_to_bytes(const std::string &hex, std::array<uint8_t, 32> &out)
  {
    if (hex.size() != 64)
      return false;
    auto hex_value = [](char ch) -> int {
      if (ch >= '0' && ch <= '9')
        return ch - '0';
      if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
      if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
      return -1;
    };
    for (std::size_t i = 0; i < out.size(); ++i)
    {
      const int hi = hex_value(hex[i * 2]);
      const int lo = hex_value(hex[i * 2 + 1]);
      if (hi < 0 || lo < 0)
        return false;
      out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
  }

  bool write_label_cache_meta(const tlg::v8::DumpContext::LabelCacheState &state,
                              std::string &err)
  {
    if (state.meta_path.empty())
      return true;

    std::vector<LabelCacheInputMeta> inputs;
    if (!gather_label_cache_inputs(state.input_paths, inputs, err))
      return false;

    std::filesystem::path bin_path = std::filesystem::u8path(state.bin_path);
    std::error_code ec;
    const auto bin_size = std::filesystem::file_size(bin_path, ec);
    if (ec)
    {
      err = "tlg8: ラベルキャッシュバイナリのサイズ取得に失敗しました: " + bin_path.u8string();
      return false;
    }
    const uint64_t expected_size = state.record_count * tlg::v8::kLabelRecordSize;
    if (bin_size != static_cast<std::uintmax_t>(expected_size))
    {
      err = "tlg8: ラベルキャッシュのレコード数とファイルサイズが一致しません";
      return false;
    }

    Sha256 dataset_hasher;
    for (const auto &meta : inputs)
    {
      std::array<uint8_t, 32> bytes{};
      if (!hex_to_bytes(meta.sha256_hex, bytes))
      {
        err = "tlg8: ラベルキャッシュ入力の SHA-256 が不正です";
        return false;
      }
      dataset_hasher.update(bytes.data(), bytes.size());
      std::array<uint8_t, 8> size_bytes{};
      std::uint64_t size_le = static_cast<std::uint64_t>(meta.size);
      for (std::size_t i = 0; i < size_bytes.size(); ++i)
        size_bytes[i] = static_cast<uint8_t>((size_le >> (i * 8)) & 0xffu);
      dataset_hasher.update(size_bytes.data(), size_bytes.size());
      const auto path_utf8 = meta.resolved_path.u8string();
      dataset_hasher.update(path_utf8.data(), path_utf8.size());
    }
    const std::string dataset_sha = dataset_hasher.hexdigest();

    std::filesystem::path meta_path = std::filesystem::u8path(state.meta_path);
    if (!meta_path.parent_path().empty())
    {
      std::filesystem::create_directories(meta_path.parent_path(), ec);
      if (ec)
      {
        err = "tlg8: ラベルキャッシュメタデータのディレクトリ作成に失敗しました: " + meta_path.parent_path().u8string();
        return false;
      }
    }

    std::filesystem::path tmp_path = meta_path;
    tmp_path += ".tmp";
    std::ofstream out_fp(tmp_path, std::ios::binary);
    if (!out_fp)
    {
      err = "tlg8: ラベルキャッシュメタデータを書き出すファイルを開けません: " + tmp_path.u8string();
      return false;
    }
    out_fp << "{\n";
    out_fp << "  \"schema\": 1,\n";
    out_fp << "  \"record_size\": " << tlg::v8::kLabelRecordSize << ",\n";
    out_fp << "  \"record_count\": " << static_cast<unsigned long long>(state.record_count) << ",\n";
    out_fp << "  \"inputs\": [\n";
    for (std::size_t index = 0; index < inputs.size(); ++index)
    {
      const auto &meta = inputs[index];
      out_fp << "    {\n";
      out_fp << "      \"path\": " << escape_json_string(meta.resolved_path.u8string()) << ",\n";
      out_fp << "      \"size\": " << static_cast<unsigned long long>(meta.size) << ",\n";
      out_fp << std::setprecision(17);
      out_fp << "      \"mtime\": " << meta.mtime_seconds << ",\n";
      out_fp << "      \"sha256\": " << escape_json_string(meta.sha256_hex) << "\n";
      out_fp << "    }";
      if (index + 1 < inputs.size())
        out_fp << ",";
      out_fp << "\n";
    }
    out_fp << "  ],\n";
    out_fp << "  \"dataset_sha256\": " << escape_json_string(dataset_sha) << "\n";
    out_fp << "}\n";
    if (!out_fp)
    {
      err = "tlg8: ラベルキャッシュメタデータの書き込みに失敗しました";
      return false;
    }
    out_fp.close();
    if (!out_fp)
    {
      err = "tlg8: ラベルキャッシュメタデータのクローズに失敗しました";
      return false;
    }

    std::filesystem::rename(tmp_path, meta_path, ec);
    if (ec)
    {
      std::filesystem::remove(meta_path, ec);
      ec.clear();
      std::filesystem::rename(tmp_path, meta_path, ec);
      if (ec)
      {
        err = "tlg8: ラベルキャッシュメタデータの配置に失敗しました: " + meta_path.u8string();
        return false;
      }
    }
    return true;
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
      enc::golomb_ratio_array raise_ratios{};
      enc::golomb_ratio_array fall_ratios{};
      std::size_t offset = 0;
      auto read_u16 = [&]() {
        const uint16_t low = table_bytes[offset];
        const uint16_t high = table_bytes[offset + 1];
        offset += 2;
        return static_cast<uint16_t>(low | static_cast<uint16_t>(high << 8));
      };
      for (std::size_t row = 0; row < table.size(); ++row)
      {
        const uint16_t raise_value = read_u16();
        const uint16_t fall_value = read_u16();
        if (raise_value == 0 || raise_value > 16 || fall_value == 0 || fall_value > 16)
        {
          err = "tlg8: ゴロムテーブルの適応比率が不正です";
          return false;
        }
        raise_ratios[row] = static_cast<uint8_t>(raise_value);
        fall_ratios[row] = static_cast<uint8_t>(fall_value);

        uint32_t row_sum = 0;
        for (std::size_t col = 0; col < table[row].size(); ++col)
        {
          const uint16_t value = read_u16();
          table[row][col] = value;
          row_sum += value;
        }
        if (row_sum != enc::kGolombRowSum)
        {
          err = "tlg8: ゴロムテーブルの行合計が不正です";
          return false;
        }
      }
      (void)enc::apply_golomb_table(table, raise_ratios, fall_ratios);
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

  namespace
  {
    constexpr std::array<const char *, tlg::v8::enc::kReorderPatternCount> kReorderPatternNames = {
        "hilbert",
        "zigzag_diag",
        "zigzag_antidiag",
        "zigzag_horz",
        "zigzag_vert",
        "zigzag_nne_ssw",
        "zigzag_nee_sww",
        "zigzag_nww_see",
    };
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
                       std::array<uint64_t, kReorderPatternCount> *reorder_histogram,
                       DumpContext *training_ctx,
                       bool force_hilbert_reorder,
                       int force_entropy,
                       std::string &err);

    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int desired_colors,
                   const std::string &dump_residuals_path,
                   TlgOptions::DumpResidualsOrder dump_residuals_order,
                   const std::string &dump_golomb_prediction_path,
                   const std::string &reorder_histogram_path,
                   const std::string &residual_bmp_path,
                   TlgOptions::DumpResidualsOrder residual_bmp_order,
                   double residual_bmp_emphasis,
                   TlgOptions::DumpMode dump_mode,
                   const std::string &training_dump_path,
                   const std::string &training_image_tag,
                   const std::string &training_stats_path,
                   const std::string &label_cache_bin_path,
                   const std::string &label_cache_meta_path,
                   bool force_hilbert_reorder,
                   int force_entropy,
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

      if ((label_cache_bin_path.empty() ^ label_cache_meta_path.empty()))
      {
        err = "tlg8: ラベルキャッシュの出力指定は bin と meta を同時に与えてください";
        return false;
      }

      const bool mode_has_features = (dump_mode != TlgOptions::DumpMode::Labels);
      const bool mode_has_labels = (dump_mode != TlgOptions::DumpMode::Features);

      if (!mode_has_labels && (!label_cache_bin_path.empty() || !label_cache_meta_path.empty()))
      {
        err = "tlg8: --tlg8-dump-mode=features ではラベルキャッシュを出力できません";
        return false;
      }
      if (!mode_has_features && !training_stats_path.empty())
      {
        err = "tlg8: --tlg8-dump-mode=labels では特徴量統計を出力できません";
        return false;
      }

      std::unique_ptr<FILE, FileCloser> training_dump_file;
      std::unique_ptr<FILE, FileCloser> label_cache_file;
      DumpContext training_context;
      const bool enable_training_dump = !training_dump_path.empty();
      const bool enable_feature_stats = mode_has_features && !training_stats_path.empty();
      const bool enable_label_cache = mode_has_labels && (!label_cache_bin_path.empty() || !label_cache_meta_path.empty());
      const bool need_training_ctx = enable_training_dump || enable_feature_stats || enable_label_cache;
      if (need_training_ctx)
      {
        training_context.image_tag = training_image_tag;
        training_context.image_width = src.width;
        training_context.image_height = src.height;
        training_context.components = static_cast<uint32_t>(desired_colors);
        training_context.enable_features = mode_has_features;
        training_context.enable_labels = mode_has_labels;
      }

      if (enable_feature_stats)
      {
        training_context.feature_stats.path = training_stats_path;
        training_context.feature_stats.sum.assign(tlg::v8::kFeatureVectorSize, 0.0);
        training_context.feature_stats.sumsq.assign(tlg::v8::kFeatureVectorSize, 0.0);
        training_context.feature_stats.count = 0;
      }

      if (enable_training_dump)
      {
        FILE *ml_fp = std::fopen(training_dump_path.c_str(), "ab");
        if (!ml_fp)
        {
          err = "tlg8: 学習データを書き出すファイルを開けません: " + training_dump_path;
          return false;
        }
        training_dump_file.reset(ml_fp);
        training_context.training_dump.file = ml_fp;
        training_context.training_dump.path = training_dump_path;
      }

      if (enable_label_cache)
      {
        std::filesystem::path bin_path = std::filesystem::u8path(label_cache_bin_path);
        if (!bin_path.parent_path().empty())
        {
          std::error_code ec;
          std::filesystem::create_directories(bin_path.parent_path(), ec);
          if (ec)
          {
            err = "tlg8: ラベルキャッシュの出力ディレクトリを作成できません: " + bin_path.parent_path().u8string();
            return false;
          }
        }
        FILE *bin_fp = std::fopen(label_cache_bin_path.c_str(), "wb");
        if (!bin_fp)
        {
          err = "tlg8: ラベルキャッシュを書き出すファイルを開けません: " + label_cache_bin_path;
          return false;
        }
        label_cache_file.reset(bin_fp);
        training_context.label_cache.file = bin_fp;
        training_context.label_cache.bin_path = label_cache_bin_path;
        training_context.label_cache.meta_path = label_cache_meta_path;
        if (enable_training_dump)
          training_context.label_cache.input_paths.push_back(training_dump_path);
        training_context.label_cache.record_count = 0;
        training_context.enable_labels = mode_has_labels;
      }

      auto *training_ctx_ptr = need_training_ctx ? &training_context : nullptr;

      std::vector<uint8_t> packed;
      if (!copy_pixels_to_buffer(src, desired_colors, packed, err))
        return false;

      PixelBuffer residual_bitmap;
      PixelBuffer *residual_bitmap_ptr = nullptr;
      TlgOptions::DumpResidualsOrder effective_bitmap_order = residual_bmp_order;
      const double residual_bitmap_emphasis = residual_bmp_emphasis;
      std::array<uint64_t, kReorderPatternCount> reorder_histogram{};
      auto *reorder_histogram_ptr = reorder_histogram_path.empty() ? nullptr : &reorder_histogram;

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
      const auto &raise_ratios = current_m_raise_ratios();
      const auto &fall_ratios = current_m_fall_ratios();
      std::array<uint8_t, kSerializedGolombTableSize> table_bytes{};
      std::size_t table_offset = 0;
      for (std::size_t row_index = 0; row_index < golomb_table.size(); ++row_index)
      {
        const uint16_t raise_value = static_cast<uint16_t>(raise_ratios[row_index]);
        const uint16_t fall_value = static_cast<uint16_t>(fall_ratios[row_index]);
        const auto write_value = [&](uint16_t value) {
          table_bytes[table_offset++] = static_cast<uint8_t>(value & 0xFFu);
          table_bytes[table_offset++] = static_cast<uint8_t>((value >> 8) & 0xFFu);
        };
        write_value(raise_value);
        write_value(fall_value);
        for (uint16_t value : golomb_table[row_index])
        {
          write_value(value);
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
                               reorder_histogram_ptr,
                               training_ctx_ptr,
                               force_hilbert_reorder,
                               force_entropy,
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

      if (reorder_histogram_ptr)
      {
        FILE *hist_fp = std::fopen(reorder_histogram_path.c_str(), "w");
        if (!hist_fp)
        {
          err = "tlg8: 並び替えヒストグラムを書き出すファイルを開けません: " + reorder_histogram_path;
          return false;
        }
        std::unique_ptr<FILE, FileCloser> histogram_file(hist_fp);
        for (uint32_t index = 0; index < kReorderPatternCount; ++index)
        {
          const char *name = kReorderPatternNames[index];
          if (std::fprintf(histogram_file.get(), "%u\t%s\t%llu\n", index, name,
                           static_cast<unsigned long long>((*reorder_histogram_ptr)[index])) < 0)
          {
            err = "tlg8: 並び替えヒストグラムを書き出せません";
            return false;
          }
        }
      }

      if (training_ctx_ptr && training_ctx_ptr->wants_training_dump())
        std::fflush(training_ctx_ptr->training_dump.file);

      if (training_ctx_ptr && training_ctx_ptr->wants_label_cache())
      {
        std::fflush(training_ctx_ptr->label_cache.file);
        training_ctx_ptr->label_cache.file = nullptr;
        label_cache_file.reset();
        if (!write_label_cache_meta(training_context.label_cache, err))
          return false;
      }

      if (training_ctx_ptr && training_ctx_ptr->wants_feature_stats())
      {
        if (!write_feature_stats_file(training_context.feature_stats, err))
          return false;
      }

      return true;
    }
  } // namespace enc
} // namespace tlg::v8
