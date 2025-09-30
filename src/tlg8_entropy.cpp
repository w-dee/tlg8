#include "tlg8_entropy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  using namespace tlg::v8::enc;
  using BitWriter = tlg::v8::detail::bitio::BitWriter;
  using BitReader = tlg::v8::detail::bitio::BitReader;
  using tlg::v8::enc::kGolombRowCount;
  constexpr uint32_t kGolombColumnCount = 9;
  constexpr uint32_t kGolombRowSum = 1024;

  using GolombRow = std::array<uint16_t, kGolombColumnCount>;
  using GolombTable = std::array<GolombRow, kGolombRowCount>;

  constexpr GolombTable DEFAULT_GOLOMB_TABLE = {GolombRow{0, 4, 4, 7, 24, 89, 270, 489, 137},
                                                GolombRow{2, 2, 5, 13, 67, 98, 230, 476, 131},
                                                GolombRow{3, 2, 5, 15, 77, 92, 238, 462, 130},
                                                GolombRow{2, 2, 4, 10, 51, 108, 237, 482, 128},
                                                GolombRow{2, 3, 7, 33, 74, 81, 237, 450, 137},
                                                GolombRow{3, 1, 5, 28, 66, 92, 246, 452, 131}};

  GolombTable g_golomb_table = DEFAULT_GOLOMB_TABLE;

  std::array<std::array<uint8_t, kGolombRowCount>, kGolombRowSum> g_bit_length_table{};
  bool g_table_ready = false;

  inline void ensure_table_initialized()
  {
    if (g_table_ready)
      return;
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      int accumulator = 0;
      for (uint32_t col = 0; col < kGolombColumnCount; ++col)
      {
        const uint16_t count = g_golomb_table[row][col];
        for (uint16_t i = 0; i < count; ++i)
        {
          const uint32_t idx = static_cast<uint32_t>(std::min(accumulator, static_cast<int>(kGolombRowSum - 1)));
          g_bit_length_table[idx][row] = static_cast<uint8_t>(col);
          ++accumulator;
        }
      }
    }
    g_table_ready = true;
  }

  inline void write_zero_bits(BitWriter &writer, uint32_t count)
  {
    while (count >= 8)
    {
      writer.put_upto8(0, 8);
      count -= 8;
    }
    if (count)
      writer.put_upto8(0, static_cast<unsigned>(count));
  }

  inline void write_bits(BitWriter &writer, uint32_t value, unsigned count)
  {
    if (count == 0)
      return;
    if (count <= 8)
    {
      writer.put_upto8(value, count);
      return;
    }
    writer.put(value, count);
  }

  inline void put_gamma(BitWriter &writer, uint32_t value)
  {
    if (value == 0)
      return;
    uint32_t t = value >> 1;
    while (t)
    {
      writer.put_upto8(0, 1);
      t >>= 1;
    }
    writer.put_upto8(1, 1);
    uint32_t zeros = 0;
    t = value >> 1;
    while (t)
    {
      ++zeros;
      t >>= 1;
    }
    for (uint32_t i = 0; i < zeros; ++i)
      writer.put_upto8((value >> i) & 1u, 1);
  }

  inline void put_run_length(BitWriter &writer, uint32_t count)
  {
    if (count == 0)
      return;
    if (count == 1)
    {
      writer.put_upto8(0, 1);
      return;
    }
    if (count == 2)
    {
      writer.put_upto8(1, 1);
      writer.put_upto8(0, 1);
      return;
    }
    if (count == 3)
    {
      writer.put_upto8(1, 1);
      writer.put_upto8(1, 1);
      writer.put_upto8(0, 1);
      return;
    }
    writer.put_upto8(1, 1);
    writer.put_upto8(1, 1);
    writer.put_upto8(1, 1);
    put_gamma(writer, count - 3);
  }

  inline uint64_t gamma_bits(uint32_t value)
  {
    if (value == 0)
      return 0;
    uint32_t t = value >> 1;
    uint32_t zeros = 0;
    while (t)
    {
      ++zeros;
      t >>= 1;
    }
    return static_cast<uint64_t>(zeros) * 2u + 1u;
  }

  inline uint64_t run_length_bits(uint32_t count)
  {
    if (count == 0)
      return 0;
    if (count == 1)
      return 1;
    if (count == 2)
      return 2;
    if (count == 3)
      return 3;
    return 3u + gamma_bits(count - 3);
  }

  uint64_t estimate_plain_component(const int16_t *values, uint32_t count, uint32_t component)
  {
    if (count == 0)
      return 0;
    ensure_table_initialized();
    const int row = golomb_row_index(GolombCodingKind::Plain, component);
    uint64_t bits = 0;
    int a = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
      const int e = static_cast<int>(values[i]);
      const uint32_t m = (e >= 0) ? static_cast<uint32_t>(2 * e) : static_cast<uint32_t>(-2 * e - 1);
      const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
      const uint32_t q = (k > 0) ? (m >> k) : m;
      bits += q + 1u + static_cast<uint32_t>(k);
      a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
    }
    return bits;
  }

  uint64_t estimate_run_length_component(const int16_t *values, uint32_t count, uint32_t component)
  {
    if (count == 0)
      return 0;
    ensure_table_initialized();
    const int row = golomb_row_index(GolombCodingKind::RunLength, component);
    uint64_t bits = 1; // 先頭要素が 0 か否かのビット
    int a = 0;
    uint32_t index = 0;
    int zero_run = 0;
    while (index < count)
    {
      const int16_t value = values[index];
      if (value != 0)
      {
        if (zero_run)
        {
          bits += run_length_bits(static_cast<uint32_t>(zero_run));
          zero_run = 0;
        }
        const uint32_t start = index;
        while (index < count && values[index] != 0)
          ++index;
        const uint32_t nonzero_count = index - start;
        bits += gamma_bits(nonzero_count);
        for (uint32_t j = start; j < index; ++j)
        {
          int64_t mapped = (values[j] >= 0) ? (static_cast<int64_t>(2) * values[j])
                                            : (static_cast<int64_t>(-2) * values[j] - 1);
          mapped -= 1;
          if (mapped < 0)
            mapped = 0;
          const uint32_t m = static_cast<uint32_t>(mapped);
          const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
          const uint32_t q = (k > 0) ? (m >> k) : m;
          bits += q + 1u + static_cast<uint32_t>(k);
          a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
        }
      }
      else
      {
        ++zero_run;
        ++index;
      }
    }
    if (zero_run)
      bits += run_length_bits(static_cast<uint32_t>(zero_run));
    return bits;
  }

  bool encode_plain_component(BitWriter &writer,
                              const int16_t *values,
                              uint32_t count,
                              uint32_t component)
  {
    if (count == 0)
      return true;
    ensure_table_initialized();
    const int row = golomb_row_index(GolombCodingKind::Plain, component);
    int a = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
      const int e = static_cast<int>(values[i]);
      const uint32_t m = (e >= 0) ? static_cast<uint32_t>(2 * e) : static_cast<uint32_t>(-2 * e - 1);
      const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
      const uint32_t q = (k > 0) ? (m >> k) : m;
      write_zero_bits(writer, q);
      writer.put_upto8(1, 1);
      if (k)
      {
        const uint32_t mask = (static_cast<uint32_t>(1u) << k) - 1u;
        write_bits(writer, m & mask, static_cast<unsigned>(k));
      }
      a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
    }
    return true;
  }

  bool encode_run_length_component(BitWriter &writer,
                                   const int16_t *values,
                                   uint32_t count,
                                   uint32_t component)
  {
    if (count == 0)
      return true;
    ensure_table_initialized();
    writer.put_upto8(values[0] ? 1u : 0u, 1);
    const int row = golomb_row_index(GolombCodingKind::RunLength, component);
    int a = 0;
    uint32_t index = 0;
    int zero_run = 0;
    while (index < count)
    {
      const int16_t value = values[index];
      if (value != 0)
      {
        if (zero_run)
        {
          put_run_length(writer, static_cast<uint32_t>(zero_run));
          zero_run = 0;
        }
        const uint32_t start = index;
        while (index < count && values[index] != 0)
          ++index;
        const uint32_t nonzero_count = index - start;
        put_gamma(writer, nonzero_count);
        for (uint32_t j = start; j < index; ++j)
        {
          int64_t mapped = (values[j] >= 0) ? (static_cast<int64_t>(2) * values[j])
                                            : (static_cast<int64_t>(-2) * values[j] - 1);
          mapped -= 1; // 0 を符号化する必要はないので -1 する
          if (mapped < 0)
            mapped = 0;
          const uint32_t m = static_cast<uint32_t>(mapped);
          const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
          const uint32_t q = (k > 0) ? (m >> k) : m;
          write_zero_bits(writer, q);
          writer.put_upto8(1, 1);
          if (k)
          {
            const uint32_t mask = (static_cast<uint32_t>(1u) << k) - 1u;
            write_bits(writer, m & mask, static_cast<unsigned>(k));
          }
          a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
        }
      }
      else
      {
        ++zero_run;
        ++index;
        continue;
      }
    }
    if (zero_run)
      put_run_length(writer, static_cast<uint32_t>(zero_run));
    return true;
  }

  inline bool read_bit(BitReader &reader, uint32_t &bit)
  {
    if (reader.eof())
      return false;
    bit = reader.get_upto8(1);
    return true;
  }

  bool read_bits(BitReader &reader, unsigned count, uint32_t &value)
  {
    value = 0;
    for (unsigned i = 0; i < count; ++i)
    {
      uint32_t bit = 0;
      if (!read_bit(reader, bit))
        return false;
      value |= (bit & 1u) << i;
    }
    return true;
  }

  bool read_gamma(BitReader &reader, uint32_t &value)
  {
    uint32_t zeros = 0;
    while (true)
    {
      uint32_t bit = 0;
      if (!read_bit(reader, bit))
        return false;
      if (bit)
        break;
      ++zeros;
      if (zeros > 31)
        return false;
    }
    uint32_t suffix = 0;
    if (zeros && !read_bits(reader, zeros, suffix))
      return false;
    value = (static_cast<uint32_t>(1u) << zeros) + suffix;
    return true;
  }

  int read_run_length(BitReader &reader)
  {
    uint32_t bit = 0;
    if (!read_bit(reader, bit))
      return 0;
    if (bit == 0)
      return 1;
    if (!read_bit(reader, bit))
      return 0;
    if (bit == 0)
      return 2;
    if (!read_bit(reader, bit))
      return 0;
    if (bit == 0)
      return 3;
    uint32_t gamma = 0;
    if (!read_gamma(reader, gamma))
      return 0;
    return static_cast<int>(gamma + 3);
  }

  bool decode_plain_component(BitReader &reader,
                              uint32_t expected_count,
                              uint32_t component,
                              int16_t *dst)
  {
    if (expected_count == 0)
      return true;
    ensure_table_initialized();
    const int row = golomb_row_index(GolombCodingKind::Plain, component);
    int a = 0;
    for (uint32_t produced = 0; produced < expected_count; ++produced)
    {
      const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
      uint32_t q = 0;
      while (true)
      {
        uint32_t bit = 0;
        if (!read_bit(reader, bit))
          return false;
        if (bit)
          break;
        ++q;
      }
      uint32_t remainder = 0;
      if (k > 0 && !read_bits(reader, static_cast<unsigned>(k), remainder))
        return false;
      const uint32_t m = (q << k) + remainder;
      const int residual = static_cast<int>((m >> 1) ^ -static_cast<int>(m & 1u));
      dst[produced] = static_cast<int16_t>(residual);
      a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
    }
    return true;
  }

  bool decode_run_length_component(BitReader &reader,
                                   uint32_t expected_count,
                                   uint32_t component,
                                   int16_t *dst)
  {
    if (expected_count == 0)
      return true;
    ensure_table_initialized();
    uint32_t first_bit = 0;
    if (!read_bit(reader, first_bit))
      return false;
    bool expect_nonzero = (first_bit != 0);
    const int row = golomb_row_index(GolombCodingKind::RunLength, component);
    int a = 0;
    uint32_t produced = 0;
    while (produced < expected_count)
    {
      if (!expect_nonzero)
      {
        const int run = read_run_length(reader);
        if (run <= 0)
          return false;
        if (produced + static_cast<uint32_t>(run) > expected_count)
          return false;
        for (int i = 0; i < run; ++i)
          dst[produced++] = 0;
        expect_nonzero = true;
        continue;
      }

      uint32_t run = 0;
      if (!read_gamma(reader, run) || run == 0)
        return false;
      if (produced + run > expected_count)
        return false;
      for (uint32_t i = 0; i < run; ++i)
      {
        const int k = g_bit_length_table[static_cast<uint32_t>(a)][row];
        uint32_t q = 0;
        while (true)
        {
          uint32_t bit = 0;
          if (!read_bit(reader, bit))
            return false;
          if (bit)
            break;
          ++q;
        }
        uint32_t remainder = 0;
        if (k > 0 && !read_bits(reader, static_cast<unsigned>(k), remainder))
          return false;
        const uint32_t m = (q << k) + remainder;
        const int sign = static_cast<int>(m & 1u) - 1;
        const int vv = static_cast<int>(m >> 1);
        const int residual = (vv ^ sign) + sign + 1; // 符号化時に -1 しているので、ここで戻す
        dst[produced++] = static_cast<int16_t>(residual);
        a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
      }
      expect_nonzero = false;
    }
    return true;
  }

  uint64_t estimate_plain(const component_colors &colors, uint32_t components, uint32_t value_count)
  {
    uint64_t total = 0;
    for (uint32_t c = 0; c < components; ++c)
      total += estimate_plain_component(colors.values[c].data(), value_count, c);
    return total;
  }

  uint64_t estimate_run_length(const component_colors &colors, uint32_t components, uint32_t value_count)
  {
    uint64_t total = 0;
    for (uint32_t c = 0; c < components; ++c)
      total += estimate_run_length_component(colors.values[c].data(), value_count, c);
    return total;
  }

  bool encode_plain(BitWriter &writer,
                    const component_colors &colors,
                    uint32_t components,
                    uint32_t value_count,
                    std::string &err)
  {
    (void)err;
    for (uint32_t c = 0; c < components; ++c)
    {
      if (!encode_plain_component(writer, colors.values[c].data(), value_count, c))
        return false;
    }
    return true;
  }

  bool encode_run_length(BitWriter &writer,
                         const component_colors &colors,
                         uint32_t components,
                         uint32_t value_count,
                         std::string &err)
  {
    (void)err;
    for (uint32_t c = 0; c < components; ++c)
    {
      if (!encode_run_length_component(writer, colors.values[c].data(), value_count, c))
        return false;
    }
    return true;
  }
}

namespace tlg::v8
{
  bool configure_golomb_table(const std::string &path, std::string &err)
  {
    if (path.empty())
    {
      if (g_golomb_table != DEFAULT_GOLOMB_TABLE)
      {
        g_golomb_table = DEFAULT_GOLOMB_TABLE;
        g_table_ready = false;
      }
      err.clear();
      return true;
    }

    std::ifstream in(path);
    if (!in)
    {
      err = "tlg8: failed to open Golomb table: " + path;
      return false;
    }

    std::string line;
    std::size_t line_number = 0;
    std::vector<GolombRow> parsed_rows;
    parsed_rows.reserve(static_cast<std::size_t>(kGolombRowCount));

    while (std::getline(in, line))
    {
      ++line_number;
      const auto comment_pos = line.find_first_of("#;");
      if (comment_pos != std::string::npos)
        line.erase(comment_pos);

      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        continue;

      if (parsed_rows.size() >= static_cast<std::size_t>(kGolombRowCount))
      {
        err = "tlg8: extra data in Golomb table '" + path + "' at line " + std::to_string(line_number);
        return false;
      }

      const std::size_t row_index = parsed_rows.size();
      std::istringstream iss(line);
      GolombRow row_values{};
      int sum = 0;
      int value = 0;
      std::size_t col = 0;
      while (col < row_values.size() && (iss >> value))
      {
        if (value < 0)
        {
          err = "tlg8: negative value in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
          return false;
        }
        if (value > std::numeric_limits<uint16_t>::max())
        {
          err = "tlg8: value too large in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
          return false;
        }
        row_values[col++] = static_cast<uint16_t>(value);
        sum += value;
      }

      if (col != row_values.size())
      {
        if (iss.fail() && !iss.eof())
        {
          err = "tlg8: invalid token in Golomb table '" + path + "' at line " + std::to_string(line_number);
        }
        else
        {
          err = "tlg8: expected 9 values in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        }
        return false;
      }

      if (iss >> value)
      {
        err = "tlg8: too many values in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        return false;
      }

      if (sum != static_cast<int>(kGolombRowSum))
      {
        err = "tlg8: row sum must be 1024 in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        return false;
      }

      parsed_rows.push_back(row_values);
    }

    if (parsed_rows.empty())
    {
      err = "tlg8: Golomb table '" + path + "' must contain at least one row";
      return false;
    }

    std::vector<GolombRow> expanded_rows;
    const std::size_t parsed_count = parsed_rows.size();
    if (parsed_count == static_cast<std::size_t>(kGolombRowCount))
    {
      expanded_rows = parsed_rows;
    }
    else if (static_cast<std::size_t>(kGolombRowCount) % parsed_count == 0)
    {
      const std::size_t repeat = static_cast<std::size_t>(kGolombRowCount) / parsed_count;
      expanded_rows.reserve(static_cast<std::size_t>(kGolombRowCount));
      for (std::size_t i = 0; i < repeat; ++i)
        expanded_rows.insert(expanded_rows.end(), parsed_rows.begin(), parsed_rows.end());
    }
    else
    {
      err = "tlg8: Golomb table '" + path + "' must contain " + std::to_string(kGolombRowCount) +
            " rows or a divisor of that count";
      return false;
    }

    GolombTable candidate{};
    for (std::size_t i = 0; i < static_cast<std::size_t>(kGolombRowCount); ++i)
      candidate[i] = expanded_rows[i];

    if (candidate != g_golomb_table)
    {
      g_golomb_table = candidate;
      g_table_ready = false;
    }
    err.clear();
    return true;
  }
}

namespace tlg::v8::enc
{
  int golomb_row_index(GolombCodingKind kind, uint32_t component)
  {
    const uint32_t c = (component < 4u) ? component : 3u;
    if (kind == GolombCodingKind::Plain)
    {
      switch (c)
      {
      case 0:
        return 0;
      case 1:
        return 1;
      case 2:
        return 2;
      default:
        return 0;
      }
    }
    switch (c)
    {
    case 0:
      return 3;
    case 1:
      return 4;
    case 2:
      return 5;
    default:
      return 3;
    }
  }

  GolombCodingKind golomb_row_kind(uint32_t row)
  {
    switch (row)
    {
    case 0:
    case 1:
    case 2:
      return GolombCodingKind::Plain;
    case 3:
    case 4:
    case 5:
      return GolombCodingKind::RunLength;
    default:
      return GolombCodingKind::Plain;
    }
  }

  uint32_t golomb_row_component(uint32_t row)
  {
    switch (row)
    {
    case 0:
    case 3:
      return 0;
    case 1:
    case 4:
      return 1;
    case 2:
    case 5:
      return 2;
    default:
      return 0;
    }
  }

  bool encode_values(detail::bitio::BitWriter &writer,
                     GolombCodingKind kind,
                     uint32_t component,
                     const int16_t *values,
                     uint32_t count,
                     std::string &err)
  {
    (void)err;
    if (kind == GolombCodingKind::Plain)
      return encode_plain_component(writer, values, count, component);
    return encode_run_length_component(writer, values, count, component);
  }

  bool decode_values(detail::bitio::BitReader &reader,
                     GolombCodingKind kind,
                     uint32_t component,
                     uint32_t count,
                     int16_t *dst,
                     std::string &err)
  {
    (void)err;
    if (kind == GolombCodingKind::Plain)
      return decode_plain_component(reader, count, component, dst);
    return decode_run_length_component(reader, count, component, dst);
  }

  const std::array<entropy_encoder, kNumEntropyEncoders> &entropy_encoder_table()
  {
    static constexpr std::array<entropy_encoder, kNumEntropyEncoders> kEncoders = {
        entropy_encoder{GolombCodingKind::Plain, &estimate_plain, &encode_plain},
        entropy_encoder{GolombCodingKind::RunLength, &estimate_run_length, &encode_run_length}};
    return kEncoders;
  }

  bool decode_block(detail::bitio::BitReader &reader,
                    GolombCodingKind kind,
                    uint32_t components,
                    uint32_t value_count,
                    component_colors &out,
                    std::string &err)
  {
    for (auto &component : out.values)
      component.fill(0);

    for (uint32_t c = 0; c < components; ++c)
    {
      bool ok = false;
      if (kind == GolombCodingKind::Plain)
        ok = decode_plain_component(reader, value_count, c, out.values[c].data());
      else
        ok = decode_run_length_component(reader, value_count, c, out.values[c].data());
      if (!ok)
      {
        err = "tlg8: エントロピー復号に失敗しました";
        return false;
      }
    }
    return true;
  }
}
