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
  using tlg::v8::enc::golomb_table_counts;
  using tlg::v8::enc::kGolombColumnCount;
  using tlg::v8::enc::kGolombRowCount;
  using tlg::v8::enc::kGolombRowSum;

  using GolombRow = std::array<uint16_t, kGolombColumnCount>;
  using GolombTable = golomb_table_counts;

  constexpr std::size_t kLegacyGolombRowCount = 6;

  struct ParsedGolombTable
  {
    GolombTable table{};
    bool filled = false;
  };

  constexpr bool is_whitespace(char c)
  {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
  }

  constexpr bool is_digit(char c)
  {
    return c >= '0' && c <= '9';
  }

  constexpr ParsedGolombTable parse_default_golomb_table()
  {
    constexpr char data[] = R"(
3 1 6 10 19 106 254 498 127
3 1 7 10 102 78 241 447 135
3 1 6 14 86 102 255 435 122
2 3 4 6 23 97 238 513 138
2 3 4 10 63 109 263 454 116
3 2 5 12 96 70 245 452 139
3 1 6 10 19 106 254 498 127
2 3 4 6 23 97 238 513 138
        )";

    ParsedGolombTable result{};
    std::size_t row = 0;
    std::size_t col = 0;
    std::size_t index = 0;
    bool overflow = false;

    while (index < sizeof(data) - 1)
    {
      while (index < sizeof(data) - 1 && is_whitespace(data[index]))
        ++index;
      if (index >= sizeof(data) - 1)
        break;

      uint16_t value = 0;
      bool has_digit = false;
      while (index < sizeof(data) - 1 && is_digit(data[index]))
      {
        has_digit = true;
        value = static_cast<uint16_t>(value * 10 + static_cast<uint16_t>(data[index] - '0'));
        ++index;
      }

      if (!has_digit)
      {
        overflow = true;
        break;
      }

      if (row < result.table.size())
      {
        result.table[row][col] = value;
        ++col;
        if (col == result.table[row].size())
        {
          col = 0;
          ++row;
        }
      }
      else
      {
        overflow = true;
      }
    }

    result.filled = (row == result.table.size() && col == 0 && !overflow);
    return result;
  }

  inline constexpr GolombTable DEFAULT_GOLOMB_TABLE = []() constexpr
  {
    constexpr auto parsed = parse_default_golomb_table();
    static_assert(parsed.filled, "DEFAULT_GOLOMB_TABLE のデータが不足しています");
    return parsed.table;
  }();

  GolombTable g_golomb_table = DEFAULT_GOLOMB_TABLE;

  std::array<std::array<uint8_t, kGolombRowCount>, kGolombRowSum> g_bit_length_table{};
  bool g_table_ready = false;

  inline constexpr int kGolombGiveUpQ = 16;

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

  bool normalize_histogram_row(const std::array<uint64_t, kGolombColumnCount> &hist,
                               GolombRow &row)
  {
    row.fill(0);
    uint64_t total = 0;
    for (auto value : hist)
      total += value;
    if (total == 0)
      return false;

    std::array<uint64_t, kGolombColumnCount> remainders{};
    uint32_t assigned = 0;
    for (std::size_t col = 0; col < kGolombColumnCount; ++col)
    {
      const uint64_t numerator = hist[col] * static_cast<uint64_t>(kGolombRowSum);
      const uint16_t base = static_cast<uint16_t>(std::min<uint64_t>(numerator / total, kGolombRowSum));
      row[col] = base;
      remainders[col] = numerator % total;
      assigned += base;
    }

    if (assigned > kGolombRowSum)
    {
      uint32_t overflow = assigned - kGolombRowSum;
      for (std::size_t col = kGolombColumnCount; col-- > 0 && overflow > 0;)
      {
        const uint16_t reducible = static_cast<uint16_t>(std::min<uint32_t>(overflow, row[col]));
        row[col] = static_cast<uint16_t>(row[col] - reducible);
        overflow -= reducible;
      }
      assigned = kGolombRowSum;
    }

    while (assigned < kGolombRowSum)
    {
      std::size_t best_col = kGolombColumnCount - 1;
      uint64_t best_remainder = 0;
      for (std::size_t col = 0; col < kGolombColumnCount; ++col)
      {
        if (row[col] >= kGolombRowSum)
          continue;
        if (remainders[col] > best_remainder)
        {
          best_remainder = remainders[col];
          best_col = col;
        }
      }
      if (best_remainder == 0)
      {
        for (std::size_t col = 0; col < kGolombColumnCount; ++col)
        {
          if (row[col] >= kGolombRowSum)
            continue;
          best_col = col;
          if (hist[col] > 0)
            break;
        }
      }
      ++row[best_col];
      if (remainders[best_col] > 0)
        --remainders[best_col];
      ++assigned;
    }
    return true;
  }

  inline constexpr int A_SHIFT = 2; // fixed-point fraction
  inline constexpr int A_BIAS = 1 << (A_SHIFT - 1);
  inline constexpr int reduce_a(int a)
  {
    return (a + A_BIAS) >> A_SHIFT;
  }
  inline constexpr int mix_a_m(int a, int m)
  {
    if (a < m)
      return ((m << A_SHIFT) * 3 + a * 5 + 4) >> 3; // mix 3:5
    else
      return ((m << A_SHIFT) + a * 3 + 2) >> 2; // mix 25% of m and 75% of a
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
    writer.put_upto8(1, 1);
    writer.put_upto8(1, 1);
    put_gamma(writer, count - 2);
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
    return 2u + gamma_bits(count - 2);
  }

  uint64_t estimate_plain_with_row(const int16_t *values, uint32_t count, int row)
  {
    if (count == 0)
      return 0;
    ensure_table_initialized();
    uint64_t bits = 0;
    int a = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
      const int e = static_cast<int>(values[i]);
      const uint32_t m = (e >= 0) ? static_cast<uint32_t>(2 * e) : static_cast<uint32_t>(-2 * e - 1);
      const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
      const uint32_t q = (k > 0) ? (m >> k) : m;
      if (q >= static_cast<uint32_t>(kGolombGiveUpQ))
      {
        const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
        const uint32_t direct_value = (m >= base) ? (m - base + 1u) : 1u;
        bits += static_cast<uint64_t>(base) + gamma_bits(direct_value);
      }
      else
        bits += q + 1u + static_cast<uint32_t>(k);
      a = mix_a_m(a, m);
    }
    return bits;
  }

  uint64_t estimate_plain_component(const int16_t *values, uint32_t count, uint32_t component)
  {
    const int row = golomb_row_index(GolombCodingKind::Plain, component);
    return estimate_plain_with_row(values, count, row);
  }

  uint64_t estimate_run_length_with_row(const int16_t *values, uint32_t count, int row)
  {
    if (count == 0)
      return 0;
    ensure_table_initialized();
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
          const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
          const uint32_t q = (k > 0) ? (m >> k) : m;
          if (q >= static_cast<uint32_t>(kGolombGiveUpQ))
          {
            const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
            const uint32_t direct_value = (m >= base) ? (m - base + 1u) : 1u;
            bits += static_cast<uint64_t>(base) + gamma_bits(direct_value);
          }
          else
            bits += q + 1u + static_cast<uint32_t>(k);
          a = mix_a_m(a, m);
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

  uint64_t estimate_run_length_component(const int16_t *values, uint32_t count, uint32_t component)
  {
    const int row = golomb_row_index(GolombCodingKind::RunLength, component);
    return estimate_run_length_with_row(values, count, row);
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
      const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
      const uint32_t q = (k > 0) ? (m >> k) : m;
      // printf("P e=%d q=%d m=%d, k=%d\n", (int)e, (int)(kGolombGiveUpQ > q ? q : kGolombGiveUpQ), (int)m, (int)k);
      if (q >= static_cast<uint32_t>(kGolombGiveUpQ))
      {
        const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
        write_zero_bits(writer, base);
        const uint32_t direct_value = (m >= base) ? (m - base + 1u) : 1u;
        put_gamma(writer, direct_value);
      }
      else
      {
        write_zero_bits(writer, q);
        writer.put_upto8(1, 1);
        if (k)
        {
          const uint32_t mask = (static_cast<uint32_t>(1u) << k) - 1u;
          write_bits(writer, m & mask, static_cast<unsigned>(k));
        }
      }
      a = mix_a_m(a, m);
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
          const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
          const uint32_t q = (k > 0) ? (m >> k) : m;
          // printf("R e=%d q=%d m=%d k=%d\n", (int)values[j], (int)(kGolombGiveUpQ > q ? q : kGolombGiveUpQ), (int)m, (int)k);
          if (q >= static_cast<uint32_t>(kGolombGiveUpQ))
          {
            const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
            write_zero_bits(writer, base);
            const uint32_t direct_value = (m >= base) ? (m - base + 1u) : 1u;
            put_gamma(writer, direct_value);
          }
          else
          {
            write_zero_bits(writer, q);
            writer.put_upto8(1, 1);
            if (k)
            {
              const uint32_t mask = (static_cast<uint32_t>(1u) << k) - 1u;
              write_bits(writer, m & mask, static_cast<unsigned>(k));
            }
          }
          a = mix_a_m(a, m);
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
    uint32_t gamma = 0;
    if (!read_gamma(reader, gamma))
      return 0;
    return static_cast<int>(gamma + 2);
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
      const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
      uint32_t q = 0;
      while (true)
      {
        uint32_t bit = 0;
        if (!read_bit(reader, bit))
          return false;
        if (bit)
          break;
        ++q;
        if (q >= kGolombGiveUpQ)
          break;
      }
      const bool use_direct = (q >= static_cast<uint32_t>(kGolombGiveUpQ));
      uint32_t m = 0;
      if (use_direct)
      {
        uint32_t direct_value = 0;
        if (!read_gamma(reader, direct_value) || direct_value == 0)
          return false;
        const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
        m = direct_value + base - 1u;
      }
      else
      {
        uint32_t remainder = 0;
        if (k > 0 && !read_bits(reader, static_cast<unsigned>(k), remainder))
          return false;
        m = (q << k) + remainder;
      }
      const int residual = static_cast<int>((m >> 1) ^ -static_cast<int>(m & 1u));
      dst[produced] = static_cast<int16_t>(residual);

      // printf("P e=%d q=%d m=%d, k=%d\n", (int)residual, (int)q, (int)m, (int)k);
      a = mix_a_m(a, m);
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
        const int k = g_bit_length_table[static_cast<uint32_t>(reduce_a(a))][row];
        uint32_t q = 0;
        while (true)
        {
          uint32_t bit = 0;
          if (!read_bit(reader, bit))
            return false;
          if (bit)
            break;
          ++q;
          if (q >= kGolombGiveUpQ)
            break;
        }
        const bool use_direct = (q >= static_cast<uint32_t>(kGolombGiveUpQ));
        uint32_t m = 0;
        if (use_direct)
        {
          uint32_t direct_value = 0;
          if (!read_gamma(reader, direct_value) || direct_value == 0)
            return false;
          const uint32_t base = static_cast<uint32_t>(kGolombGiveUpQ);
          m = direct_value + base - 1u;
        }
        else
        {
          uint32_t remainder = 0;
          if (k > 0 && !read_bits(reader, static_cast<unsigned>(k), remainder))
            return false;
          m = (q << k) + remainder;
        }
        const int sign = static_cast<int>(m & 1u) - 1;
        const int vv = static_cast<int>(m >> 1);
        const int residual = (vv ^ sign) + sign + 1; // 符号化時に -1 しているので、ここで戻す

        // printf("R e=%d q=%d m=%d, k=%d\n", (int)residual, (int)q, (int)m, (int)k);
        dst[produced++] = static_cast<int16_t>(residual);
        a = mix_a_m(a, m);
      }
      expect_nonzero = false;
    }
    return true;
  }

  uint64_t estimate_plain(const component_colors &colors,
                          uint32_t components,
                          uint32_t value_count,
                          bool uses_interleave)
  {
    if (!uses_interleave)
    {
      uint64_t total = 0;
      for (uint32_t c = 0; c < components; ++c)
        total += estimate_plain_component(colors.values[c].data(), value_count, c);
      return total;
    }

    if (components == 0 || value_count == 0)
      return 0;

    constexpr std::size_t kMaxCombined = kMaxBlockPixels * 4u;
    std::array<int16_t, kMaxCombined> combined{};
    std::size_t offset = 0;
    const uint32_t available = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    // インターリーブ時は専用の行へ集約して符号化するため、推定でも一括して評価する。
    for (uint32_t c = 0; c < available; ++c)
    {
      const std::size_t required = static_cast<std::size_t>(value_count);
      std::copy_n(colors.values[c].begin(), value_count, combined.begin() + offset);
      offset += required;
    }
    return estimate_plain_with_row(combined.data(), static_cast<uint32_t>(offset),
                                   static_cast<int>(kInterleavedPlainRow));
  }

  uint64_t estimate_run_length(const component_colors &colors,
                               uint32_t components,
                               uint32_t value_count,
                               bool uses_interleave)
  {
    if (!uses_interleave)
    {
      uint64_t total = 0;
      for (uint32_t c = 0; c < components; ++c)
        total += estimate_run_length_component(colors.values[c].data(), value_count, c);
      return total;
    }

    if (components == 0 || value_count == 0)
      return 0;

    constexpr std::size_t kMaxCombined = kMaxBlockPixels * 4u;
    std::array<int16_t, kMaxCombined> combined{};
    std::size_t offset = 0;
    const uint32_t available = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    // ランレングスでもインターリーブ時は専用の行へまとめるので、推定も同じ行を用いる。
    for (uint32_t c = 0; c < available; ++c)
    {
      const std::size_t required = static_cast<std::size_t>(value_count);
      std::copy_n(colors.values[c].begin(), value_count, combined.begin() + offset);
      offset += required;
    }
    return estimate_run_length_with_row(combined.data(), static_cast<uint32_t>(offset),
                                        static_cast<int>(kInterleavedRunLengthRow));
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
    std::vector<std::size_t> parsed_line_numbers;
    parsed_line_numbers.reserve(static_cast<std::size_t>(kGolombRowCount));

    while (std::getline(in, line))
    {
      ++line_number;
      const auto comment_pos = line.find_first_of("#;");
      if (comment_pos != std::string::npos)
        line.erase(comment_pos);

      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        continue;

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
      parsed_line_numbers.push_back(line_number);
    }

    if (parsed_rows.empty())
    {
      err = "tlg8: Golomb table '" + path + "' must contain at least one row";
      return false;
    }

    std::vector<GolombRow> expanded_rows;
    const std::size_t parsed_count = parsed_rows.size();
    if (parsed_count > static_cast<std::size_t>(kGolombRowCount))
    {
      const std::size_t offending_row = static_cast<std::size_t>(kGolombRowCount);
      const std::size_t offending_line = (offending_row < parsed_line_numbers.size())
                                           ? parsed_line_numbers[offending_row]
                                           : line_number;
      err = "tlg8: extra data in Golomb table '" + path + "' at line " + std::to_string(offending_line);
      return false;
    }

    if (parsed_count == static_cast<std::size_t>(kGolombRowCount))
    {
      expanded_rows = parsed_rows;
    }
    else if (parsed_count == kLegacyGolombRowCount)
    {
      // 旧形式のテーブルに対応するため、専用のインターリーブ行を追加する
      expanded_rows = parsed_rows;
      expanded_rows.push_back(parsed_rows[0]);
      expanded_rows.push_back(parsed_rows[3]);
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
  bool rebuild_golomb_table_from_histogram(const golomb_histogram &histogram)
  {
    GolombTable candidate = g_golomb_table;
    bool changed = false;
    for (std::size_t row = 0; row < candidate.size(); ++row)
    {
      GolombRow new_row{};
      if (!normalize_histogram_row(histogram[row], new_row))
        continue;
      if (candidate[row] != new_row)
      {
        candidate[row] = new_row;
        changed = true;
      }
    }
    if (changed)
    {
      g_golomb_table = candidate;
      g_table_ready = false;
    }
    return changed;
  }

  const golomb_table_counts &current_golomb_table()
  {
    return g_golomb_table;
  }

  bool apply_golomb_table(const golomb_table_counts &table)
  {
    if (table != g_golomb_table)
    {
      g_golomb_table = table;
      g_table_ready = false;
      return true;
    }
    return false;
  }

  uint64_t estimate_row_bits(GolombCodingKind kind,
                             uint32_t component,
                             const int16_t *values,
                             uint32_t count)
  {
    if (kind == GolombCodingKind::Plain)
      return estimate_plain_component(values, count, component);
    return estimate_run_length_component(values, count, component);
  }

  int golomb_row_index(GolombCodingKind kind, uint32_t component)
  {
    if (component == kInterleavedComponentIndex)
    {
      return (kind == GolombCodingKind::Plain) ? static_cast<int>(kInterleavedPlainRow)
                                               : static_cast<int>(kInterleavedRunLengthRow);
    }
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
    case kInterleavedPlainRow:
      return GolombCodingKind::Plain;
    case 3:
    case 4:
    case 5:
    case kInterleavedRunLengthRow:
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
    case kInterleavedPlainRow:
    case kInterleavedRunLengthRow:
      return kInterleavedComponentIndex;
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
