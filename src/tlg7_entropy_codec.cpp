#include "tlg7_entropy_codec.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tlg::v7
{
  namespace
  {
    constexpr int GOLOMB_COMPONENT_ROW_COUNT = 3;
    constexpr int GOLOMB_MODE_COUNT = 2;
    constexpr int GOLOMB_N_COUNT = GOLOMB_COMPONENT_ROW_COUNT * GOLOMB_MODE_COUNT;
    constexpr int GOLOMB_ROW_SUM = 1024;

    using GolombRow = std::array<uint16_t, 9>;
    using GolombTable = std::array<GolombRow, GOLOMB_N_COUNT>;

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
0 4 4 7 24 89 270 489 137
2 2 5 13 67 98 230 476 131
3 2 5 15 77 92 238 462 130
2 2 4 10 51 108 237 482 128
2 3 7 33 74 81 237 450 137
3 1 5 28 66 92 246 452 131
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
        while (index < sizeof(data) - 1 && is_digit(data[index]))
        {
          value = static_cast<uint16_t>(value * 10 + static_cast<uint16_t>(data[index] - '0'));
          ++index;
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

    static_assert(GOLOMB_N_COUNT > 0, "invalid Golomb row count");

    inline constexpr GolombTable DEFAULT_GOLOMB_TABLE = []() constexpr
    {
      constexpr auto parsed = parse_default_golomb_table();
      static_assert(parsed.filled, "DEFAULT_GOLOMB_TABLE data is incomplete");
      return parsed.table;
    }();

    GolombTable g_golomb_table = DEFAULT_GOLOMB_TABLE;
    unsigned char GolombBitLengthTable[GOLOMB_ROW_SUM][GOLOMB_N_COUNT];
    bool golomb_tables_ready = false;

    inline int component_row_index(std::size_t component_index, GolombCodingKind kind)
    {
      if (GOLOMB_COMPONENT_ROW_COUNT <= 0)
        return 0;
      const std::size_t max_index = static_cast<std::size_t>(GOLOMB_COMPONENT_ROW_COUNT - 1);
      const std::size_t clamped = (component_index < max_index) ? component_index : max_index;
      const std::size_t mode_offset = (kind == GolombCodingKind::Plain)
                                          ? static_cast<std::size_t>(GOLOMB_COMPONENT_ROW_COUNT)
                                          : 0u;
      return static_cast<int>(clamped + mode_offset);
    }

    inline void init_golomb_tables()
    {
      if (golomb_tables_ready)
        return;
      for (int n = 0; n < GOLOMB_N_COUNT; ++n)
      {
        int a = 0;
        for (int i = 0; i < 9; ++i)
        {
          const uint16_t count = g_golomb_table[static_cast<std::size_t>(n)][static_cast<std::size_t>(i)];
          for (uint16_t j = 0; j < count; ++j)
          {
            const int idx = std::min(a, GOLOMB_ROW_SUM - 1);
            GolombBitLengthTable[idx][n] = static_cast<unsigned char>(i);
            ++a;
          }
        }
        assert(a == GOLOMB_ROW_SUM);
      }
      golomb_tables_ready = true;
    }

    class GolombBitStream
    {
    public:
      explicit GolombBitStream(std::vector<uint8_t> &out) : out_(out) {}
      ~GolombBitStream() { Flush(); }

      size_t GetBytePos() const { return byte_pos_; }
      size_t GetBitLength() const { return byte_pos_ * 8 + bit_pos_; }

      void Put1Bit(bool bit)
      {
        ensure_capacity();
        if (bit)
          buffer_[byte_pos_] |= static_cast<uint8_t>(1u << bit_pos_);
        ++bit_pos_;
        if (bit_pos_ == 8)
        {
          bit_pos_ = 0;
          ++byte_pos_;
        }
      }

      void PutValue(long value, int len)
      {
        for (int i = 0; i < len; ++i)
          Put1Bit((value >> i) & 1);
      }

      void PutGamma(int v)
      {
        if (v <= 0)
          return;
        int t = v >> 1;
        int cnt = 0;
        while (t)
        {
          Put1Bit(0);
          t >>= 1;
          ++cnt;
        }
        Put1Bit(1);
        while (cnt--)
        {
          Put1Bit(v & 1);
          v >>= 1;
        }
      }

      // Encode zero-run lengths using a small fixed prefix table for short runs
      // Codebook (prefix bits written in this order):
      // 1 -> 0
      // 2 -> 10
      // 3 -> 110
      // >=4 -> 111 + gamma(count-3)
      void PutRunLength(int count)
      {
        if (count <= 0)
          return;
        if (count == 1)
        {
          Put1Bit(0);
          return;
        }
        if (count == 2)
        {
          Put1Bit(1);
          Put1Bit(0);
          return;
        }
        if (count == 3)
        {
          Put1Bit(1);
          Put1Bit(1);
          Put1Bit(0);
          return;
        }
        // count >= 4
        Put1Bit(1);
        Put1Bit(1);
        Put1Bit(1);
        PutGamma(count - 3);
      }

      void Flush()
      {
        const size_t bytes = byte_pos_ + (bit_pos_ ? 1 : 0);
        if (bytes)
        {
          ensure_capacity();
          out_.insert(out_.end(), buffer_.begin(), buffer_.begin() + bytes);
        }
        buffer_.clear();
        byte_pos_ = 0;
        bit_pos_ = 0;
      }

    private:
      void ensure_capacity()
      {
        if (buffer_.size() <= byte_pos_)
          buffer_.resize(byte_pos_ + 1, 0);
      }

      std::vector<uint8_t> &out_;
      std::vector<uint8_t> buffer_;
      size_t byte_pos_ = 0;
      int bit_pos_ = 0;
    };

    void compress_residuals_run_length(GolombBitStream &bs,
                                       const std::vector<int16_t> &buf,
                                       std::size_t component_index)
    {
      if (buf.empty())
        return;

      init_golomb_tables();

      bs.PutValue(buf[0] ? 1 : 0, 1);
      const int row_index = component_row_index(component_index, GolombCodingKind::RunLength);
      int a = 0;
      int count = 0;
      const size_t size = buf.size();

      for (size_t i = 0; i < size; ++i)
      {
        long e = buf[i];
        if (e != 0)
        {
          if (count)
          {
            bs.PutRunLength(count);
            count = 0;
          }

          size_t ii = i;
          while (ii < size && buf[ii] != 0)
            ++ii;
          const size_t nonzero_count = ii - i;
          bs.PutGamma(static_cast<int>(nonzero_count));

          for (; i < ii; ++i)
          {
            e = buf[i];
            long m = ((e >= 0) ? (2 * e) : (-2 * e - 1)) - 1;
            if (m < 0)
              m = 0;
            int k = GolombBitLengthTable[a][row_index];
            long q = (k > 0) ? (m >> k) : m;
            for (; q > 0; --q)
              bs.Put1Bit(0);
            bs.Put1Bit(1);
            if (k)
              bs.PutValue(m & ((1 << k) - 1), k);
            a = (m + a + 1) >> 1;
          }
          i = ii - 1;
        }
        else
        {
          ++count;
        }
      }

      if (count)
        bs.PutRunLength(count);
    }

    void compress_residuals_plain(GolombBitStream &bs,
                                  const std::vector<int16_t> &buf,
                                  std::size_t component_index)
    {
      if (buf.empty())
        return;

      init_golomb_tables();

      const int row_index = component_row_index(component_index, GolombCodingKind::Plain);
      int a = 0;
      for (const auto value : buf)
      {
        const int e = static_cast<int>(value);
        const uint32_t m = ((e >= 0) ? (2 * e) : (-2 * e - 1));
        const int k = GolombBitLengthTable[a][row_index];
        uint32_t q = (k > 0) ? (m >> k) : m;
        while (q--)
          bs.Put1Bit(0);
        bs.Put1Bit(1);
        if (k)
        {
          const uint32_t mask = (static_cast<uint32_t>(1u) << k) - 1u;
          bs.PutValue(static_cast<long>(m & mask), k);
        }
        a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
      }
    }

    void compress_residuals(GolombBitStream &bs,
                            const std::vector<int16_t> &buf,
                            std::size_t component_index,
                            GolombCodingKind kind)
    {
      switch (kind)
      {
      case GolombCodingKind::Plain:
        compress_residuals_plain(bs, buf, component_index);
        break;
      case GolombCodingKind::RunLength:
      default:
        compress_residuals_run_length(bs, buf, component_index);
        break;
      }
    }

    struct GolombDecoder
    {
      GolombDecoder() = default;

      GolombDecoder(const uint8_t *data, size_t size)
      {
        reset(data, size);
      }

      void reset(const uint8_t *data, size_t size)
      {
        data_ = data;
        size_ = size;
        byte_pos_ = 0;
        bits_ = 0;
        bit_pos_ = 0;
        fill();
      }

      void fill()
      {
        while (bit_pos_ <= 24 && byte_pos_ < size_)
        {
          bits_ |= static_cast<uint32_t>(data_[byte_pos_++]) << bit_pos_;
          bit_pos_ += 8;
        }
      }

      bool read_bit(uint32_t &bit)
      {
        fill();
        if (bit_pos_ <= 0)
          return false;
        bit = bits_ & 1u;
        bits_ >>= 1;
        --bit_pos_;
        return true;
      }

      bool read_bits(unsigned count, uint32_t &value)
      {
        value = 0;
        for (unsigned i = 0; i < count; ++i)
        {
          uint32_t bit = 0;
          if (!read_bit(bit))
            return false;
          value |= (bit << i);
        }
        return true;
      }

      int read_gamma()
      {
        uint32_t bit = 0;
        unsigned zeros = 0;
        while (true)
        {
          if (!read_bit(bit))
            return 0;
          if (bit)
            break;
          ++zeros;
        }
        int value = 1 << zeros;
        if (zeros)
        {
          uint32_t suffix = 0;
          if (!read_bits(zeros, suffix))
            return 0;
          value += static_cast<int>(suffix);
        }
        return value;
      }

      // Read run-length encoded with the fixed small table defined in PutRunLength.
      // Returns 0 on error, otherwise the decoded count.
      int read_run_length()
      {
        // Peek bits to decide
        uint32_t bit = 0;
        if (!read_bit(bit))
          return 0;
        if (bit == 0)
        {
          // code '0' => count == 1
          return 1;
        }
        // first bit was 1, need to read next two bits to distinguish 2/3/>=4
        if (!read_bit(bit))
          return 0;
        if (bit == 0)
        {
          // '10' => count == 2
          return 2;
        }
        // got '11', read one more bit
        if (!read_bit(bit))
          return 0;
        if (bit == 0)
        {
          // '110' => count == 3
          return 3;
        }
        // '111' prefix: read gamma for (count-3)
        int v = read_gamma();
        if (v <= 0)
          return 0;
        return v + 3;
      }

      uint32_t bits_ = 0;
      int bit_pos_ = 0;

    private:
      const uint8_t *data_ = nullptr;
      size_t size_ = 0;
      size_t byte_pos_ = 0;

    public:
      [[nodiscard]] bool exhausted() const noexcept
      {
        return byte_pos_ >= size_ && bit_pos_ <= 0;
      }
    };

    bool decode_residuals_run_length(GolombDecoder &decoder,
                                     size_t expected_count,
                                     std::vector<int16_t> &out,
                                     std::size_t component_index)
    {
      out.clear();
      out.reserve(expected_count);

      if (expected_count == 0)
        return true;

      init_golomb_tables();

      uint32_t first_bit = 0;
      if (!decoder.read_bit(first_bit))
        return false;

      bool expect_nonzero = (first_bit != 0);
      int a = 0;
      const int row_index = component_row_index(component_index, GolombCodingKind::RunLength);

      while (out.size() < expected_count)
      {
        int run = 0;
        if (!expect_nonzero)
        {
          run = decoder.read_run_length();
        }
        else
        {
          run = decoder.read_gamma();
        }
        if (run <= 0)
          return false;

        const size_t remaining = expected_count - out.size();
        if (!expect_nonzero)
        {
          if (static_cast<size_t>(run) > remaining)
            return false;
          out.insert(out.end(), static_cast<std::size_t>(run), 0);
          expect_nonzero = true;
          continue;
        }

        if (static_cast<size_t>(run) > remaining)
          return false;

        for (int i = 0; i < run; ++i)
        {
          const int k = GolombBitLengthTable[a][row_index];
          int q = 0;
          while (true)
          {
            uint32_t bit = 0;
            if (!decoder.read_bit(bit))
              return false;
            if (bit)
              break;
            ++q;
          }
          uint32_t remainder_bits = 0;
          if (k > 0 && !decoder.read_bits(static_cast<unsigned>(k), remainder_bits))
            return false;
          const int m = (q << k) + static_cast<int>(remainder_bits);
          const int sign = (m & 1) - 1;
          const int vv = m >> 1;
          const int residual = (vv ^ sign) + sign + 1;

          out.push_back(static_cast<int16_t>(residual));
          a = (m + a + 1) >> 1;
        }

        expect_nonzero = false;
      }

      return out.size() == expected_count;
    }

    bool decode_residuals_plain(GolombDecoder &decoder,
                                size_t expected_count,
                                std::vector<int16_t> &out,
                                std::size_t component_index)
    {
      out.clear();
      out.reserve(expected_count);

      if (expected_count == 0)
        return true;

      init_golomb_tables();

      int a = 0;
      const int row_index = component_row_index(component_index, GolombCodingKind::Plain);

      while (out.size() < expected_count)
      {
        const int k = GolombBitLengthTable[a][row_index];
        int q = 0;
        while (true)
        {
          uint32_t bit = 0;
          if (!decoder.read_bit(bit))
            return false;
          if (bit)
            break;
          ++q;
        }

        uint32_t remainder_bits = 0;
        if (k > 0 && !decoder.read_bits(static_cast<unsigned>(k), remainder_bits))
          return false;

        const uint32_t m = (static_cast<uint32_t>(q) << static_cast<unsigned>(k)) + remainder_bits;
        const int residual = static_cast<int>((m >> 1) ^ -static_cast<int>(m & 1u));

        out.push_back(static_cast<int16_t>(residual));
        a = static_cast<int>((m + static_cast<uint32_t>(a) + 1u) >> 1);
      }

      return out.size() == expected_count;
    }

    bool decode_residuals(GolombDecoder &decoder,
                          size_t expected_count,
                          std::vector<int16_t> &out,
                          std::size_t component_index,
                          GolombCodingKind kind)
    {
      switch (kind)
      {
      case GolombCodingKind::Plain:
        return decode_residuals_plain(decoder, expected_count, out, component_index);
      case GolombCodingKind::RunLength:
      default:
        return decode_residuals_run_length(decoder, expected_count, out, component_index);
      }
    }

  } // namespace

  struct GolombResidualEntropyDecoder::Impl
  {
    GolombDecoder decoder;
  };

  void GolombResidualEntropyEncoder::encode(const std::vector<int16_t> &residuals,
                                            std::vector<uint8_t> &out,
                                            uint32_t &bit_length)
  {
    struct Candidate
    {
      GolombCodingKind kind = GolombCodingKind::RunLength;
      uint32_t bits = 0;
      std::vector<uint8_t> data;
    };

    const auto make_candidate = [&](GolombCodingKind kind)
    {
      Candidate candidate;
      candidate.kind = kind;
      GolombBitStream bs(candidate.data);
      compress_residuals(bs, residuals, component_index_, kind);
      candidate.bits = static_cast<uint32_t>(bs.GetBitLength());
      bs.Flush();
      const std::size_t expected_bytes = (candidate.bits + 7u) / 8u;
      if (candidate.data.size() != expected_bytes)
        candidate.data.resize(expected_bytes);
      return candidate;
    };

    Candidate run_candidate = make_candidate(GolombCodingKind::RunLength);
    Candidate plain_candidate = make_candidate(GolombCodingKind::Plain);

    Candidate *best = &run_candidate;
    if (plain_candidate.bits < run_candidate.bits)
      best = &plain_candidate;

    last_mode_ = best->kind;
    bit_length = best->bits;
    out = std::move(best->data);
  }

  std::uint64_t GolombResidualEntropyEncoder::estimate_bits(const std::vector<int16_t> &residuals) const
  {
    auto compute_bits = [&](GolombCodingKind kind)
    {
      std::vector<uint8_t> tmp;
      GolombBitStream bs(tmp);
      compress_residuals(bs, residuals, component_index_, kind);
      return bs.GetBitLength();
    };

    const std::uint64_t rle_bits = compute_bits(GolombCodingKind::RunLength);
    const std::uint64_t plain_bits = compute_bits(GolombCodingKind::Plain);
    return std::min(rle_bits, plain_bits);
  }

  GolombResidualEntropyDecoder::GolombResidualEntropyDecoder()
      : decoder_impl_(std::make_unique<Impl>())
  {
  }

  GolombResidualEntropyDecoder::~GolombResidualEntropyDecoder() = default;

  void GolombResidualEntropyDecoder::init_stream(const uint8_t *data, std::size_t size)
  {
    if (!decoder_impl_)
      decoder_impl_ = std::make_unique<Impl>();
    decoder_impl_->decoder.reset(data, size);
    decoder_initialized_ = true;
  }

  bool GolombResidualEntropyDecoder::decode_next(std::size_t expected_count,
                                                 std::vector<int16_t> &out)
  {
    if (!decoder_initialized_ || !decoder_impl_)
      return false;
    return decode_residuals(decoder_impl_->decoder, expected_count, out, component_index_, coding_kind_);
  }

  bool GolombResidualEntropyDecoder::stream_consumed() const noexcept
  {
    if (!decoder_impl_)
      return true;
    return decoder_impl_->decoder.exhausted();
  }

  bool GolombResidualEntropyDecoder::decode(const uint8_t *data,
                                            std::size_t size,
                                            std::size_t expected_count,
                                            std::vector<int16_t> &out)
  {
    init_stream(data, size);
    return decode_next(expected_count, out);
  }

  bool configure_golomb_table(const std::string &path, std::string &err)
  {
    if (path.empty())
    {
      if (g_golomb_table != DEFAULT_GOLOMB_TABLE)
      {
        g_golomb_table = DEFAULT_GOLOMB_TABLE;
        golomb_tables_ready = false;
      }
      err.clear();
      return true;
    }

    std::ifstream in(path);
    if (!in)
    {
      err = "tlg7: failed to open Golomb table: " + path;
      return false;
    }

    std::string line;
    std::size_t line_number = 0;
    std::vector<GolombRow> parsed_rows;
    parsed_rows.reserve(static_cast<std::size_t>(GOLOMB_N_COUNT));

    while (std::getline(in, line))
    {
      ++line_number;
      const auto comment_pos = line.find_first_of("#;");
      if (comment_pos != std::string::npos)
        line.erase(comment_pos);

      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        continue;

      if (parsed_rows.size() >= static_cast<std::size_t>(GOLOMB_N_COUNT))
      {
        err = "tlg7: extra data in Golomb table '" + path + "' at line " + std::to_string(line_number);
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
          err = "tlg7: negative value in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
          return false;
        }
        if (value > std::numeric_limits<uint16_t>::max())
        {
          err = "tlg7: value too large in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
          return false;
        }
        row_values[col++] = static_cast<uint16_t>(value);
        sum += value;
      }

      if (col != row_values.size())
      {
        if (iss.fail() && !iss.eof())
        {
          err = "tlg7: invalid token in Golomb table '" + path + "' at line " + std::to_string(line_number);
        }
        else
        {
          err = "tlg7: expected 9 values in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        }
        return false;
      }

      if (iss >> value)
      {
        err = "tlg7: too many values in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        return false;
      }

      if (sum != GOLOMB_ROW_SUM)
      {
        err = "tlg7: row sum must be 1024 in Golomb table '" + path + "' at row " + std::to_string(row_index + 1);
        return false;
      }

      parsed_rows.push_back(row_values);
    }

    if (parsed_rows.empty())
    {
      err = "tlg7: Golomb table '" + path + "' must contain at least one row";
      return false;
    }

    std::vector<GolombRow> expanded_rows;
    const std::size_t parsed_count = parsed_rows.size();
    if (parsed_count == static_cast<std::size_t>(GOLOMB_N_COUNT))
    {
      expanded_rows = parsed_rows;
    }
    else if (static_cast<std::size_t>(GOLOMB_N_COUNT) % parsed_count == 0)
    {
      const std::size_t repeat = static_cast<std::size_t>(GOLOMB_N_COUNT) / parsed_count;
      expanded_rows.reserve(static_cast<std::size_t>(GOLOMB_N_COUNT));
      for (std::size_t i = 0; i < repeat; ++i)
        expanded_rows.insert(expanded_rows.end(), parsed_rows.begin(), parsed_rows.end());
    }
    else
    {
      err = "tlg7: Golomb table '" + path + "' must contain " + std::to_string(GOLOMB_N_COUNT) +
            " rows or a divisor of that count";
      return false;
    }

    GolombTable candidate{};
    for (std::size_t i = 0; i < static_cast<std::size_t>(GOLOMB_N_COUNT); ++i)
      candidate[i] = expanded_rows[i];

    if (candidate != g_golomb_table)
    {
      g_golomb_table = candidate;
      golomb_tables_ready = false;
    }
    err.clear();
    return true;
  }

} // namespace tlg::v7
