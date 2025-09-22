#include "tlg7_entropy_codec.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace tlg::v7
{
  namespace
  {
    constexpr int GOLOMB_N_COUNT = 2;
    constexpr int GOLOMB_ROW_SUM = 1024;

    using GolombRow = std::array<uint16_t, 9>;
    using GolombTable = std::array<GolombRow, GOLOMB_N_COUNT>;

    constexpr GolombTable DEFAULT_GOLOMB_TABLE = {
        GolombRow{3, 6, 15, 23, 54, 130, 261, 518, 14},
        GolombRow{2, 4, 9, 16, 60, 145, 269, 514, 5},
    };

    GolombTable g_golomb_table = DEFAULT_GOLOMB_TABLE;
    unsigned char GolombBitLengthTable[GOLOMB_ROW_SUM][GOLOMB_N_COUNT];
    bool golomb_tables_ready = false;

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

    void compress_residuals_golomb(GolombBitStream &bs, const std::vector<int16_t> &buf)
    {
      if (buf.empty())
        return;

      init_golomb_tables();

      bs.PutValue(buf[0] ? 1 : 0, 1);
      int n = GOLOMB_N_COUNT - 1;
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
            bs.PutGamma(count);
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
            int k = GolombBitLengthTable[a][n];
            long q = (k > 0) ? (m >> k) : m;
            for (; q > 0; --q)
              bs.Put1Bit(0);
            bs.Put1Bit(1);
            if (k)
              bs.PutValue(m & ((1 << k) - 1), k);
            a += static_cast<int>(m);
            if (--n < 0)
            {
              n = GOLOMB_N_COUNT - 1;
              a >>= 1;
            }
          }
          i = ii - 1;
        }
        else
        {
          ++count;
        }
      }

      if (count)
        bs.PutGamma(count);
    }

    struct GolombDecoder
    {
      GolombDecoder(const uint8_t *data, size_t size)
          : data_(data), size_(size) {}

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

      uint32_t bits_ = 0;
      int bit_pos_ = 0;

    private:
      const uint8_t *data_ = nullptr;
      size_t size_ = 0;
      size_t byte_pos_ = 0;
    };

    bool decode_residuals_golomb(const uint8_t *data,
                                 size_t size,
                                 size_t expected_count,
                                 std::vector<int16_t> &out)
    {
      out.clear();
      out.reserve(expected_count);

      if (expected_count == 0)
        return true;

      init_golomb_tables();

      GolombDecoder decoder(data, size);
      decoder.fill();

      uint32_t first_bit = 0;
      if (!decoder.read_bit(first_bit))
        return false;

      bool expect_nonzero = (first_bit != 0);
      int a = 0;
      int n = GOLOMB_N_COUNT - 1;

      while (out.size() < expected_count)
      {
        int run = decoder.read_gamma();
        if (run <= 0)
          return false;

        const size_t remaining = expected_count - out.size();
        if (!expect_nonzero)
        {
          if (static_cast<size_t>(run) > remaining)
            return false;
          out.insert(out.end(), run, 0);
          expect_nonzero = true;
          continue;
        }

        if (static_cast<size_t>(run) > remaining)
          return false;

        for (int i = 0; i < run; ++i)
        {
          int k = GolombBitLengthTable[a][n];
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
          int m = (q << k) + static_cast<int>(remainder_bits);
          int sign = (m & 1) - 1;
          int vv = m >> 1;
          int residual = (vv ^ sign) + sign + 1;
          a += m;

          out.push_back(static_cast<int16_t>(residual));

          if (--n < 0)
          {
            a >>= 1;
            n = GOLOMB_N_COUNT - 1;
          }
        }

        expect_nonzero = false;
      }

      return out.size() == expected_count;
    }

  } // namespace

  void GolombResidualEntropyEncoder::encode(const std::vector<int16_t> &residuals,
                                            std::vector<uint8_t> &out,
                                            uint32_t &bit_length)
  {
    out.clear();
    GolombBitStream bs(out);
    compress_residuals_golomb(bs, residuals);
    bit_length = static_cast<uint32_t>(bs.GetBitLength());
    bs.Flush();
    const std::size_t expected_bytes = (bit_length + 7u) / 8u;
    if (out.size() != expected_bytes)
      out.resize(expected_bytes);
  }

  std::uint64_t GolombResidualEntropyEncoder::estimate_bits(const std::vector<int16_t> &residuals) const
  {
    std::vector<uint8_t> tmp;
    GolombBitStream bs(tmp);
    compress_residuals_golomb(bs, residuals);
    return bs.GetBitLength();
  }

  bool GolombResidualEntropyDecoder::decode(const uint8_t *data,
                                            std::size_t size,
                                            std::size_t expected_count,
                                            std::vector<int16_t> &out)
  {
    return decode_residuals_golomb(data, size, expected_count, out);
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

    GolombTable candidate{};
    std::size_t row = 0;

    std::string line;
    std::size_t line_number = 0;

    while (std::getline(in, line))
    {
      ++line_number;
      const auto comment_pos = line.find_first_of("#;");
      if (comment_pos != std::string::npos)
        line.erase(comment_pos);

      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        continue;

      if (row >= GOLOMB_N_COUNT)
      {
        err = "tlg7: extra data in Golomb table '" + path + "' at line " + std::to_string(line_number);
        return false;
      }

      std::istringstream iss(line);
      GolombRow row_values{};
      int sum = 0;
      int value = 0;
      std::size_t col = 0;
      while (col < row_values.size() && (iss >> value))
      {
        if (value < 0)
        {
          err = "tlg7: negative value in Golomb table '" + path + "' at row " + std::to_string(row + 1);
          return false;
        }
        if (value > std::numeric_limits<uint16_t>::max())
        {
          err = "tlg7: value too large in Golomb table '" + path + "' at row " + std::to_string(row + 1);
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
          err = "tlg7: expected 9 values in Golomb table '" + path + "' at row " + std::to_string(row + 1);
        }
        return false;
      }

      if (iss >> value)
      {
        err = "tlg7: too many values in Golomb table '" + path + "' at row " + std::to_string(row + 1);
        return false;
      }

      if (sum != GOLOMB_ROW_SUM)
      {
        err = "tlg7: row sum must be 1024 in Golomb table '" + path + "' at row " + std::to_string(row + 1);
        return false;
      }

      candidate[row++] = row_values;
    }

    if (row != GOLOMB_N_COUNT)
    {
      err = "tlg7: Golomb table '" + path + "' must contain " + std::to_string(GOLOMB_N_COUNT) + " rows";
      return false;
    }

    if (candidate != g_golomb_table)
    {
      g_golomb_table = candidate;
      golomb_tables_ready = false;
    }
    err.clear();
    return true;
  }

} // namespace tlg::v7
