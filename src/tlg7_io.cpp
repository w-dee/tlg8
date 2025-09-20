#include "tlg7_io.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "tlg_io_common.h"

namespace
{
  using tlg::detail::read_exact;
  using tlg::detail::read_u32le;
  using tlg::detail::write_u32le;
}

namespace tlg::v7::detail
{

std::vector<image<uint8_t>> split_components_from_gray(const image<uint8_t> &gray)
{
  std::vector<image<uint8_t>> planes;
  if (!gray.empty())
    planes.push_back(gray);
  return planes;
}

std::vector<image<uint8_t>> split_components_from_packed(const image<uint32_t> &packed,
                                                         std::size_t component_count)
{
  std::vector<image<uint8_t>> planes;
  if (packed.empty() || component_count == 0)
    return planes;

  const std::size_t width = packed.get_width();
  const std::size_t height = packed.get_height();
  planes.reserve(component_count);
  for (std::size_t i = 0; i < component_count; ++i)
    planes.emplace_back(width, height);

  for (std::size_t y = 0; y < height; ++y)
  {
    const uint32_t *src = packed.row_ptr(y);
    for (std::size_t x = 0; x < width; ++x)
    {
      const uint32_t v = src[x];
      if (component_count >= 1)
        planes[0].row_ptr(y)[x] = static_cast<uint8_t>(v & 0xFF);           // B
      if (component_count >= 2)
        planes[1].row_ptr(y)[x] = static_cast<uint8_t>((v >> 8) & 0xFF);    // G
      if (component_count >= 3)
        planes[2].row_ptr(y)[x] = static_cast<uint8_t>((v >> 16) & 0xFF);   // R
      if (component_count >= 4)
        planes[3].row_ptr(y)[x] = static_cast<uint8_t>((v >> 24) & 0xFF);   // A
    }
  }
  return planes;
}

} // namespace tlg::v7::detail

namespace
{
  constexpr std::size_t BLOCK_SIZE = 8;
  constexpr std::size_t CHUNK_SCAN_LINES = 64;

  constexpr std::array<uint8_t, 64> HILBERT8x8 = {
      0,  1,  8,  9,  16, 17, 24, 25,
      2,  3,  10, 11, 18, 19, 26, 27,
      4,  5,  12, 13, 20, 21, 28, 29,
      6,  7,  14, 15, 22, 23, 30, 31,
      48, 49, 56, 57, 58, 59, 50, 51,
      40, 41, 32, 33, 34, 35, 42, 43,
      38, 39, 36, 37, 44, 45, 46, 47,
      52, 53, 54, 55, 62, 63, 60, 61};

  constexpr std::array<uint8_t, 64> make_hilbert_inverse()
  {
    std::array<uint8_t, 64> inv{};
    for (uint8_t i = 0; i < 64; ++i)
      inv[HILBERT8x8[i]] = i;
    return inv;
  }

  constexpr std::array<uint8_t, 64> HILBERT8x8_INV = make_hilbert_inverse();

  constexpr int GOLOMB_N_COUNT = 4;

  static const short GOLOMB_COMPRESSED[GOLOMB_N_COUNT][9] = {
      {3, 7, 15, 27, 63, 108, 223, 448, 130},
      {3, 5, 13, 24, 51, 95, 192, 384, 257},
      {2, 5, 12, 21, 39, 86, 155, 320, 384},
      {2, 3, 9, 18, 33, 61, 129, 258, 511},
  };

  static unsigned char GolombBitLengthTable[GOLOMB_N_COUNT * 2 * 128][GOLOMB_N_COUNT];
  static bool golomb_tables_ready = false;

  inline void init_golomb_tables()
  {
    if (golomb_tables_ready)
      return;
    golomb_tables_ready = true;
    int a;
    for (int n = 0; n < GOLOMB_N_COUNT; ++n)
    {
      a = 0;
      for (int i = 0; i < 9; ++i)
      {
        for (int j = 0; j < GOLOMB_COMPRESSED[n][i]; ++j)
          GolombBitLengthTable[a++][n] = static_cast<unsigned char>(i);
      }
    }
  }

  template <typename T>
  void reorder_to_hilbert(std::vector<T> &values)
  {
    if (values.size() != 64)
      return;
    std::array<T, 64> tmp{};
    for (std::size_t i = 0; i < 64; ++i)
      tmp[i] = values[HILBERT8x8[i]];
    for (std::size_t i = 0; i < 64; ++i)
      values[i] = tmp[i];
  }

  template <typename T>
  void reorder_from_hilbert(std::vector<T> &values)
  {
    if (values.size() != 64)
      return;
    std::array<T, 64> tmp{};
    for (std::size_t i = 0; i < 64; ++i)
      tmp[HILBERT8x8[i]] = values[i];
    for (std::size_t i = 0; i < 64; ++i)
      values[i] = tmp[i];
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
          a += static_cast<int>(m >> 1);
          if (--n < 0)
          {
            a >>= 1;
            n = GOLOMB_N_COUNT - 1;
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
        a += vv;

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

  class CAS8
  {
  public:
    enum class Class : uint8_t
    {
      FLAT = 0,
      HORZ = 1,
      VERT = 2,
      DIAG = 3
    };

    enum PredId : uint8_t
    {
      P0 = 0,
      P1,
      P2,
      P3,
      P4,
      P5,
      P6,
      P7,
      Pp,
      PRED_COUNT
    };

    struct Config
    {
      int T1 = 2;
      int T2 = 6;
      bool enablePlanarLite = true;
      int errDecayShift = 2;
    };

    struct State
    {
      std::array<uint16_t, PRED_COUNT> errScore{};
      void update(PredId pid, int abs_e, int decay)
      {
        const uint16_t prev = errScore[pid];
        const uint16_t dec = static_cast<uint16_t>(prev - (prev >> decay));
        errScore[pid] = static_cast<uint16_t>(dec + std::min(abs_e, 0xFFFF));
      }
    };

    CAS8(Config cfg, int lo, int hi)
        : cfg_(cfg), lo_(lo), hi_(hi) {}

    template <typename T>
    std::pair<int, PredId> predict_and_choose(int a, int b, int c, int d, int f, const State &st) const
    {
      const Class cl = classify_(a, b, c);
      const PredId pid = select_(st, cl);
      return {predict_<T>(pid, a, b, c, d, f), pid};
    }

    template <typename T>
    int predict_only(PredId pid, int a, int b, int c, int d, int f) const
    {
      return predict_<T>(pid, a, b, c, d, f);
    }

    void update_state(State &st, PredId pid, int abs_e) const
    {
      st.update(pid, abs_e, cfg_.errDecayShift);
    }

  private:
    static inline int iabs_(int v) { return v < 0 ? -v : v; }

    Class classify_(int a, int b, int c) const
    {
      const int Dh = iabs_(a - b);
      const int Dv = iabs_(b - c);
      if (Dh <= cfg_.T1 && Dv <= cfg_.T1)
        return Class::FLAT;
      if ((Dh - Dv) >= cfg_.T2)
        return Class::VERT;
      if ((Dv - Dh) >= cfg_.T2)
        return Class::HORZ;
      return Class::DIAG;
    }

    PredId select_(const State &st, Class cl) const
    {
      PredId a = P2;
      PredId b = P3;
      switch (cl)
      {
      case Class::FLAT:
        a = P2;
        b = cfg_.enablePlanarLite && st.errScore[Pp] < st.errScore[P3] ? Pp : P3;
        break;
      case Class::VERT:
        a = P0;
        b = P6;
        break;
      case Class::HORZ:
        a = P1;
        b = P7;
        break;
      case Class::DIAG:
      default:
      {
        const PredId tilt = (st.errScore[P4] <= st.errScore[P5]) ? P4 : P5;
        a = cfg_.enablePlanarLite && st.errScore[Pp] < st.errScore[P3] ? Pp : P3;
        b = tilt;
        break;
      }
      }
      return (st.errScore[a] <= st.errScore[b]) ? a : b;
    }

    template <typename T>
    int clip_(int v) const
    {
      if (v < lo_)
        v = lo_;
      else if (v > hi_)
        v = hi_;
      return static_cast<int>(static_cast<T>(v));
    }

    template <typename T>
    int predict_(PredId pid, int a, int b, int c, int d, int f) const
    {
      switch (pid)
      {
      case P0:
        return clip_<T>(a);
      case P1:
        return clip_<T>(b);
      case P2:
        return clip_<T>((a + b + 1) >> 1);
      case P3:
        return clip_<T>(a + b - c);
      case P4:
        return clip_<T>(a + ((b - c) >> 1));
      case P5:
        return clip_<T>(b + ((a - c) >> 1));
      case P6:
        return clip_<T>(((a << 1) + b - c + 2) >> 2);
      case P7:
        return clip_<T>((a + (b << 1) - c + 2) >> 2);
      case Pp:
      {
        if (!cfg_.enablePlanarLite)
          return clip_<T>(a + b - c);
        const int Hx = (a - c) + (b - d);
        const int Vy = (b - c) + (f - b);
        return clip_<T>(a + ((Hx + Vy + 2) >> 2));
      }
      default:
        return clip_<T>(b);
      }
    }

    Config cfg_;
    int lo_;
    int hi_;
  };

  inline int sample_pixel(const tlg::v7::detail::image<uint8_t> &img, int x, int y)
  {
    if (x < 0 || y < 0)
      return 0;
    const std::size_t ux = static_cast<std::size_t>(x);
    const std::size_t uy = static_cast<std::size_t>(y);
    if (ux >= img.get_width() || uy >= img.get_height())
      return 0;
    return img.row_ptr(uy)[ux];
  }

  void apply_color_filter(int code,
                          std::vector<uint8_t> &b,
                          std::vector<uint8_t> &g,
                          std::vector<uint8_t> &r)
  {
    const std::size_t n = b.size();
    switch (code)
    {
    case 0:
      return;
    case 1:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 2:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 3:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 4:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 5:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 6:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      break;
    case 7:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      break;
    case 8:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      break;
    case 9:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 10:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 11:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 12:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 13:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 14:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 15:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(b[i] << 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 16:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      break;
    case 17:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 18:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      break;
    case 19:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 20:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      break;
    case 21:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (g[i] >> 1));
      break;
    case 22:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (b[i] >> 1));
      break;
    case 23:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 24:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 25:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (r[i] >> 1));
      break;
    case 26:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (g[i] >> 1));
      break;
    case 27:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(r[i] >> 1);
        g[i] = static_cast<uint8_t>(g[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 28:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (b[i] >> 1));
      break;
    case 29:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(g[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 30:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(b[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 31:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (r[i] >> 1));
      break;
    default:
      break;
    }
  }

  void undo_color_filter(int code,
                         std::vector<uint8_t> &b,
                         std::vector<uint8_t> &g,
                         std::vector<uint8_t> &r)
  {
    const std::size_t n = b.size();
    switch (code)
    {
    case 0:
      return;
    case 1:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
      }
      break;
    case 2:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
      }
      break;
    case 3:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
      }
      break;
    case 4:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
      }
      break;
    case 5:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
      }
      break;
    case 6:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
      break;
    case 7:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
      break;
    case 8:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
      break;
    case 9:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
      }
      break;
    case 10:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
      }
      break;
    case 11:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
      }
      break;
    case 12:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
      }
      break;
    case 13:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
      }
      break;
    case 14:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
      }
      break;
    case 15:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(b[i] << 1);
        g[i] = static_cast<uint8_t>(g[i] + t);
        r[i] = static_cast<uint8_t>(r[i] + t);
      }
      break;
    case 16:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] + r[i]);
      break;
    case 17:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] + g[i]);
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
      }
      break;
    case 18:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] + b[i]);
      break;
    case 19:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
      }
      break;
    case 20:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
      break;
    case 21:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] + (g[i] >> 1));
      break;
    case 22:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] + (b[i] >> 1));
      break;
    case 23:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
      }
      break;
    case 24:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] + b[i]);
        r[i] = static_cast<uint8_t>(r[i] + g[i]);
        b[i] = static_cast<uint8_t>(b[i] + r[i]);
      }
      break;
    case 25:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] + (r[i] >> 1));
      break;
    case 26:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] + (g[i] >> 1));
      break;
    case 27:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(r[i] >> 1);
        b[i] = static_cast<uint8_t>(b[i] + t);
        g[i] = static_cast<uint8_t>(g[i] + t);
      }
      break;
    case 28:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] + (b[i] >> 1));
      break;
    case 29:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(g[i] >> 1);
        b[i] = static_cast<uint8_t>(b[i] + t);
        r[i] = static_cast<uint8_t>(r[i] + t);
      }
      break;
    case 30:
      for (std::size_t i = 0; i < n; ++i)
      {
        const uint8_t t = static_cast<uint8_t>(b[i] >> 1);
        g[i] = static_cast<uint8_t>(g[i] + t);
        r[i] = static_cast<uint8_t>(r[i] + t);
      }
      break;
    case 31:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] + (r[i] >> 1));
      break;
    default:
      break;
    }
  }

  std::int64_t estimate_sequence_cost(const std::vector<uint8_t> &seq)
  {
    if (seq.empty())
      return 0;
    std::int64_t cost = 0;
    uint8_t prev = seq[0];
    for (std::size_t i = 1; i < seq.size(); ++i)
    {
      cost += std::abs(static_cast<int>(seq[i]) - static_cast<int>(prev));
      prev = seq[i];
    }
    return cost;
  }

  int choose_filter(const std::vector<uint8_t> &src_b,
                    const std::vector<uint8_t> &src_g,
                    const std::vector<uint8_t> &src_r,
                    std::vector<uint8_t> &dst_b,
                    std::vector<uint8_t> &dst_g,
                    std::vector<uint8_t> &dst_r)
  {
    int best_code = 0;
    std::int64_t best_score = std::numeric_limits<std::int64_t>::max();
    std::vector<uint8_t> tb, tg, tr;
    tb.reserve(src_b.size());
    tg.reserve(src_g.size());
    tr.reserve(src_r.size());
    for (int code = 0; code < 32; ++code)
    {
      tb.assign(src_b.begin(), src_b.end());
      tg.assign(src_g.begin(), src_g.end());
      tr.assign(src_r.begin(), src_r.end());
      apply_color_filter(code, tb, tg, tr);
      const std::int64_t score = estimate_sequence_cost(tb) + estimate_sequence_cost(tg) + estimate_sequence_cost(tr);
      if (score < best_score)
      {
        best_score = score;
        best_code = code;
        dst_b = tb;
        dst_g = tg;
        dst_r = tr;
      }
    }
    return best_code;
  }

  template <typename T>
  void copy_block_from_plane(const tlg::v7::detail::image<T> &plane,
                             std::size_t x0,
                             std::size_t y0,
                             std::size_t bw,
                             std::size_t bh,
                             std::vector<T> &out)
  {
    out.resize(bw * bh);
    std::size_t idx = 0;
    for (std::size_t y = 0; y < bh; ++y)
    {
      const T *row = plane.row_ptr(y0 + y);
      for (std::size_t x = 0; x < bw; ++x)
      {
        out[idx++] = row[x0 + x];
      }
    }
  }

  template <typename T>
  void store_block_to_plane(tlg::v7::detail::image<T> &plane,
                            std::size_t x0,
                            std::size_t y0,
                            std::size_t bw,
                            std::size_t bh,
                            const std::vector<T> &values)
  {
    std::size_t idx = 0;
    for (std::size_t y = 0; y < bh; ++y)
    {
      T *row = plane.row_ptr(y0 + y);
      for (std::size_t x = 0; x < bw; ++x)
      {
        row[x0 + x] = values[idx++];
      }
    }
  }

} // namespace

namespace tlg::v7
{

namespace
{
  struct Header
  {
    uint8_t colors = 0;
    uint8_t reserved[3] = {0, 0, 0};
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t block_count = 0;
    uint32_t chunk_count = 0;
  };

  bool write_header(FILE *fp, const Header &hdr)
  {
    const unsigned char mark[11] = {'T', 'L', 'G', '7', '.', '0', 0, 'r', 'a', 'w', 0x1a};
    if (fwrite(mark, 1, sizeof(mark), fp) != sizeof(mark))
      return false;
    if (std::fputc(hdr.colors, fp) == EOF)
      return false;
    if (std::fputc(hdr.reserved[0], fp) == EOF)
      return false;
    if (std::fputc(hdr.reserved[1], fp) == EOF)
      return false;
    if (std::fputc(hdr.reserved[2], fp) == EOF)
      return false;
    write_u32le(fp, hdr.width);
    write_u32le(fp, hdr.height);
    write_u32le(fp, hdr.block_count);
    write_u32le(fp, hdr.chunk_count);
    return !std::ferror(fp);
  }

  bool read_header(FILE *fp, Header &hdr, std::string &err)
  {
    unsigned char colors = 0;
    unsigned char r1 = 0, r2 = 0, r3 = 0;
    int c0 = std::fgetc(fp);
    int c1 = std::fgetc(fp);
    int c2 = std::fgetc(fp);
    int c3 = std::fgetc(fp);
    if (c0 == EOF || c1 == EOF || c2 == EOF || c3 == EOF)
    {
      err = "tlg7: read header";
      return false;
    }
    colors = static_cast<unsigned char>(c0);
    r1 = static_cast<unsigned char>(c1);
    r2 = static_cast<unsigned char>(c2);
    r3 = static_cast<unsigned char>(c3);

    hdr.colors = colors;
    hdr.reserved[0] = r1;
    hdr.reserved[1] = r2;
    hdr.reserved[2] = r3;

    hdr.width = read_u32le(fp);
    hdr.height = read_u32le(fp);
    hdr.block_count = read_u32le(fp);
    hdr.chunk_count = read_u32le(fp);

    if (!(hdr.colors == 1 || hdr.colors == 3 || hdr.colors == 4))
    {
      err = "tlg7: unsupported color count";
      return false;
    }
    if (hdr.reserved[0] || hdr.reserved[1] || hdr.reserved[2])
    {
      err = "tlg7: reserved flags not zero";
      return false;
    }
    if (hdr.width == 0 || hdr.height == 0)
    {
      err = "tlg7: invalid dimensions";
      return false;
    }
    if (hdr.block_count == 0 || hdr.chunk_count == 0)
    {
      err = "tlg7: inconsistent block metadata";
      return false;
    }
    return true;
  }

  std::vector<detail::image<uint8_t>> extract_planes(const PixelBuffer &src, int colors)
  {
    const std::size_t width = src.width;
    const std::size_t height = src.height;
    if (colors == 1)
    {
      detail::image<uint8_t> gray(width, height);
      for (std::size_t y = 0; y < height; ++y)
      {
        const std::size_t row_idx = y * width;
        for (std::size_t x = 0; x < width; ++x)
        {
          const std::size_t idx = row_idx + x;
          uint8_t r = 0;
          uint8_t g = 0;
          uint8_t b = 0;
          if (src.channels == 4)
          {
            r = src.data[idx * 4 + 1];
            g = src.data[idx * 4 + 2];
            b = src.data[idx * 4 + 3];
          }
          else
          {
            r = src.data[idx * 3 + 0];
            g = src.data[idx * 3 + 1];
            b = src.data[idx * 3 + 2];
          }
          const uint16_t gray_val = static_cast<uint16_t>(r) * 299u + static_cast<uint16_t>(g) * 587u + static_cast<uint16_t>(b) * 114u;
          gray.row_ptr(y)[x] = static_cast<uint8_t>(gray_val / 1000u);
        }
      }
      return detail::split_components_from_gray(gray);
    }

    detail::image<uint32_t> packed(width, height);
    for (std::size_t y = 0; y < height; ++y)
    {
      const std::size_t row_idx = y * width;
      uint32_t *dst = packed.row_ptr(y);
      for (std::size_t x = 0; x < width; ++x)
      {
        const std::size_t idx = row_idx + x;
        uint8_t a = 255;
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        if (src.channels == 4)
        {
          a = src.data[idx * 4 + 0];
          r = src.data[idx * 4 + 1];
          g = src.data[idx * 4 + 2];
          b = src.data[idx * 4 + 3];
        }
        else
        {
          r = src.data[idx * 3 + 0];
          g = src.data[idx * 3 + 1];
          b = src.data[idx * 3 + 2];
        }
        dst[x] = static_cast<uint32_t>(b) | (static_cast<uint32_t>(g) << 8) |
                 (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(a) << 24);
      }
    }

    const std::size_t component_count = (colors == 4) ? 4u : 3u;
    auto planes = detail::split_components_from_packed(packed, component_count);

    if (colors == 4 && planes.size() < 4)
    {
      planes.resize(4, detail::image<uint8_t>(width, height, 255));
    }
    if (colors == 3 && planes.size() < 3)
    {
      planes.resize(3, detail::image<uint8_t>(width, height, 0));
    }
    return planes;
  }

  PixelBuffer assemble_pixelbuffer(const std::vector<detail::image<uint8_t>> &planes,
                                    uint32_t width,
                                    uint32_t height,
                                    int colors)
  {
    PixelBuffer out;
    out.width = width;
    out.height = height;
    if (colors == 4)
      out.channels = 4;
    else if (colors == 3)
      out.channels = 3;
    else
      out.channels = 3;

    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    out.data.resize(pixels * out.channels);

    for (std::size_t y = 0; y < height; ++y)
    {
      for (std::size_t x = 0; x < width; ++x)
      {
        const std::size_t idx = y * width + x;
        const uint8_t b = planes.empty() ? 0 : planes[0].row_ptr(y)[x];
        const uint8_t g = planes.size() > 1 ? planes[1].row_ptr(y)[x] : b;
        const uint8_t r = planes.size() > 2 ? planes[2].row_ptr(y)[x] : g;
        const uint8_t a = planes.size() > 3 ? planes[3].row_ptr(y)[x] : 255;
        if (out.channels == 4)
        {
          out.data[idx * 4 + 0] = a;
          out.data[idx * 4 + 1] = r;
          out.data[idx * 4 + 2] = g;
          out.data[idx * 4 + 3] = b;
        }
        else
        {
          out.data[idx * 3 + 0] = r;
          out.data[idx * 3 + 1] = g;
          out.data[idx * 3 + 2] = b;
        }
      }
    }
    return out;
  }

  struct BlockContext
  {
    std::size_t x0 = 0;
    std::size_t y0 = 0;
    std::size_t bw = 0;
    std::size_t bh = 0;
    std::size_t index = 0;
  };

  BlockContext make_block_context(std::size_t block_x,
                                  std::size_t block_y,
                                  std::size_t width,
                                  std::size_t height,
                                  std::size_t blocks_x)
  {
    BlockContext ctx;
    ctx.x0 = block_x * BLOCK_SIZE;
    ctx.y0 = block_y * BLOCK_SIZE;
    ctx.bw = std::min<std::size_t>(BLOCK_SIZE, width - ctx.x0);
    ctx.bh = std::min<std::size_t>(BLOCK_SIZE, height - ctx.y0);
    ctx.index = block_y * blocks_x + block_x;
    return ctx;
  }

} // namespace

bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err)
{
  err.clear();
  Header hdr{};
  if (!read_header(fp, hdr, err))
    return false;

  const std::size_t width = hdr.width;
  const std::size_t height = hdr.height;
  const std::size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const std::size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (static_cast<uint64_t>(blocks_x) * static_cast<uint64_t>(blocks_y) != hdr.block_count)
  {
    err = "tlg7: block count mismatch";
    return false;
  }

  std::vector<uint8_t> filter_indices(hdr.block_count);
  if (!filter_indices.empty())
  {
    if (!read_exact(fp, filter_indices.data(), filter_indices.size()))
    {
      err = "tlg7: read filter indices";
      return false;
    }
  }

  const std::size_t component_count = (hdr.colors == 4) ? 4u : (hdr.colors == 3 ? 3u : 1u);

  std::vector<detail::image<uint8_t>> filtered_planes;
  filtered_planes.reserve(component_count);
  std::vector<detail::image<uint8_t>> output_planes;
  output_planes.reserve(component_count);
  for (std::size_t i = 0; i < component_count; ++i)
  {
    filtered_planes.emplace_back(width, height, 0);
    output_planes.emplace_back(width, height, 0);
  }

  CAS8::Config cas_cfg;
  cas_cfg.enablePlanarLite = true;
  CAS8 cas(cas_cfg, 0, 255);

  const std::size_t chunk_rows = (height + CHUNK_SCAN_LINES - 1) / CHUNK_SCAN_LINES;
  std::vector<int16_t> block_buffer;
  block_buffer.reserve(BLOCK_SIZE * BLOCK_SIZE);

  for (std::size_t chunk_y = 0; chunk_y < chunk_rows; ++chunk_y)
  {
    const std::size_t chunk_y0 = chunk_y * CHUNK_SCAN_LINES;
    const std::size_t chunk_height = std::min<std::size_t>(CHUNK_SCAN_LINES, height - chunk_y0);
    const std::size_t chunk_pixels = chunk_height * width;

    std::vector<std::vector<int16_t>> chunk_residuals(component_count);

    for (std::size_t c = 0; c < component_count; ++c)
    {
      const uint32_t bit_length = read_u32le(fp);
      const std::size_t byte_count = (bit_length + 7u) / 8u;
      std::vector<uint8_t> buf(byte_count);
      if (byte_count && !read_exact(fp, buf.data(), byte_count))
      {
        err = "tlg7: read residual stream";
        return false;
      }
      if (!decode_residuals_golomb(buf.data(), buf.size(), chunk_pixels, chunk_residuals[c]))
      {
        err = "tlg7: residual decode error";
        return false;
      }
    }

    std::vector<CAS8::State> states(component_count);
    std::vector<std::size_t> residual_cursor(component_count, 0);

    const std::size_t chunk_block_row_start = chunk_y0 / BLOCK_SIZE;
    const std::size_t chunk_block_row_end = std::min<std::size_t>((chunk_y0 + chunk_height + BLOCK_SIZE - 1) / BLOCK_SIZE, blocks_y);

    for (std::size_t by = chunk_block_row_start; by < chunk_block_row_end; ++by)
    {
      const std::size_t y0 = by * BLOCK_SIZE;
      if (y0 >= height)
        break;
      for (std::size_t bx = 0; bx < blocks_x; ++bx)
      {
        const BlockContext ctx = make_block_context(bx, by, width, height, blocks_x);
        const std::size_t pixel_count = ctx.bw * ctx.bh;
        const bool is_full_block = (ctx.bw == BLOCK_SIZE && ctx.bh == BLOCK_SIZE);

        std::vector<std::vector<uint8_t>> filtered_block(component_count);
        for (std::size_t c = 0; c < component_count; ++c)
        {
          block_buffer.assign(chunk_residuals[c].begin() + residual_cursor[c],
                               chunk_residuals[c].begin() + residual_cursor[c] + pixel_count);
          residual_cursor[c] += pixel_count;

          if (is_full_block)
            reorder_from_hilbert(block_buffer);

          filtered_block[c].resize(pixel_count);

          std::size_t idx = 0;
          for (std::size_t y = 0; y < ctx.bh; ++y)
          {
            for (std::size_t x = 0; x < ctx.bw; ++x)
            {
              const int gx = static_cast<int>(ctx.x0 + x);
              const int gy = static_cast<int>(ctx.y0 + y);

              const int a = sample_pixel(filtered_planes[c], gx - 1, gy);
              const int b = sample_pixel(filtered_planes[c], gx, gy - 1);
              const int cdiag = sample_pixel(filtered_planes[c], gx - 1, gy - 1);
              const int d = sample_pixel(filtered_planes[c], gx + 1, gy - 1);
              const int f = sample_pixel(filtered_planes[c], gx, gy - 2);

              auto [pred, pid] = cas.predict_and_choose<uint8_t>(a, b, cdiag, d, f, states[c]);
              int recon = pred + block_buffer[idx];
              if (recon < 0)
                recon = 0;
              else if (recon > 255)
                recon = 255;

              filtered_planes[c].row_ptr(ctx.y0 + y)[ctx.x0 + x] = static_cast<uint8_t>(recon);
              filtered_block[c][idx] = static_cast<uint8_t>(recon);

              cas.update_state(states[c], pid, std::abs(recon - pred));
              ++idx;
            }
          }
        }

        if (component_count >= 3)
        {
          std::vector<uint8_t> block_b = filtered_block[0];
          std::vector<uint8_t> block_g = filtered_block[1];
          std::vector<uint8_t> block_r = filtered_block[2];
          undo_color_filter(filter_indices[ctx.index], block_b, block_g, block_r);
          store_block_to_plane(output_planes[0], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_b);
          store_block_to_plane(output_planes[1], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_g);
          store_block_to_plane(output_planes[2], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_r);
        }
        else
        {
          store_block_to_plane(output_planes[0], ctx.x0, ctx.y0, ctx.bw, ctx.bh, filtered_block[0]);
        }

        if (component_count == 4)
        {
          store_block_to_plane(output_planes[3], ctx.x0, ctx.y0, ctx.bw, ctx.bh, filtered_block[3]);
        }
      }
    }

    for (std::size_t c = 0; c < component_count; ++c)
    {
      if (residual_cursor[c] != chunk_pixels)
      {
        err = "tlg7: residual cursor mismatch";
        return false;
      }
    }
  }

  out = assemble_pixelbuffer(output_planes, hdr.width, hdr.height, hdr.colors == 1 ? 3 : hdr.colors);
  if (hdr.colors == 1)
  {
    // grayscale to RGB replicate
    const std::size_t pixels = static_cast<std::size_t>(hdr.width) * hdr.height;
    for (std::size_t i = 0; i < pixels; ++i)
    {
      const uint8_t g = out.data[i * 3 + 0];
      out.data[i * 3 + 1] = g;
      out.data[i * 3 + 2] = g;
    }
  }
  return true;
}

namespace enc
{

bool write_raw(FILE *fp, const PixelBuffer &src, int colors, std::string &err)
{
  err.clear();
  if (!(colors == 1 || colors == 3 || colors == 4))
  {
    err = "tlg7: unsupported color count";
    return false;
  }
  if (!(src.channels == 3 || src.channels == 4))
  {
    err = "tlg7: unsupported source format";
    return false;
  }
  if (src.width == 0 || src.height == 0)
  {
    err = "tlg7: empty image";
    return false;
  }

  const std::size_t width = src.width;
  const std::size_t height = src.height;
  const std::size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const std::size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const std::size_t block_count = blocks_x * blocks_y;
  const std::size_t chunk_rows = (height + CHUNK_SCAN_LINES - 1) / CHUNK_SCAN_LINES;

  if (block_count == 0 || chunk_rows == 0)
  {
    err = "tlg7: invalid block geometry";
    return false;
  }
  if (block_count > std::numeric_limits<uint32_t>::max() || chunk_rows > std::numeric_limits<uint32_t>::max())
  {
    err = "tlg7: image too large";
    return false;
  }

  Header hdr;
  hdr.colors = static_cast<uint8_t>(colors);
  hdr.width = src.width;
  hdr.height = src.height;
  hdr.block_count = static_cast<uint32_t>(block_count);
  hdr.chunk_count = static_cast<uint32_t>(chunk_rows);

  if (!write_header(fp, hdr))
  {
    err = "tlg7: write header";
    return false;
  }

  const std::size_t component_count = (colors == 4) ? 4u : (colors == 3 ? 3u : 1u);

  std::vector<uint8_t> filter_indices(block_count, 0);
  if (!filter_indices.empty())
  {
    std::vector<uint8_t> zero(block_count, 0);
    if (!zero.empty() && fwrite(zero.data(), 1, zero.size(), fp) != zero.size())
    {
      err = "tlg7: write filter placeholder";
      return false;
    }
  }
  const long filter_pos = std::ftell(fp) - static_cast<long>(filter_indices.size());

  auto planes = extract_planes(src, colors);
  std::vector<detail::image<uint8_t>> filtered_planes;
  filtered_planes.reserve(component_count);
  for (std::size_t i = 0; i < component_count; ++i)
    filtered_planes.emplace_back(width, height, 0);

  CAS8::Config cas_cfg;
  cas_cfg.enablePlanarLite = true;
  CAS8 cas(cas_cfg, 0, 255);

  std::vector<int16_t> residual_block;
  residual_block.reserve(BLOCK_SIZE * BLOCK_SIZE);

  for (std::size_t chunk_y = 0; chunk_y < chunk_rows; ++chunk_y)
  {
    const std::size_t chunk_y0 = chunk_y * CHUNK_SCAN_LINES;
    const std::size_t chunk_height = std::min<std::size_t>(CHUNK_SCAN_LINES, height - chunk_y0);
    const std::size_t chunk_pixels = chunk_height * width;

    std::vector<std::vector<int16_t>> chunk_residuals(component_count);
    for (auto &vec : chunk_residuals)
      vec.reserve(chunk_pixels);

    std::vector<CAS8::State> states(component_count);

    const std::size_t chunk_block_row_start = chunk_y0 / BLOCK_SIZE;
    const std::size_t chunk_block_row_end = std::min<std::size_t>((chunk_y0 + chunk_height + BLOCK_SIZE - 1) / BLOCK_SIZE, blocks_y);

    for (std::size_t by = chunk_block_row_start; by < chunk_block_row_end; ++by)
    {
      const std::size_t y0 = by * BLOCK_SIZE;
      if (y0 >= height)
        break;
      for (std::size_t bx = 0; bx < blocks_x; ++bx)
      {
        const BlockContext ctx = make_block_context(bx, by, width, height, blocks_x);
        const std::size_t pixel_count = ctx.bw * ctx.bh;
        const bool is_full_block = (ctx.bw == BLOCK_SIZE && ctx.bh == BLOCK_SIZE);

        std::vector<std::vector<uint8_t>> filtered_block(component_count, std::vector<uint8_t>(pixel_count));

        if (component_count >= 3)
        {
          std::vector<uint8_t> block_b;
          std::vector<uint8_t> block_g;
          std::vector<uint8_t> block_r;
          copy_block_from_plane(planes[0], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_b);
          copy_block_from_plane(planes[1], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_g);
          copy_block_from_plane(planes[2], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_r);

          std::vector<uint8_t> filtered_b;
          std::vector<uint8_t> filtered_g;
          std::vector<uint8_t> filtered_r;
          const int code = choose_filter(block_b, block_g, block_r, filtered_b, filtered_g, filtered_r);
          filter_indices[ctx.index] = static_cast<uint8_t>(code);

          filtered_block[0] = std::move(filtered_b);
          filtered_block[1] = std::move(filtered_g);
          filtered_block[2] = std::move(filtered_r);
        }
        else
        {
          copy_block_from_plane(planes[0], ctx.x0, ctx.y0, ctx.bw, ctx.bh, filtered_block[0]);
        }

        if (component_count == 4)
        {
          copy_block_from_plane(planes[3], ctx.x0, ctx.y0, ctx.bw, ctx.bh, filtered_block[3]);
        }

        for (std::size_t c = 0; c < component_count; ++c)
        {
          residual_block.clear();
          residual_block.reserve(pixel_count);
          std::size_t idx = 0;
          for (std::size_t y = 0; y < ctx.bh; ++y)
          {
            for (std::size_t x = 0; x < ctx.bw; ++x)
            {
              const int gx = static_cast<int>(ctx.x0 + x);
              const int gy = static_cast<int>(ctx.y0 + y);
              const uint8_t value = filtered_block[c][idx];

              const int a = sample_pixel(filtered_planes[c], gx - 1, gy);
              const int b = sample_pixel(filtered_planes[c], gx, gy - 1);
              const int cdiag = sample_pixel(filtered_planes[c], gx - 1, gy - 1);
              const int d = sample_pixel(filtered_planes[c], gx + 1, gy - 1);
              const int f = sample_pixel(filtered_planes[c], gx, gy - 2);

              auto [pred, pid] = cas.predict_and_choose<uint8_t>(a, b, cdiag, d, f, states[c]);
              const int residual = static_cast<int>(value) - pred;
              residual_block.push_back(static_cast<int16_t>(residual));
              cas.update_state(states[c], pid, std::abs(residual));

              filtered_planes[c].row_ptr(ctx.y0 + y)[ctx.x0 + x] = value;
              ++idx;
            }
          }

          if (is_full_block)
            reorder_to_hilbert(residual_block);

          chunk_residuals[c].insert(chunk_residuals[c].end(), residual_block.begin(), residual_block.end());
        }
      }
    }

    for (std::size_t c = 0; c < component_count; ++c)
    {
      if (chunk_residuals[c].size() != chunk_pixels)
      {
        err = "tlg7: residual size mismatch";
        return false;
      }
    }

    for (std::size_t c = 0; c < component_count; ++c)
    {
      std::vector<uint8_t> encoded;
      GolombBitStream bs(encoded);
      compress_residuals_golomb(bs, chunk_residuals[c]);
      const uint32_t bit_length = static_cast<uint32_t>(bs.GetBitLength());
      bs.Flush();
      write_u32le(fp, bit_length);
      const std::size_t byte_count = (bit_length + 7u) / 8u;
      if (encoded.size() != byte_count)
        encoded.resize(byte_count);
      if (byte_count && fwrite(encoded.data(), 1, byte_count, fp) != byte_count)
      {
        err = "tlg7: write residual data";
        return false;
      }
    }
  }

  if (!filter_indices.empty())
  {
    if (std::fseek(fp, filter_pos, SEEK_SET) != 0)
    {
      err = "tlg7: seek filter table";
      return false;
    }
    if (!filter_indices.empty() && fwrite(filter_indices.data(), 1, filter_indices.size(), fp) != filter_indices.size())
    {
      err = "tlg7: write filter table";
      return false;
    }
    if (std::fseek(fp, 0, SEEK_END) != 0)
    {
      err = "tlg7: seek end";
      return false;
    }
  }

  return true;
}

} // namespace enc

} // namespace tlg::v7
