#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <string>

#include "tlg7_io.h"

namespace tlg::v7
{

  inline constexpr std::size_t BLOCK_SIZE = 8;
  inline constexpr std::size_t CHUNK_SCAN_LINES = 64;

  inline constexpr std::array<uint8_t, 64> HILBERT8x8 = {
      0, 1, 8, 9, 16, 17, 24, 25,
      2, 3, 10, 11, 18, 19, 26, 27,
      4, 5, 12, 13, 20, 21, 28, 29,
      6, 7, 14, 15, 22, 23, 30, 31,
      48, 49, 56, 57, 58, 59, 50, 51,
      40, 41, 32, 33, 34, 35, 42, 43,
      38, 39, 36, 37, 44, 45, 46, 47,
      52, 53, 54, 55, 62, 63, 60, 61};

  inline constexpr std::array<uint8_t, 64> HILBERT8x8_INV = []
  {
    std::array<uint8_t, 64> inv{};
    for (uint8_t i = 0; i < 64; ++i)
      inv[HILBERT8x8[i]] = i;
    return inv;
  }();

  inline constexpr int CAS_DEFAULT_T1 = 3;
  inline constexpr int CAS_DEFAULT_T2 = 7;
  inline constexpr int CAS_DEFAULT_ERR_DECAY = 4;
  inline constexpr int CAS_DEFAULT_ZERO_BIAS_DELTA = 3;
  inline constexpr int CAS_ADAPT_HIGH_ACTIVITY = 128;

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
                                  std::size_t blocks_x);

  int sample_pixel(const detail::image<uint8_t> &img, int x, int y);

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
      int T1 = CAS_DEFAULT_T1;
      int T2 = CAS_DEFAULT_T2;
      int errDecayShift = CAS_DEFAULT_ERR_DECAY;
      bool enablePlanarLiteFlat = false;
      bool enablePlanarLiteDiag = true;
      int zeroBiasDelta = CAS_DEFAULT_ZERO_BIAS_DELTA;
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

    CAS8(Config cfg, int lo, int hi) : cfg_(cfg), lo_(lo), hi_(hi) {}

    template <typename T>
    std::pair<int, PredId> predict_and_choose(int a, int b, int c, int d, int f, const State &st) const
    {
      const Class cl = classify_(a, b, c);
      const PredId pid = select_(st, cl, a, b);
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

    void update_state(State &st, PredId pid, int abs_e, int Dh, int Dv) const
    {
      const int high = std::max(Dh, Dv);
      const int shift = (high >= CAS_ADAPT_HIGH_ACTIVITY) ? std::max(1, cfg_.errDecayShift - 1) : cfg_.errDecayShift;
      st.update(pid, abs_e, shift);
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

    PredId select_(const State &st, Class cl, int a, int b) const
    {
      switch (cl)
      {
      case Class::FLAT:
      {
        if (iabs_(a - b) <= 1)
          return P2;
        const bool allow_planar = cfg_.enablePlanarLiteFlat;
        const PredId secondary = (allow_planar && st.errScore[Pp] < st.errScore[P3]) ? Pp : P3;
        return tie_break_zero_friendly_(st, P2, secondary);
      }
      case Class::VERT:
      {
        if (iabs_(a - b) <= 1)
          return P0;
        return tie_break_zero_friendly_(st, P0, P6);
      }
      case Class::HORZ:
      {
        if (iabs_(a - b) <= 1)
          return P1;
        return tie_break_zero_friendly_(st, P1, P7);
      }
      case Class::DIAG:
      default:
      {
        const PredId tilt = (st.errScore[P4] <= st.errScore[P5]) ? P4 : P5;
        PredId primary = P3;
        if (cfg_.enablePlanarLiteDiag && st.errScore[Pp] < st.errScore[primary])
          primary = Pp;
        return tie_break_zero_friendly_(st, primary, tilt);
      }
      }
    }

    PredId tie_break_zero_friendly_(const State &st, PredId first, PredId second) const
    {
      const uint16_t ef = st.errScore[first];
      const uint16_t es = st.errScore[second];
      if (ef == es)
      {
        if (is_zero_friendly_(first))
          return first;
        if (is_zero_friendly_(second))
          return second;
        return first;
      }
      const int diff = static_cast<int>(ef) - static_cast<int>(es);
      if (iabs_(diff) <= cfg_.zeroBiasDelta)
      {
        const bool first_zero = is_zero_friendly_(first);
        const bool second_zero = is_zero_friendly_(second);
        if (first_zero != second_zero)
          return first_zero ? first : second;
      }
      return (ef < es) ? first : second;
    }

    static inline bool is_zero_friendly_(PredId pid)
    {
      return pid == P2 || pid == P0 || pid == P1;
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
        if (!(cfg_.enablePlanarLiteFlat || cfg_.enablePlanarLiteDiag))
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

  void apply_color_filter(int code,
                          std::vector<int16_t> &b,
                          std::vector<int16_t> &g,
                          std::vector<int16_t> &r);

  void undo_color_filter(int code,
                         std::vector<int16_t> &b,
                         std::vector<int16_t> &g,
                         std::vector<int16_t> &r);

  template <typename T>
  void copy_block_from_plane(const detail::image<T> &plane,
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
        out[idx++] = row[x0 + x];
    }
  }

  template <typename T>
  void store_block_to_plane(detail::image<T> &plane,
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
        row[x0 + x] = values[idx++];
    }
  }

  std::vector<detail::image<uint8_t>> extract_planes(const PixelBuffer &src, int colors);

  PixelBuffer assemble_pixelbuffer(const std::vector<detail::image<uint8_t>> &planes,
                                   uint32_t width,
                                   uint32_t height,
                                   int colors);

  namespace detail
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

    bool write_header(FILE *fp, const Header &hdr);
    bool read_header(FILE *fp, Header &hdr, std::string &err);

  } // namespace detail

} // namespace tlg::v7
