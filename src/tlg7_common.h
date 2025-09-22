#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>
#include <string>

#include "tlg7_io.h"

namespace tlg::v7
{

  inline constexpr std::size_t BLOCK_SIZE = 8;
  inline constexpr std::size_t CHUNK_SCAN_LINES = 64;

  inline constexpr std::array<uint8_t, 64> HILBERT8x8 = {

      /* 0*/ 8 * 0 + 0,
      /* 1*/ 8 * 1 + 0,
      /* 2*/ 8 * 1 + 1,
      /* 3*/ 8 * 0 + 1,
      /* 4*/ 8 * 0 + 2,
      /* 5*/ 8 * 0 + 3,
      /* 6*/ 8 * 1 + 3,
      /* 7*/ 8 * 1 + 2,
      /* 8*/ 8 * 2 + 2,
      /* 9*/ 8 * 2 + 3,
      /*10*/ 8 * 3 + 3,
      /*11*/ 8 * 3 + 2,
      /*12*/ 8 * 3 + 1,
      /*13*/ 8 * 2 + 1,
      /*14*/ 8 * 2 + 0,
      /*15*/ 8 * 3 + 0,
      /*16*/ 8 * 4 + 0,
      /*17*/ 8 * 4 + 1,
      /*18*/ 8 * 5 + 1,
      /*19*/ 8 * 5 + 0,
      /*20*/ 8 * 6 + 0,
      /*21*/ 8 * 7 + 0,
      /*22*/ 8 * 7 + 1,
      /*23*/ 8 * 6 + 1,
      /*24*/ 8 * 6 + 2,
      /*25*/ 8 * 7 + 2,
      /*26*/ 8 * 7 + 3,
      /*27*/ 8 * 6 + 3,
      /*28*/ 8 * 5 + 3,
      /*29*/ 8 * 5 + 2,
      /*30*/ 8 * 4 + 2,
      /*31*/ 8 * 4 + 3,
      /*32*/ 8 * 4 + 4,
      /*33*/ 8 * 4 + 5,
      /*34*/ 8 * 5 + 5,
      /*35*/ 8 * 5 + 4,
      /*36*/ 8 * 6 + 4,
      /*37*/ 8 * 7 + 4,
      /*38*/ 8 * 7 + 5,
      /*39*/ 8 * 6 + 5,
      /*40*/ 8 * 6 + 6,
      /*41*/ 8 * 7 + 6,
      /*42*/ 8 * 7 + 7,
      /*43*/ 8 * 6 + 7,
      /*44*/ 8 * 5 + 7,
      /*45*/ 8 * 5 + 6,
      /*46*/ 8 * 4 + 6,
      /*47*/ 8 * 4 + 7,
      /*48*/ 8 * 3 + 7,
      /*49*/ 8 * 2 + 7,
      /*50*/ 8 * 2 + 6,
      /*51*/ 8 * 3 + 6,
      /*52*/ 8 * 3 + 5,
      /*53*/ 8 * 3 + 4,
      /*54*/ 8 * 2 + 4,
      /*55*/ 8 * 2 + 5,
      /*56*/ 8 * 1 + 5,
      /*57*/ 8 * 1 + 4,
      /*58*/ 8 * 0 + 4,
      /*59*/ 8 * 0 + 5,
      /*60*/ 8 * 0 + 6,
      /*61*/ 8 * 1 + 6,
      /*62*/ 8 * 1 + 7,
      /*63*/ 8 * 0 + 7,

  };

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

#ifdef TLG7_USE_MED_PREDICTOR

  // MED predictor mirrors the TLG6 "// MED method" implementation for single-channel data.
  class MedPredictor
  {
  public:
    using PredId = uint8_t;
    struct State
    {
    };

    template <typename T>
    std::pair<int, PredId> predict_and_choose(int a, int b, int c, int d, int f, const State &) const
    {
      return {predict_only<T>(static_cast<PredId>(0), a, b, c, d, f), static_cast<PredId>(0)};
    }

    template <typename T>
    int predict_only(PredId, int a, int b, int c, int, int) const
    {
      return med_predict<T>(a, b, c);
    }

    void update_state(State &, PredId, int) const {}
    void update_state(State &, PredId, int, int, int) const {}

  private:
    template <typename T>
    static int med_predict(int a, int b, int c)
    {
      const int max_ab = std::max(a, b);
      const int min_ab = std::min(a, b);
      int pred;
      if (c >= max_ab)
        pred = min_ab;
      else if (c <= min_ab)
        pred = max_ab;
      else
        pred = a + b - c;
      const int lo = static_cast<int>(std::numeric_limits<T>::min());
      const int hi = static_cast<int>(std::numeric_limits<T>::max());
      if (pred < lo)
        pred = lo;
      else if (pred > hi)
        pred = hi;
      return pred;
    }
  };

  using ActivePredictor = MedPredictor;

#else
  using ActivePredictor = CAS8;
#endif

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
