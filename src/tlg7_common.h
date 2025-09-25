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

  inline constexpr int COLOR_FILTER_PERMUTATIONS = 6;
  inline constexpr int COLOR_FILTER_PRIMARY_PREDICTORS = 4;
  inline constexpr int COLOR_FILTER_SECONDARY_PREDICTORS = 4;
  inline constexpr int COLOR_FILTER_CODE_COUNT = COLOR_FILTER_PERMUTATIONS * COLOR_FILTER_PRIMARY_PREDICTORS *
                                                COLOR_FILTER_SECONDARY_PREDICTORS;

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

  inline constexpr int MWR_DEFAULT_T2 = 100; // 7;
  inline constexpr int MWR_DEFAULT_Td = 100; // 6;
  inline constexpr int MWR_DEFAULT_Tc = 3;   // 3;

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

  class MWR
  {
  public:
    enum class PredId : uint8_t
    {
      MED = 0,
      LEFT,
      TOP,
      DIAG_NE,
      DIAG_NW
    };

    struct Config
    {
      int T2 = MWR_DEFAULT_T2;
      int Td = MWR_DEFAULT_Td;
      int Tc = MWR_DEFAULT_Tc;
    };

    struct State
    {
    };

    MWR() : cfg_{}, lo_(0), hi_(255) {}
    MWR(Config cfg, int lo, int hi) : cfg_(cfg), lo_(lo), hi_(hi) {}

    template <typename T>
    std::pair<int, PredId> predict_and_choose(int a, int b, int c, int d, int f, const State &) const
    {
      const PredId pid = select_(a, b, c, d, f);
      return {predict_<T>(pid, a, b, c, d, f), pid};
    }

    template <typename T>
    int predict_only(PredId pid, int a, int b, int c, int d, int f) const
    {
      return predict_<T>(pid, a, b, c, d, f);
    }

    void update_state(State &, PredId, int) const {}
    void update_state(State &, PredId, int, int, int) const {}

  private:
    static inline int abs_diff_(int x, int y)
    {
      return (x >= y) ? (x - y) : (y - x);
    }

    PredId select_(int a, int b, int c, int d, int f) const
    {
      const int Dh = abs_diff_(a, b);
      const int Dv = abs_diff_(b, c);
      const int Dc = abs_diff_(a, c);

      if ((Dh - Dv) >= cfg_.T2)
      {
        if (abs_diff_(d, b) <= cfg_.Tc)
          return PredId::LEFT;
      }

      if ((Dv - Dh) >= cfg_.T2)
      {
        if (abs_diff_(f, b) <= cfg_.Tc)
          return PredId::TOP;
      }

      if (Dc >= cfg_.Td)
      {
        if (abs_diff_(d, b) <= cfg_.Tc)
          return PredId::DIAG_NE;
        if (abs_diff_(f, a) <= cfg_.Tc)
          return PredId::DIAG_NW;
      }

      return PredId::MED;
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
    int predict_(PredId pid, int a, int b, int c, int /*d*/, int /*f*/) const
    {
      switch (pid)
      {
      case PredId::LEFT:
        return clip_<T>(a);
      case PredId::TOP:
        return clip_<T>(b);
      case PredId::DIAG_NE:
        return clip_<T>(b + (a - c));
      case PredId::DIAG_NW:
        return clip_<T>(a + (b - c));
      case PredId::MED:
      default:
        return med_predict_<T>(a, b, c);
      }
    }

    template <typename T>
    int med_predict_(int a, int b, int c) const
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
      return clip_<T>(pred);
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
  using ActivePredictor = MWR;
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
