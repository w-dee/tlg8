#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tlg::v8::enc
{
  // CAS-8 予測器本体。提示された参照実装に合わせつつコメントを日本語化した。
  class CAS8
  {
  public:
    enum class Class : uint8_t
    {
      Flat = 0,
      Horizontal = 1,
      Vertical = 2,
      Diagonal = 3,
    };

    enum class PredId : uint8_t
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
      PredCount,
    };

    struct Config
    {
      int T1;
      int T2;
      bool enable_planar_lite;
      int err_decay_shift;

      constexpr Config(int t1 = 1, int t2 = 2, bool enable = true, int decay = 3)
          : T1(t1), T2(t2), enable_planar_lite(enable), err_decay_shift(decay)
      {
      }
    };

    struct State
    {
      static constexpr size_t N = static_cast<size_t>(PredId::PredCount);
      std::array<uint16_t, N> err_score{};

      inline void update(PredId pid, int abs_e, int err_decay_shift)
      {
        const size_t index = static_cast<size_t>(pid);
        const uint16_t prev = err_score[index];
        const uint16_t decayed = static_cast<uint16_t>(prev - (prev >> err_decay_shift));
        err_score[index] = static_cast<uint16_t>(decayed + static_cast<uint16_t>(std::min(abs_e, 0xFFFF)));
      }
    };

    CAS8(Config cfg = Config{}, int lo = 0, int hi = 255) : cfg_(cfg), lo_(lo), hi_(hi) {}

    template <typename T>
    inline std::pair<int, PredId> predict_and_choose(int a,
                                                     int b,
                                                     int c,
                                                     int d,
                                                     int f,
                                                     const State &st) const
    {
      const Class klass = classify(a, b, c);
      const PredId pid = select(klass, st);
      const int pred = dispatch_predict<T>(pid, a, b, c, d, f);
      return {pred, pid};
    }

    inline PredId choose_only(int a, int b, int c, const State &st) const
    {
      const Class klass = classify(a, b, c);
      return select(klass, st);
    }

    template <typename T>
    inline int predict_only(PredId pid, int a, int b, int c, int d, int f) const
    {
      return dispatch_predict<T>(pid, a, b, c, d, f);
    }

    inline std::pair<int, PredId> predict8_and_choose(int a,
                                                      int b,
                                                      int c,
                                                      int d,
                                                      int f,
                                                      const State &st) const
    {
      return predict_and_choose<uint8_t>(a, b, c, d, f, st);
    }

    inline int predict8_only(PredId pid, int a, int b, int c, int d, int f) const
    {
      return predict_only<uint8_t>(pid, a, b, c, d, f);
    }

    inline void update_state(State &st, PredId pid, int abs_e) const
    {
      st.update(pid, abs_e, cfg_.err_decay_shift);
    }

    inline Class classify_for_debug(int a, int b, int c) const { return classify(a, b, c); }

    const Config &config() const { return cfg_; }
    int lo() const { return lo_; }
    int hi() const { return hi_; }

  private:
    static inline int iabs(int v) { return v < 0 ? -v : v; }

    template <typename T>
    inline int clip(int v) const
    {
      if (v < lo_)
        v = lo_;
      if (v > hi_)
        v = hi_;
      return static_cast<int>(static_cast<T>(v));
    }

    inline Class classify(int a, int b, int c) const
    {
      const int dh = iabs(a - b);
      const int dv = iabs(b - c);
      if (dh <= cfg_.T1 && dv <= cfg_.T1)
        return Class::Flat;
      if ((dh - dv) >= cfg_.T2)
        return Class::Vertical;
      if ((dv - dh) >= cfg_.T2)
        return Class::Horizontal;
      return Class::Diagonal;
    }

    template <typename T, PredId PID>
    inline int predict_core(int a, int b, int c, int d, int f) const
    {
      if constexpr (PID == PredId::P0)
        return clip<T>(a);
      else if constexpr (PID == PredId::P1)
        return clip<T>(b);
      else if constexpr (PID == PredId::P2)
        return clip<T>((a + b + 1) >> 1);
      else if constexpr (PID == PredId::P3)
        return clip<T>(a + b - c);
      else if constexpr (PID == PredId::P4)
        return clip<T>(a + ((b - c) >> 1));
      else if constexpr (PID == PredId::P5)
        return clip<T>(b + ((a - c) >> 1));
      else if constexpr (PID == PredId::P6)
        return clip<T>(((a << 1) + b - c + 2) >> 2);
      else if constexpr (PID == PredId::P7)
        return clip<T>((a + (b << 1) - c + 2) >> 2);
      else
      {
        const int hx = (a - c) + (b - d);
        const int vy = (b - c) + (f - b);
        return clip<T>(a + ((hx + vy + 2) >> 2));
      }
    }

    template <typename T>
    inline int dispatch_predict(PredId pid, int a, int b, int c, int d, int f) const
    {
      switch (pid)
      {
      case PredId::P0:
        return predict_core<T, PredId::P0>(a, b, c, d, f);
      case PredId::P1:
        return predict_core<T, PredId::P1>(a, b, c, d, f);
      case PredId::P2:
        return predict_core<T, PredId::P2>(a, b, c, d, f);
      case PredId::P3:
        return predict_core<T, PredId::P3>(a, b, c, d, f);
      case PredId::P4:
        return predict_core<T, PredId::P4>(a, b, c, d, f);
      case PredId::P5:
        return predict_core<T, PredId::P5>(a, b, c, d, f);
      case PredId::P6:
        return predict_core<T, PredId::P6>(a, b, c, d, f);
      case PredId::P7:
        return predict_core<T, PredId::P7>(a, b, c, d, f);
      case PredId::Pp:
        if (cfg_.enable_planar_lite)
          return predict_core<T, PredId::Pp>(a, b, c, d, f);
        return predict_core<T, PredId::P3>(a, b, c, d, f);
      default:
        return clip<T>(b);
      }
    }

    inline PredId select(Class klass, const State &st) const
    {
      const auto score = [&](PredId id) -> uint16_t
      { return st.err_score[static_cast<size_t>(id)]; };

      PredId first = PredId::P2;
      PredId second = PredId::P3;
      switch (klass)
      {
      case Class::Flat:
        first = PredId::P2;
        second = PredId::P3;
        if (cfg_.enable_planar_lite && score(PredId::Pp) < score(second))
          second = PredId::Pp;
        break;

      case Class::Vertical:
        first = PredId::P0;
        second = PredId::P6;
        break;

      case Class::Horizontal:
        first = PredId::P1;
        second = PredId::P7;
        break;

      case Class::Diagonal:
      default:
      {
        const PredId tilt = (score(PredId::P4) <= score(PredId::P5)) ? PredId::P4 : PredId::P5;
        first = PredId::P3;
        second = tilt;
        if (cfg_.enable_planar_lite && score(PredId::Pp) < score(first))
          first = PredId::Pp;
        break;
      }
      }

      return (score(first) <= score(second)) ? first : second;
    }

  private:
    Config cfg_;
    int lo_;
    int hi_;
  };

  // チャネル毎の状態を保持しつつ CAS-8 を利用するための補助クラス。
  class Cas8Predictor
  {
  public:
    Cas8Predictor();

    void reset(uint32_t components);

    std::pair<uint8_t, CAS8::PredId> predict(uint32_t component,
                                             uint8_t a,
                                             uint8_t b,
                                             uint8_t c,
                                             uint8_t d,
                                             uint8_t f);

    void update(uint32_t component, CAS8::PredId pid, int abs_error);

  private:
    CAS8 core_;
    std::array<CAS8::State, 4> states_{};
    uint32_t components_ = 0;
  };
}
