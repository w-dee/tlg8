#include "tlg8_predictors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace
{
  using tlg::v8::enc::predictor_fn;

  constexpr int16_t clamp_to_i16(int32_t value) noexcept
  {
    if (value > std::numeric_limits<int16_t>::max())
      return std::numeric_limits<int16_t>::max();
    if (value < std::numeric_limits<int16_t>::min())
      return std::numeric_limits<int16_t>::min();
    return static_cast<int16_t>(value);
  }

  // JPEG-LS 同様の MED 関数。今後 predictor の差し替えが必要になっても
  // ここを通じて共有できるため、無名名前空間内で保持する。
  constexpr int16_t med(int16_t a, int16_t b, int16_t c) noexcept
  {
    const int16_t max_a_b = a > b ? a : b;
    const int16_t min_a_b = a < b ? a : b;
    if (c > max_a_b)
      return min_a_b;
    if (c < min_a_b)
      return max_a_b;
    const int32_t sum = static_cast<int32_t>(a) + static_cast<int32_t>(b) - static_cast<int32_t>(c);
    return clamp_to_i16(sum);
  }

  int16_t f0(int16_t a, int16_t b, int16_t c, [[maybe_unused]] int16_t d) noexcept
  {
    return med(a, b, c);
  }

  int16_t f1(int16_t a,
             [[maybe_unused]] int16_t b,
             int16_t c,
             [[maybe_unused]] int16_t d) noexcept
  {
    const int32_t sum = static_cast<int32_t>(a) + static_cast<int32_t>(c) + 1;
    return clamp_to_i16(sum >> 1);
  }

  int16_t f2([[maybe_unused]] int16_t a,
             int16_t b,
             int16_t c,
             [[maybe_unused]] int16_t d) noexcept
  {
    const int32_t sum = static_cast<int32_t>(b) + static_cast<int32_t>(c) + 1;
    return clamp_to_i16(sum >> 1);
  }

  int16_t f3([[maybe_unused]] int16_t a,
             int16_t b,
             [[maybe_unused]] int16_t c,
             int16_t d) noexcept
  {
    const int32_t sum = static_cast<int32_t>(b) + static_cast<int32_t>(d) + 1;
    return clamp_to_i16(sum >> 1);
  }

  int16_t f4(int16_t a,
             [[maybe_unused]] int16_t b,
             [[maybe_unused]] int16_t c,
             [[maybe_unused]] int16_t d) noexcept
  {
    return a;
  }

  int16_t f5([[maybe_unused]] int16_t a,
             int16_t b,
             [[maybe_unused]] int16_t c,
             [[maybe_unused]] int16_t d) noexcept
  {
    return b;
  }

  int16_t f6([[maybe_unused]] int16_t a,
             [[maybe_unused]] int16_t b,
             int16_t c,
             [[maybe_unused]] int16_t d) noexcept
  {
    return c;
  }

  int16_t f7([[maybe_unused]] int16_t a,
             [[maybe_unused]] int16_t b,
             [[maybe_unused]] int16_t c,
             int16_t d) noexcept
  {
    return d;
  }

  constexpr std::array<predictor_fn, tlg::v8::enc::kNumPredictors> kPredictors = {
      &f0, &f1, &f2, &f3, &f4, &f5, &f6, &f7};
}

namespace tlg::v8::enc
{
  const std::array<predictor_fn, kNumPredictors> &predictor_table()
  {
    return kPredictors;
  }
}
