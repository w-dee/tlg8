#include "tlg8_predictors.h"

namespace
{
  using tlg::v8::enc::predictor_fn;

  constexpr uint8_t med(uint8_t a, uint8_t b, uint8_t c) noexcept
  {
    const uint8_t max_a_b = a > b ? a : b;
    const uint8_t min_a_b = a < b ? a : b;
    if (c > max_a_b)
      return min_a_b;
    if (c < min_a_b)
      return max_a_b;
    return static_cast<uint8_t>(a + b - c);
  }

  uint8_t f0(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)d;
    return med(a, b, c);
  }

  uint8_t f1(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)b;
    (void)d;
    return static_cast<uint8_t>((a + c) >> 1);
  }

  uint8_t f2(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)a;
    (void)d;
    return static_cast<uint8_t>((b + c) >> 1);
  }

  uint8_t f3(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)a;
    (void)c;
    return static_cast<uint8_t>((b + d) >> 1);
  }

  uint8_t f4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)b;
    (void)c;
    (void)d;
    return a;
  }

  uint8_t f5(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)a;
    (void)c;
    (void)d;
    return b;
  }

  uint8_t f6(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)a;
    (void)b;
    (void)d;
    return c;
  }

  uint8_t f7(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept
  {
    (void)a;
    (void)b;
    (void)c;
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
