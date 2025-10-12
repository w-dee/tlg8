#include "tlg8_predictors.h"

namespace
{
  using tlg::v8::enc::predictor_fn;

  // JPEG-LS 同様の MED 関数。今後 predictor の差し替えが必要になっても
  // ここを通じて共有できるため、無名名前空間内で保持する。
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

  uint8_t f0(uint8_t a, uint8_t b, uint8_t c, [[maybe_unused]] uint8_t d) noexcept
  {
    return med(a, b, c);
  }

  uint8_t f1(uint8_t a,
             [[maybe_unused]] uint8_t b,
             uint8_t c,
             [[maybe_unused]] uint8_t d) noexcept
  {
    return static_cast<uint8_t>((a + c + 1) >> 1);
  }

  uint8_t f2([[maybe_unused]] uint8_t a,
             uint8_t b,
             uint8_t c,
             [[maybe_unused]] uint8_t d) noexcept
  {
    return static_cast<uint8_t>((b + c + 1) >> 1);
  }

  uint8_t f3([[maybe_unused]] uint8_t a,
             uint8_t b,
             [[maybe_unused]] uint8_t c,
             uint8_t d) noexcept
  {
    return static_cast<uint8_t>((b + d + 1) >> 1);
  }

  uint8_t f4(uint8_t a,
             [[maybe_unused]] uint8_t b,
             [[maybe_unused]] uint8_t c,
             [[maybe_unused]] uint8_t d) noexcept
  {
    return a;
  }

  uint8_t f5([[maybe_unused]] uint8_t a,
             uint8_t b,
             [[maybe_unused]] uint8_t c,
             [[maybe_unused]] uint8_t d) noexcept
  {
    return b;
  }

  uint8_t f6([[maybe_unused]] uint8_t a,
             [[maybe_unused]] uint8_t b,
             uint8_t c,
             [[maybe_unused]] uint8_t d) noexcept
  {
    return c;
  }

  uint8_t f7([[maybe_unused]] uint8_t a,
             [[maybe_unused]] uint8_t b,
             [[maybe_unused]] uint8_t c,
             uint8_t d) noexcept
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
