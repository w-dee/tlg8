#pragma once

#include <array>
#include <cstdint>

namespace tlg::v8::enc
{
  // 予測器は JPEG-LS 互換の 8 通りを当面利用する。
  // 後段のフィルター／エントロピーとの組み合わせ最適化の際に
  // インデックスで参照するため、順序は固定する。
  constexpr uint32_t kNumPredictors = 8;

  using predictor_fn = int16_t (*)(int16_t, int16_t, int16_t, int16_t);

  const std::array<predictor_fn, kNumPredictors> &predictor_table();
}
