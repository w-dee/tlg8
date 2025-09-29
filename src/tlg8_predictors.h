#pragma once

#include <array>
#include <cstdint>

namespace tlg::v8::enc
{
  constexpr uint32_t kNumPredictors = 8;

  using predictor_fn = uint8_t (*)(uint8_t, uint8_t, uint8_t, uint8_t);

  const std::array<predictor_fn, kNumPredictors> &predictor_table();
}
