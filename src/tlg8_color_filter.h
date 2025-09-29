#pragma once

#include "tlg8_block.h"

#include <cstdint>

namespace tlg::v8::enc
{
  inline constexpr int kColorFilterPermutations = 6;
  inline constexpr int kColorFilterPrimaryPredictors = 4;
  inline constexpr int kColorFilterSecondaryPredictors = 4;
  inline constexpr int kColorFilterCodeCount =
      kColorFilterPermutations * kColorFilterPrimaryPredictors * kColorFilterSecondaryPredictors;

  // predictor の出力残差にカラー相関フィルターを適用する。
  void apply_color_filter(int code, component_colors &colors, uint32_t components, uint32_t value_count);

  // カラー相関フィルターを逆変換し、元の残差信号を復元する。
  void undo_color_filter(int code, component_colors &colors, uint32_t components, uint32_t value_count);
}
