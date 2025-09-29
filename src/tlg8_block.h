#pragma once

#include <array>
#include <cstdint>

namespace tlg::v8::enc
{
  constexpr uint32_t kBlockSize = 8;
  constexpr uint32_t kMaxBlockPixels = kBlockSize * kBlockSize;

  // 8x8 ブロックをパイプラインで受け渡すための色差コンテナ。
  // predictor で得た予測誤差をカラー相関フィルターや並び替えに
  // 受け渡し、最終的にエントロピー符号化へ供給する際に段間の
  // データ形式を揃える目的で component_colors を定義している。
  struct component_colors
  {
    using values_64 = std::array<int16_t, kMaxBlockPixels>;
    std::array<values_64, 4> values{};
  };
}
