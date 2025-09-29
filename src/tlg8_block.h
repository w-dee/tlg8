#pragma once

#include <array>
#include <cstdint>

namespace tlg::v8::enc
{
  constexpr uint32_t kBlockSize = 8;
  constexpr uint32_t kMaxBlockPixels = kBlockSize * kBlockSize;

  // 8x8 ブロックをパイプラインで受け渡すための色差コンテナ。
  // 将来的には predictor の後にカラー相関フィルターや並び替えを
  // 差し込む予定であり、その各段の入出力を統一する目的で
  // component_colors を定義している。
  struct component_colors
  {
    using values_64 = std::array<int16_t, kMaxBlockPixels>;
    std::array<values_64, 4> values{};
  };
}
