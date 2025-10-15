#pragma once

#include <array>
#include <cstdint>

#include "tlg8_block.h"

namespace tlg::v8::enc
{
  enum class ReorderPattern : uint32_t
  {
    Hilbert = 0,
    ZigzagDiag,
    ZigzagAntiDiag,
    ZigzagHorz,
    ZigzagVert,
    ZigzagNNESSW,
    ZigzagNEESWW,
    ZigzagNWWSEE,
    Count,
  };

  constexpr uint32_t kReorderPatternCount = static_cast<uint32_t>(ReorderPattern::Count);

  // カラーフィルター後の係数を指定した走査順へ並び替える。
  void reorder_to_scan(component_colors &colors,
                       uint32_t components,
                       uint32_t block_w,
                       uint32_t block_h,
                       ReorderPattern pattern);

  // エントロピー復号後の係数を元の走査順へ戻す。
  void reorder_from_scan(component_colors &colors,
                         uint32_t components,
                         uint32_t block_w,
                         uint32_t block_h,
                         ReorderPattern pattern);
}

