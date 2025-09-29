#pragma once

#include <array>
#include <cstdint>

#include "tlg8_block.h"

namespace tlg::v8::enc
{
  // カラーフィルター後の係数をヒルベルト順へ並び替える。
  void reorder_to_hilbert(component_colors &colors, uint32_t components, uint32_t block_w, uint32_t block_h);

  // エントロピー復号後の係数を元の走査順へ戻す。
  void reorder_from_hilbert(component_colors &colors, uint32_t components, uint32_t block_w, uint32_t block_h);
}

