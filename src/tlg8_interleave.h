#pragma once

#include <cstdint>

#include "tlg8_block.h"

namespace tlg::v8::enc
{
  enum class InterleaveFilter : uint8_t
  {
    None = 0, // インターリーブフィルターを使わない
    Interleave = 1, // インターリーブフィルターを使う
  };
  inline constexpr uint32_t kNumInterleaveFilter = 2;

  void apply_interleave_filter(InterleaveFilter filter,
                               component_colors &colors,
                               uint32_t components,
                               uint32_t value_count);

  void undo_interleave_filter(InterleaveFilter filter,
                              component_colors &colors,
                              uint32_t components,
                              uint32_t value_count);
}
