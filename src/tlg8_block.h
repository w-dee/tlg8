#pragma once

#include <array>
#include <cstdint>

namespace tlg::v8::enc
{
  constexpr uint32_t kBlockSize = 8;
  constexpr uint32_t kMaxBlockPixels = kBlockSize * kBlockSize;

  struct component_colors
  {
    using values_64 = std::array<int16_t, kMaxBlockPixels>;
    std::array<values_64, 4> values{};
  };
}
