#include "tlg8_reorder.h"

#include <algorithm>

namespace
{
  constexpr std::array<uint8_t, tlg::v8::enc::kMaxBlockPixels> kHilbertOrder = {
      8 * 0 + 0, 8 * 1 + 0, 8 * 1 + 1, 8 * 0 + 1,
      8 * 0 + 2, 8 * 0 + 3, 8 * 1 + 3, 8 * 1 + 2,
      8 * 2 + 2, 8 * 2 + 3, 8 * 3 + 3, 8 * 3 + 2,
      8 * 3 + 1, 8 * 2 + 1, 8 * 2 + 0, 8 * 3 + 0,
      8 * 4 + 0, 8 * 4 + 1, 8 * 5 + 1, 8 * 5 + 0,
      8 * 6 + 0, 8 * 7 + 0, 8 * 7 + 1, 8 * 6 + 1,
      8 * 6 + 2, 8 * 7 + 2, 8 * 7 + 3, 8 * 6 + 3,
      8 * 5 + 3, 8 * 5 + 2, 8 * 4 + 2, 8 * 4 + 3,
      8 * 4 + 4, 8 * 4 + 5, 8 * 5 + 5, 8 * 5 + 4,
      8 * 6 + 4, 8 * 7 + 4, 8 * 7 + 5, 8 * 6 + 5,
      8 * 6 + 6, 8 * 7 + 6, 8 * 7 + 7, 8 * 6 + 7,
      8 * 5 + 7, 8 * 5 + 6, 8 * 4 + 6, 8 * 4 + 7,
      8 * 3 + 7, 8 * 2 + 7, 8 * 2 + 6, 8 * 3 + 6,
      8 * 3 + 5, 8 * 3 + 4, 8 * 2 + 4, 8 * 2 + 5,
      8 * 1 + 5, 8 * 1 + 4, 8 * 0 + 4, 8 * 0 + 5,
      8 * 0 + 6, 8 * 1 + 6, 8 * 1 + 7, 8 * 0 + 7,
  };

  inline uint32_t build_hilbert_sequence(uint32_t block_w,
                                         uint32_t block_h,
                                         std::array<uint8_t, tlg::v8::enc::kMaxBlockPixels> &sequence)
  {
    const uint32_t value_count = block_w * block_h;
    if (value_count == 0)
      return 0;

    uint32_t index = 0;
    for (uint8_t order : kHilbertOrder)
    {
      const uint32_t x = order % tlg::v8::enc::kBlockSize;
      const uint32_t y = order / tlg::v8::enc::kBlockSize;
      if (x >= block_w || y >= block_h)
        continue;
      if (index < sequence.size())
        sequence[index] = static_cast<uint8_t>(y * block_w + x);
      ++index;
    }
    return index;
  }
}

namespace tlg::v8::enc
{
  void reorder_to_hilbert(component_colors &colors, uint32_t components, uint32_t block_w, uint32_t block_h)
  {
    const uint32_t value_count = block_w * block_h;
    if (value_count == 0)
      return;

    std::array<uint8_t, kMaxBlockPixels> sequence{};
    const uint32_t sequence_size = build_hilbert_sequence(block_w, block_h, sequence);
    if (sequence_size != value_count)
      return;

    std::array<int16_t, kMaxBlockPixels> temp{};
    const uint32_t used_components = std::min<uint32_t>(components, colors.values.size());
    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      for (uint32_t i = 0; i < value_count; ++i)
        temp[i] = colors.values[comp][sequence[i]];
      for (uint32_t i = 0; i < value_count; ++i)
        colors.values[comp][i] = temp[i];
    }
  }

  void reorder_from_hilbert(component_colors &colors, uint32_t components, uint32_t block_w, uint32_t block_h)
  {
    const uint32_t value_count = block_w * block_h;
    if (value_count == 0)
      return;

    std::array<uint8_t, kMaxBlockPixels> sequence{};
    const uint32_t sequence_size = build_hilbert_sequence(block_w, block_h, sequence);
    if (sequence_size != value_count)
      return;

    std::array<int16_t, kMaxBlockPixels> temp{};
    const uint32_t used_components = std::min<uint32_t>(components, colors.values.size());
    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      for (uint32_t i = 0; i < value_count; ++i)
        temp[i] = colors.values[comp][i];
      for (uint32_t i = 0; i < value_count; ++i)
        colors.values[comp][sequence[i]] = temp[i];
    }
  }
}

