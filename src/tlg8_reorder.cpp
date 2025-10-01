#include "tlg8_reorder.h"

#include <algorithm>

namespace
{
  constexpr std::array<uint8_t, tlg::v8::enc::kMaxBlockPixels> kHilbertOrder = {
      8 * 0 + 0,
      8 * 1 + 0,
      8 * 1 + 1,
      8 * 0 + 1,
      8 * 0 + 2,
      8 * 0 + 3,
      8 * 1 + 3,
      8 * 1 + 2,
      8 * 2 + 2,
      8 * 2 + 3,
      8 * 3 + 3,
      8 * 3 + 2,
      8 * 3 + 1,
      8 * 2 + 1,
      8 * 2 + 0,
      8 * 3 + 0,
      8 * 4 + 0,
      8 * 4 + 1,
      8 * 5 + 1,
      8 * 5 + 0,
      8 * 6 + 0,
      8 * 7 + 0,
      8 * 7 + 1,
      8 * 6 + 1,
      8 * 6 + 2,
      8 * 7 + 2,
      8 * 7 + 3,
      8 * 6 + 3,
      8 * 5 + 3,
      8 * 5 + 2,
      8 * 4 + 2,
      8 * 4 + 3,
      8 * 4 + 4,
      8 * 4 + 5,
      8 * 5 + 5,
      8 * 5 + 4,
      8 * 6 + 4,
      8 * 7 + 4,
      8 * 7 + 5,
      8 * 6 + 5,
      8 * 6 + 6,
      8 * 7 + 6,
      8 * 7 + 7,
      8 * 6 + 7,
      8 * 5 + 7,
      8 * 5 + 6,
      8 * 4 + 6,
      8 * 4 + 7,
      8 * 3 + 7,
      8 * 2 + 7,
      8 * 2 + 6,
      8 * 3 + 6,
      8 * 3 + 5,
      8 * 3 + 4,
      8 * 2 + 4,
      8 * 2 + 5,
      8 * 1 + 5,
      8 * 1 + 4,
      8 * 0 + 4,
      8 * 0 + 5,
      8 * 0 + 6,
      8 * 1 + 6,
      8 * 1 + 7,
      8 * 0 + 7,
  };

  inline constexpr std::array<uint8_t, 64> kZigzagDiagOrder = {
      0, 1, 8, 16, 9, 2, 3, 10,
      17, 24, 32, 25, 18, 11, 4, 5, 12,
      19, 26, 33, 40, 48, 41, 34, 27, 20,
      13, 6, 7, 14, 21, 28, 35, 42, 49,
      56, 57, 50, 43, 36, 29, 22,
      15, 23, 30, 37, 44, 51, 58,
      59, 52, 45, 38, 31,
      39, 46, 53, 60,
      61, 54, 47,
      55, 62, 63};

  // vertical mirror of given array
  constexpr std::array<uint8_t, 64> vert_mirror_of(const std::array<uint8_t, 64> &arr)
  {
    std::array<uint8_t, 64> mirror{};
    for (uint8_t i = 0; i < 64; ++i)
    {
      int x = i % 8;
      int y = i / 8;
      mirror[i] = arr[y * 8 + (7 - x)];
    }
    return mirror;
  }

  // horizontal mirror of given array
  constexpr std::array<uint8_t, 64> horz_mirror_of(const std::array<uint8_t, 64> &arr)
  {
    std::array<uint8_t, 64> mirror{};
    for (uint8_t i = 0; i < 64; ++i)
    {
      int x = i % 8;
      int y = i / 8;
      mirror[i] = arr[(7 - y) * 8 + x];
    }
    return mirror;
  }

  inline constexpr std::array<uint8_t, 64> kZigzagAntiDiagOrder = vert_mirror_of(kZigzagDiagOrder);

  inline constexpr std::array<uint8_t, 64> kZigzagHorzOrder = {
      8 * 0 + 0,
      8 * 0 + 1,
      8 * 0 + 2,
      8 * 0 + 3,
      8 * 0 + 4,
      8 * 0 + 5,
      8 * 0 + 6,
      8 * 0 + 7,
      8 * 1 + 7,
      8 * 1 + 6,
      8 * 1 + 5,
      8 * 1 + 4,
      8 * 1 + 3,
      8 * 1 + 2,
      8 * 1 + 1,
      8 * 1 + 0,
      8 * 2 + 0,
      8 * 2 + 1,
      8 * 2 + 2,
      8 * 2 + 3,
      8 * 2 + 4,
      8 * 2 + 5,
      8 * 2 + 6,
      8 * 2 + 7,
      8 * 3 + 7,
      8 * 3 + 6,
      8 * 3 + 5,
      8 * 3 + 4,
      8 * 3 + 3,
      8 * 3 + 2,
      8 * 3 + 1,
      8 * 3 + 0,
      8 * 4 + 0,
      8 * 4 + 1,
      8 * 4 + 2,
      8 * 4 + 3,
      8 * 4 + 4,
      8 * 4 + 5,
      8 * 4 + 6,
      8 * 4 + 7,
      8 * 5 + 7,
      8 * 5 + 6,
      8 * 5 + 5,
      8 * 5 + 4,
      8 * 5 + 3,
      8 * 5 + 2,
      8 * 5 + 1,
      8 * 5 + 0,
      8 * 6 + 0,
      8 * 6 + 1,
      8 * 6 + 2,
      8 * 6 + 3,
      8 * 6 + 4,
      8 * 6 + 5,
      8 * 6 + 6,
      8 * 6 + 7,
      8 * 7 + 7,
      8 * 7 + 6,
      8 * 7 + 5,
      8 * 7 + 4,
      8 * 7 + 3,
      8 * 7 + 2,
      8 * 7 + 1,
      8 * 7 + 0,
  };

  inline constexpr std::array<uint8_t, 64> kZigzagVertOrder = {
      8 * 0 + 0,
      8 * 1 + 0,
      8 * 2 + 0,
      8 * 3 + 0,
      8 * 4 + 0,
      8 * 5 + 0,
      8 * 6 + 0,
      8 * 7 + 0,
      8 * 7 + 1,
      8 * 6 + 1,
      8 * 5 + 1,
      8 * 4 + 1,
      8 * 3 + 1,
      8 * 2 + 1,
      8 * 1 + 1,
      8 * 0 + 1,
      8 * 0 + 2,
      8 * 1 + 2,
      8 * 2 + 2,
      8 * 3 + 2,
      8 * 4 + 2,
      8 * 5 + 2,
      8 * 6 + 2,
      8 * 7 + 2,
      8 * 7 + 3,
      8 * 6 + 3,
      8 * 5 + 3,
      8 * 4 + 3,
      8 * 3 + 3,
      8 * 2 + 3,
      8 * 1 + 3,
      8 * 0 + 3,
      8 * 0 + 4,
      8 * 1 + 4,
      8 * 2 + 4,
      8 * 3 + 4,
      8 * 4 + 4,
      8 * 5 + 4,
      8 * 6 + 4,
      8 * 7 + 4,
      8 * 7 + 5,
      8 * 6 + 5,
      8 * 5 + 5,
      8 * 4 + 5,
      8 * 3 + 5,
      8 * 2 + 5,
      8 * 1 + 5,
      8 * 0 + 5,
      8 * 0 + 6,
      8 * 1 + 6,
      8 * 2 + 6,
      8 * 3 + 6,
      8 * 4 + 6,
      8 * 5 + 6,
      8 * 6 + 6,
      8 * 7 + 6,
      8 * 7 + 7,
      8 * 6 + 7,
      8 * 5 + 7,
      8 * 4 + 7,
      8 * 3 + 7,
      8 * 2 + 7,
      8 * 1 + 7,
      8 * 0 + 7,
  };

  inline constexpr std::array<uint8_t, 64> kZigzagNNESSWOrder = {
      0,
      2,
      11,
      12,
      27,
      28,
      43,
      44,
      1,
      3,
      10,
      13,
      26,
      29,
      42,
      45,
      4,
      9,
      14,
      25,
      30,
      41,
      46,
      57,
      5,
      8,
      15,
      24,
      31,
      40,
      47,
      56,
      6,
      16,
      23,
      32,
      39,
      48,
      55,
      58,
      7,
      17,
      22,
      33,
      38,
      49,
      54,
      59,
      18,
      21,
      34,
      37,
      50,
      53,
      60,
      62,
      19,
      20,
      35,
      36,
      51,
      52,
      61,
      63,
  };

  inline constexpr std::array<uint8_t, 64> kZigzagNNWSSEOrder = vert_mirror_of(kZigzagNNESSWOrder);

  inline constexpr std::array<uint8_t, 64> kZigzagNEESWWOrder = {
      0,
      1,
      4,
      5,
      6,
      7,
      18,
      19,
      2,
      3,
      9,
      8,
      16,
      17,
      21,
      20,
      11,
      10,
      14,
      15,
      23,
      22,
      34,
      35,
      12,
      13,
      25,
      24,
      32,
      33,
      37,
      36,
      27,
      26,
      30,
      31,
      39,
      38,
      50,
      51,
      28,
      29,
      41,
      40,
      48,
      49,
      53,
      52,
      43,
      42,
      46,
      47,
      55,
      54,
      60,
      61,
      44,
      45,
      57,
      56,
      58,
      59,
      62,
      63,
  };

  inline constexpr std::array<uint8_t, 64> kZigzagNWWSEEOrder = horz_mirror_of(kZigzagNEESWWOrder);

  constexpr bool _check_reorder_array(const std::array<uint8_t, 64> &arr)
  {
    std::array<bool, 64> seen{};
    for (uint8_t v : arr)
    {
      if (v >= seen.size())
        return false;
      if (seen[v])
        return false;
      seen[v] = true;
    }
    for (bool v : seen)
    {
      if (!v)
        return false;
    }
    return true;
  }

  static_assert(_check_reorder_array(kHilbertOrder), "kHilbertOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagDiagOrder), "kZigzagDiagOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagAntiDiagOrder), "kZigzagAntiDiagOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagHorzOrder), "kZigzagHorzOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagVertOrder), "kZigzagVertOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagNNESSWOrder), "kZigzagNNESSWOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagNEESWWOrder), "kZigzagNEESWWOrder の内容が不正です");
  static_assert(_check_reorder_array(kZigzagNWWSEEOrder), "kZigzagNWWSEEOrder の内容が不正です");

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
