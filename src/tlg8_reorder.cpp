#include "tlg8_reorder.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace
{
  using reorder_table = std::array<uint8_t, tlg::v8::enc::kMaxBlockPixels>;
  using component_values_array = tlg::v8::enc::component_colors::values_64;

  constexpr reorder_table kHilbertOrder = {
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

  inline constexpr reorder_table kZigzagDiagOrder = {
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
  constexpr reorder_table vert_mirror_of(const reorder_table &arr)
  {
    reorder_table mirror{};
    for (uint8_t i = 0; i < 64; ++i)
    {
      int x = i % 8;
      int y = i / 8;
      mirror[i] = arr[y * 8 + (7 - x)];
    }
    return mirror;
  }

  // horizontal mirror of given array
  constexpr reorder_table horz_mirror_of(const reorder_table &arr)
  {
    reorder_table mirror{};
    for (uint8_t i = 0; i < 64; ++i)
    {
      int x = i % 8;
      int y = i / 8;
      mirror[i] = arr[(7 - y) * 8 + x];
    }
    return mirror;
  }

  inline constexpr reorder_table kZigzagAntiDiagOrder = vert_mirror_of(kZigzagDiagOrder);

  inline constexpr reorder_table kZigzagHorzOrder = {
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

  inline constexpr reorder_table kZigzagVertOrder = {
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

  inline constexpr reorder_table kZigzagNNESSWOrder = {
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

  inline constexpr reorder_table kZigzagNNWSSEOrder = vert_mirror_of(kZigzagNNESSWOrder);

  inline constexpr reorder_table kZigzagNEESWWOrder = {
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

  inline constexpr reorder_table kZigzagNWWSEEOrder = horz_mirror_of(kZigzagNEESWWOrder);

  constexpr bool _check_reorder_array(const reorder_table &arr)
  {
    std::array<bool, tlg::v8::enc::kMaxBlockPixels> seen{};
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

  template <bool ToScan, const reorder_table &Table, size_t... Indices>
  inline void reorder_component_impl(component_values_array &component_values,
                                     component_values_array &temp,
                                     std::index_sequence<Indices...>)
  {
    if constexpr (ToScan)
    {
      ((void)(temp[Indices] =
                  component_values[static_cast<size_t>(std::get<Indices>(Table))]), ...);
      ((void)(component_values[Indices] = temp[Indices]), ...);
    }
    else
    {
      ((void)(temp[Indices] = component_values[Indices]), ...);
      ((void)(component_values[static_cast<size_t>(std::get<Indices>(Table))] = temp[Indices]),
       ...);
    }
  }

  template <bool ToScan, const reorder_table &Table>
  inline void reorder_component_with_table(component_values_array &component_values,
                                           component_values_array &temp)
  {
    reorder_component_impl<ToScan, Table>(
        component_values, temp,
        std::make_index_sequence<tlg::v8::enc::kMaxBlockPixels>{});
  }

  template <bool ToScan>
  inline void reorder_component(component_values_array &component_values,
                                tlg::v8::enc::ReorderPattern pattern,
                                component_values_array &temp)
  {
    using tlg::v8::enc::ReorderPattern;
    switch (pattern)
    {
    case ReorderPattern::Hilbert:
      reorder_component_with_table<ToScan, kHilbertOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagDiag:
      reorder_component_with_table<ToScan, kZigzagDiagOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagAntiDiag:
      reorder_component_with_table<ToScan, kZigzagAntiDiagOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagHorz:
      reorder_component_with_table<ToScan, kZigzagHorzOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagVert:
      reorder_component_with_table<ToScan, kZigzagVertOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagNNESSW:
      reorder_component_with_table<ToScan, kZigzagNNESSWOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagNEESWW:
      reorder_component_with_table<ToScan, kZigzagNEESWWOrder>(component_values, temp);
      break;
    case ReorderPattern::ZigzagNWWSEE:
      reorder_component_with_table<ToScan, kZigzagNWWSEEOrder>(component_values, temp);
      break;
    default:
      reorder_component_with_table<ToScan, kHilbertOrder>(component_values, temp);
      break;
    }
  }
}

namespace tlg::v8::enc
{
  namespace
  {
    // enum 値順（0..7）でテーブルを並べる（JSONL の reorder インデックスと一致させる）
    constexpr std::array<const reorder_table *, kReorderPatternCount> kReorderTvTables = {
        &kHilbertOrder,
        &kZigzagDiagOrder,
        &kZigzagAntiDiagOrder,
        &kZigzagHorzOrder,
        &kZigzagVertOrder,
        &kZigzagNNESSWOrder,
        &kZigzagNEESWWOrder,
        &kZigzagNWWSEEOrder,
    };
  }

  std::array<float, kReorderPatternCount> compute_reorder_tv_mean(
      const std::array<float, kMaxBlockPixels> &luma)
  {
    std::array<float, kReorderPatternCount> out{};

    constexpr float inv_steps = 1.0f / 63.0f;
    for (uint32_t cls = 0; cls < kReorderPatternCount; ++cls)
    {
      const reorder_table &order = *kReorderTvTables[cls];
      float tv = 0.0f;
      float prev = luma[static_cast<size_t>(order[0])];
      for (size_t t = 1; t < kMaxBlockPixels; ++t)
      {
        const float cur = luma[static_cast<size_t>(order[t])];
        tv += std::abs(cur - prev);
        prev = cur;
      }
      out[cls] = tv * inv_steps;
    }
    return out;
  }

  std::array<float, kReorderPatternCount> compute_reorder_tv2_mean(
      const std::array<float, kMaxBlockPixels> &luma)
  {
    std::array<float, kReorderPatternCount> out{};
    constexpr float inv_steps = 1.0f / 63.0f;
    for (uint32_t cls = 0; cls < kReorderPatternCount; ++cls)
    {
      const reorder_table &order = *kReorderTvTables[cls];
      float tv2 = 0.0f;
      float prev = luma[static_cast<size_t>(order[0])];
      for (size_t t = 1; t < kMaxBlockPixels; ++t)
      {
        const float cur = luma[static_cast<size_t>(order[t])];
        const float d = cur - prev;
        tv2 += d * d;
        prev = cur;
      }
      out[cls] = tv2 * inv_steps;
    }
    return out;
  }

  void reorder_to_scan(component_colors &colors,
                       uint32_t components,
                       uint32_t block_w,
                       uint32_t block_h,
                       ReorderPattern pattern)
  {
    if (block_w != kBlockSize || block_h != kBlockSize)
    {
      // 8x8 ブロック以外は処理を簡略化するためリオーダーを行わない
      return;
    }
    component_colors::values_64 temp{};
    const uint32_t used_components = std::min<uint32_t>(components, colors.values.size());
    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      reorder_component<true>(colors.values[comp], pattern, temp);
    }
  }

  void reorder_from_scan(component_colors &colors,
                         uint32_t components,
                         uint32_t block_w,
                         uint32_t block_h,
                         ReorderPattern pattern)
  {
    if (block_w != kBlockSize || block_h != kBlockSize)
    {
      // 8x8 ブロック以外は処理を簡略化するためリオーダーを行わない
      return;
    }
    component_colors::values_64 temp{};
    const uint32_t used_components = std::min<uint32_t>(components, colors.values.size());
    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      reorder_component<false>(colors.values[comp], pattern, temp);
    }
  }
}
