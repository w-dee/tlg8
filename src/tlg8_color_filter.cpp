#include "tlg8_color_filter.h"

#include <array>

namespace
{
  struct ColorFilterParams
  {
    int perm = 0;
    int primary = 0;
    int secondary = 0;
  };

  constexpr std::array<std::array<uint8_t, 3>, tlg::v8::enc::kColorFilterPermutations> kColorPermutations = {
      std::array<uint8_t, 3>{0, 1, 2},
      std::array<uint8_t, 3>{0, 2, 1},
      std::array<uint8_t, 3>{1, 0, 2},
      std::array<uint8_t, 3>{1, 2, 0},
      std::array<uint8_t, 3>{2, 0, 1},
      std::array<uint8_t, 3>{2, 1, 0},
  };

  inline int wrap_index(int value, int mod)
  {
    if (mod <= 0)
      return 0;
    if (value < 0)
      value = 0;
    if (value >= mod)
      value %= mod;
    return value;
  }

  inline ColorFilterParams decode_color_filter_code(int code)
  {
    if (code < 0)
      code = 0;
    if (code >= tlg::v8::enc::kColorFilterCodeCount)
      code %= tlg::v8::enc::kColorFilterCodeCount;

    const int perm_raw = (code >> 4) & 0x7;
    const int primary = (code >> 2) & 0x3;
    const int secondary = code & 0x3;

    ColorFilterParams params{};
    params.perm = wrap_index(perm_raw, tlg::v8::enc::kColorFilterPermutations);
    params.primary = wrap_index(primary, tlg::v8::enc::kColorFilterPrimaryPredictors);
    params.secondary = wrap_index(secondary, tlg::v8::enc::kColorFilterSecondaryPredictors);
    return params;
  }

  inline int predict_primary(int mode, int c0)
  {
    switch (mode & 0x3)
    {
    case 0:
      return 0;
    case 1:
      return c0;
    case 2:
      return (c0 + 1) / 2;
    case 3:
      return (3 * c0 + 1) / 2;
    default:
      return 0;
    }
  }

  inline int predict_secondary(int mode, int c0, int reference1)
  {
    switch (mode & 0x3)
    {
    case 0:
      return 0;
    case 1:
      return c0;
    case 2:
      return reference1;
    case 3:
      return (c0 + reference1 + 1) / 2;
    default:
      return 0;
    }
  }

}

namespace tlg::v8::enc
{
  void apply_color_filter(int code, component_colors &colors, uint32_t components, uint32_t value_count)
  {
    if (components < 3 || value_count == 0)
      return;

    const ColorFilterParams params = decode_color_filter_code(code);
    const auto &perm = kColorPermutations[static_cast<std::size_t>(params.perm)];

    auto &b = colors.values[0];
    auto &g = colors.values[1];
    auto &r = colors.values[2];

    for (uint32_t i = 0; i < value_count; ++i)
    {
      const std::array<int, 3> source = {
          static_cast<int>(b[i]),
          static_cast<int>(g[i]),
          static_cast<int>(r[i]),
      };

      const int d0 = source[perm[0]];
      const int d1 = source[perm[1]];
      const int d2 = source[perm[2]];

      const int predicted1 = predict_primary(params.primary, d0);
      const int predicted2 = predict_secondary(params.secondary, d0, d1);

      const int residual1 = d1 - predicted1;
      const int residual2 = d2 - predicted2;

      b[i] = static_cast<int16_t>(d0);
      g[i] = static_cast<int16_t>(residual1);
      r[i] = static_cast<int16_t>(residual2);
    }
  }

  void undo_color_filter(int code, component_colors &colors, uint32_t components, uint32_t value_count)
  {
    if (components < 3 || value_count == 0)
      return;

    const ColorFilterParams params = decode_color_filter_code(code);
    const auto &perm = kColorPermutations[static_cast<std::size_t>(params.perm)];

    auto &b = colors.values[0];
    auto &g = colors.values[1];
    auto &r = colors.values[2];

    for (uint32_t i = 0; i < value_count; ++i)
    {
      const int c0 = static_cast<int>(b[i]);
      const int c1 = static_cast<int>(g[i]);
      const int c2 = static_cast<int>(r[i]);

      const int restored0 = c0;
      const int restored1 = c1 + predict_primary(params.primary, restored0);
      const int restored2 = c2 + predict_secondary(params.secondary, restored0, restored1);

      std::array<int, 3> destination = {0, 0, 0};
      destination[perm[0]] = restored0;
      destination[perm[1]] = restored1;
      destination[perm[2]] = restored2;

      b[i] = static_cast<int16_t>(destination[0]);
      g[i] = static_cast<int16_t>(destination[1]);
      r[i] = static_cast<int16_t>(destination[2]);
    }
  }
}
