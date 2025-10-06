#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_block_side_info.h"
#include "tlg8_color_filter.h"
#include "tlg8_entropy.h"
#include "tlg8_interleave.h"
#include "tlg8_reorder.h"
#include "tlg8_predictors.h"
#include "tlg8_varint.h"
#include "tlg_io_common.h"
#include "image_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{
  inline constexpr double kEarlyExitGiveUpRate = 1.3;
  using tlg::v8::enc::BlockChoiceEncoding;
  using tlg::v8::enc::kColorFilterCodeCount;
  using tlg::v8::enc::kGolombColumnCount;
  using tlg::v8::enc::kGolombRowCount;
  using tlg::v8::enc::kGolombRowSum;
  using tlg::v8::enc::kNumPredictors;
  using tlg::v8::detail::bitio::BitWriter;
  using tlg::v8::detail::bitio::put_varuint;
  using tlg::v8::detail::bitio::varuint_bits;

  // ヒストグラムで得た頻度の高い順に predictor を試行するためのインデックス列。
  constexpr std::array<uint32_t, kNumPredictors> kPredictorTrialOrder = {
      0,
      4,
      1,
      3,
      5,
      2,
      7,
      6,
  };

  // ヒストグラムで得た頻度の高い順にカラー相関フィルターを試行するためのインデックス列。
  constexpr std::array<uint32_t, kColorFilterCodeCount> kColorFilterTrialOrder = {
      71,
      86,
      7,
      87,
      23,
      0,
      39,
      6,
      55,
      70,
      27,
      94,
      79,
      75,
      31,
      37,
      59,
      95,
      11,
      91,
      43,
      22,
      53,
      5,
      19,
      8,
      14,
      78,
      51,
      69,
      57,
      40,
      15,
      38,
      3,
      56,
      25,
      13,
      88,
      16,
      36,
      93,
      24,
      4,
      10,
      21,
      32,
      2,
      72,
      26,
      77,
      48,
      54,
      90,
      44,
      85,
      18,
      41,
      64,
      74,
      45,
      30,
      67,
      73,
      50,
      61,
      80,
      47,
      1,
      83,
      29,
      35,
      60,
      63,
      9,
      49,
      58,
      42,
      17,
      33,
      52,
      66,
      82,
      89,
      65,
      84,
      68,
      20,
      46,
      34,
      81,
      92,
      62,
      12,
      76,
      28,
  };

  inline constexpr int A_SHIFT = 2;
  inline constexpr int A_BIAS = 1 << (A_SHIFT - 1);

  inline int reduce_a(int a)
  {
    return (a + A_BIAS) >> A_SHIFT;
  }

  inline int mix_a_m(int a, int m)
  {
    return ((m << A_SHIFT) + a * 3 + 2) >> 2;
  }

  using GolombRow = std::array<uint16_t, kGolombColumnCount>;

  inline uint32_t select_best_k(uint32_t m)
  {
    uint32_t best_k = 0;
    uint32_t best_cost = std::numeric_limits<uint32_t>::max();
    for (uint32_t k = 0; k < kGolombColumnCount; ++k)
    {
      const uint32_t q = (k > 0) ? (m >> k) : m;
      const uint32_t cost = q + 1u + k;
      if (cost < best_cost)
      {
        best_cost = cost;
        best_k = k;
      }
    }
    return best_k;
  }

  bool normalize_histogram_row_local(const std::array<uint64_t, kGolombColumnCount> &hist,
                                     GolombRow &row)
  {
    row.fill(0);
    uint64_t total = 0;
    for (auto value : hist)
      total += value;
    if (total == 0)
      return false;

    std::array<uint64_t, kGolombColumnCount> remainders{};
    uint32_t assigned = 0;
    for (std::size_t col = 0; col < kGolombColumnCount; ++col)
    {
      const uint64_t numerator = hist[col] * static_cast<uint64_t>(kGolombRowSum);
      const uint16_t base = static_cast<uint16_t>(std::min<uint64_t>(numerator / total, kGolombRowSum));
      row[col] = base;
      remainders[col] = numerator % total;
      assigned += base;
    }

    if (assigned > kGolombRowSum)
    {
      uint32_t overflow = assigned - kGolombRowSum;
      for (std::size_t col = kGolombColumnCount; col-- > 0 && overflow > 0;)
      {
        const uint16_t reducible = static_cast<uint16_t>(std::min<uint32_t>(overflow, row[col]));
        row[col] = static_cast<uint16_t>(row[col] - reducible);
        overflow -= reducible;
      }
      assigned = kGolombRowSum;
    }

    while (assigned < kGolombRowSum)
    {
      std::size_t best_col = kGolombColumnCount - 1;
      uint64_t best_remainder = 0;
      for (std::size_t col = 0; col < kGolombColumnCount; ++col)
      {
        if (row[col] >= kGolombRowSum)
          continue;
        if (remainders[col] > best_remainder)
        {
          best_remainder = remainders[col];
          best_col = col;
        }
      }
      if (best_remainder == 0)
      {
        for (std::size_t col = 0; col < kGolombColumnCount; ++col)
        {
          if (row[col] >= kGolombRowSum)
            continue;
          best_col = col;
          if (hist[col] > 0)
            break;
        }
      }
      ++row[best_col];
      if (remainders[best_col] > 0)
        --remainders[best_col];
      ++assigned;
    }
    return true;
  }

  // ブロック毎に選ばれたメタデータを一時保持する構造体。
  struct block_choice
  {
    uint32_t predictor = 0;
    uint32_t filter = 0;
    uint32_t entropy = 0;
    uint32_t interleave = 0;
  };

  uint64_t estimate_value_sequence_bits(BlockChoiceEncoding mode,
                                        const std::vector<uint32_t> &values,
                                        uint32_t value_bits)
  {
    if (values.empty() || value_bits == 0)
      return 0;

    switch (mode)
    {
    case BlockChoiceEncoding::Raw:
      return static_cast<uint64_t>(values.size()) * static_cast<uint64_t>(value_bits);

    case BlockChoiceEncoding::SameAsPrevious:
    {
      uint64_t bits = value_bits;
      uint32_t previous = values.front();
      for (std::size_t index = 1; index < values.size(); ++index)
      {
        const uint32_t current = values[index];
        bits += 1;
        if (current != previous)
        {
          bits += value_bits;
          previous = current;
        }
      }
      return bits;
    }

    case BlockChoiceEncoding::RunLength:
    {
      uint64_t bits = 0;
      std::size_t index = 0;
      while (index < values.size())
      {
        const uint32_t current = values[index];
        std::size_t run_end = index + 1;
        while (run_end < values.size() && values[run_end] == current)
          ++run_end;
        const uint32_t run_length = static_cast<uint32_t>(run_end - index);
        bits += value_bits;
        bits += varuint_bits(run_length - 1);
        index = run_end;
      }
      return bits;
    }

    default:
      break;
    }

    return std::numeric_limits<uint64_t>::max();
  }

  BlockChoiceEncoding select_best_sequence_mode(const std::vector<uint32_t> &values,
                                                uint32_t value_bits)
  {
    if (values.empty() || value_bits == 0)
      return BlockChoiceEncoding::Raw;

    BlockChoiceEncoding best_mode = BlockChoiceEncoding::Raw;
    uint64_t best_bits = estimate_value_sequence_bits(best_mode, values, value_bits);
    const std::array<BlockChoiceEncoding, 2> candidates = {
        BlockChoiceEncoding::SameAsPrevious,
        BlockChoiceEncoding::RunLength,
    };
    for (auto candidate : candidates)
    {
      const uint64_t bits = estimate_value_sequence_bits(candidate, values, value_bits);
      if (bits < best_bits)
      {
        best_bits = bits;
        best_mode = candidate;
      }
    }
    return best_mode;
  }

  void encode_value_sequence(BitWriter &writer,
                             BlockChoiceEncoding mode,
                             const std::vector<uint32_t> &values,
                             uint32_t value_bits)
  {
    if (values.empty() || value_bits == 0)
      return;

    switch (mode)
    {
    case BlockChoiceEncoding::Raw:
      for (uint32_t value : values)
        writer.put_upto8(value, value_bits);
      break;

    case BlockChoiceEncoding::SameAsPrevious:
    {
      writer.put_upto8(values.front(), value_bits);
      uint32_t previous = values.front();
      for (std::size_t index = 1; index < values.size(); ++index)
      {
        const uint32_t current = values[index];
        const bool same = (current == previous);
        writer.put_upto8(same ? 1u : 0u, 1);
        if (!same)
        {
          writer.put_upto8(current, value_bits);
          previous = current;
        }
      }
      break;
    }

    case BlockChoiceEncoding::RunLength:
    {
      std::size_t index = 0;
      while (index < values.size())
      {
        const uint32_t current = values[index];
        std::size_t run_end = index + 1;
        while (run_end < values.size() && values[run_end] == current)
          ++run_end;
        const uint32_t run_length = static_cast<uint32_t>(run_end - index);
        writer.put_upto8(current, value_bits);
        put_varuint(writer, run_length - 1);
        index = run_end;
      }
      break;
    }

    default:
      break;
    }
  }

  uint64_t accumulate_plain_histogram(std::array<uint64_t, kGolombColumnCount> &hist,
                                      const int16_t *values,
                                      uint32_t count)
  {
    uint64_t processed = 0;
    int a = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
      const int e = static_cast<int>(values[i]);
      const uint32_t m = (e >= 0) ? static_cast<uint32_t>(2u * static_cast<uint32_t>(e))
                                  : static_cast<uint32_t>(-2 * e - 1);
      const uint32_t k = select_best_k(m);
      hist[k] += 1u;
      a = mix_a_m(a, static_cast<int>(m));
      ++processed;
    }
    return processed;
  }

  uint64_t accumulate_run_length_histogram(std::array<uint64_t, kGolombColumnCount> &hist,
                                           const int16_t *values,
                                           uint32_t count)
  {
    uint64_t processed = 0;
    int a = 0;
    uint32_t index = 0;
    while (index < count)
    {
      if (values[index] == 0)
      {
        ++index;
        continue;
      }
      while (index < count && values[index] != 0)
      {
        int64_t mapped = (values[index] >= 0) ? static_cast<int64_t>(values[index]) * 2
                                              : -static_cast<int64_t>(values[index]) * 2 - 1;
        mapped -= 1;
        if (mapped < 0)
          mapped = 0;
        const uint32_t m = static_cast<uint32_t>(mapped);
        const uint32_t k = select_best_k(m);
        hist[k] += 1u;
        a = mix_a_m(a, static_cast<int>(m));
        ++index;
        ++processed;
      }
    }
    return processed;
  }

  // タイル全体を 8x8 ブロックへ分割し、予測→カラー相関フィルター→
  // ヒルベルト曲線による並び替え→エントロピー符号と流すパイプライン。
  // 並び替え段はタイル端で縮むブロックにも対応させている。
  struct tile_accessor
  {
    const uint8_t *base;
    uint32_t image_width;
    uint32_t components;
    uint32_t origin_x;
    uint32_t origin_y;
    uint32_t tile_width;
    uint32_t tile_height;
    size_t row_stride;

    tile_accessor(const uint8_t *base_ptr,
                  uint32_t width,
                  uint32_t comp,
                  uint32_t ox,
                  uint32_t oy,
                  uint32_t tw,
                  uint32_t th)
        : base(base_ptr),
          image_width(width),
          components(comp),
          origin_x(ox),
          origin_y(oy),
          tile_width(tw),
          tile_height(th),
          row_stride(static_cast<size_t>(width) * comp)
    {
    }

    uint8_t sample(int32_t tx, int32_t ty, uint32_t comp) const
    {
      if (tx < 0 || ty < 0)
        return 0;
      if (tx >= static_cast<int32_t>(tile_width) || ty >= static_cast<int32_t>(tile_height))
        return 0;
      const uint32_t gx = origin_x + static_cast<uint32_t>(tx);
      const uint32_t gy = origin_y + static_cast<uint32_t>(ty);
      const size_t offset = static_cast<size_t>(gy) * row_stride + static_cast<size_t>(gx) * components + comp;
      return base[offset];
    }
  };

  // predictor の出力後にカラー相関フィルターを適用し、その結果を
  // component_colors へ格納する。
  void compute_residual_block(const tile_accessor &accessor,
                              tlg::v8::enc::component_colors &out,
                              tlg::v8::enc::predictor_fn predictor,
                              uint32_t components,
                              uint32_t block_x,
                              uint32_t block_y,
                              uint32_t block_w,
                              uint32_t block_h)
  {
    for (auto &component : out.values)
      component.fill(0);

    uint32_t index = 0;
    for (uint32_t by = 0; by < block_h; ++by)
    {
      for (uint32_t bx = 0; bx < block_w; ++bx)
      {
        const int32_t tx = static_cast<int32_t>(block_x + bx);
        const int32_t ty = static_cast<int32_t>(block_y + by);
        for (uint32_t comp = 0; comp < components; ++comp)
        {
          auto sample_from_state = [&](int32_t sx, int32_t sy, uint32_t sc) -> uint8_t
          {
            return accessor.sample(sx, sy, sc);
          };

          const uint8_t a = sample_from_state(tx - 1, ty, comp);
          const uint8_t b = sample_from_state(tx, ty - 1, comp);
          const uint8_t c = sample_from_state(tx - 1, ty - 1, comp);
          const uint8_t d = sample_from_state(tx + 1, ty - 1, comp);
          const uint8_t predicted = predictor(a, b, c, d);
          const uint8_t actual = accessor.sample(tx, ty, comp);
          out.values[comp][index] = static_cast<int16_t>(static_cast<int32_t>(actual) - static_cast<int32_t>(predicted));
        }
        ++index;
      }
    }
  }

  // 残差ブロックをテキストでダンプするユーティリティ。
  void dump_residual_block(FILE *fp,
                           uint32_t tile_origin_x,
                           uint32_t tile_origin_y,
                           uint32_t block_x,
                           uint32_t block_y,
                           uint32_t block_w,
                           uint32_t block_h,
                           uint32_t components,
                           uint32_t predictor_index,
                           uint32_t filter_code,
                           uint32_t entropy_index,
                           uint32_t interleave_index,
                           uint32_t encoded_size,
                           const tlg::v8::enc::component_colors &values,
                           const char *phase_label)
  {
    if (!fp || block_w == 0 || block_h == 0 || values.values.empty() || phase_label == nullptr)
      return;

    const uint32_t value_count = block_w * block_h;
    if (value_count == 0)
      return;

    const uint32_t used_components = std::min<uint32_t>(components,
                                                        static_cast<uint32_t>(values.values.size()));

    std::fprintf(fp,
                 "# tile_origin=(%u,%u) block_origin=(%u,%u) block_size=%ux%u phase=%s\n",
                 tile_origin_x,
                 tile_origin_y,
                 tile_origin_x + block_x,
                 tile_origin_y + block_y,
                 block_w,
                 block_h,
                 phase_label);
    std::fprintf(fp,
                 "# predictor=%u filter=%u entropy=%u interleave=%u encoded_bit_size=%u\n",
                 predictor_index,
                 filter_code,
                 entropy_index,
                 interleave_index,
                 encoded_size);

    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      std::fprintf(fp, "component %u:\n", comp);
      for (uint32_t i = 0; i < value_count; ++i)
      {
        std::fprintf(fp, "%6d", static_cast<int>(values.values[comp][i]));
        if (i + 1 < value_count)
          std::fputc(',', fp);
        if (((i + 1) % block_w) == 0 || i + 1 == value_count)
          std::fputc('\n', fp);
      }
      std::fputc('\n', fp);
    }

    std::fputc('\n', fp);
  }

  void write_residual_block_to_bitmap(PixelBuffer &bitmap,
                                      uint32_t components,
                                      uint32_t tile_origin_x,
                                      uint32_t tile_origin_y,
                                      uint32_t block_x,
                                      uint32_t block_y,
                                      uint32_t block_w,
                                      uint32_t block_h,
                                      const tlg::v8::enc::component_colors &values,
                                      double emphasis)
  {
    if (bitmap.channels == 0 || bitmap.data.empty())
      return;
    const uint32_t used_components = std::min<uint32_t>(components, bitmap.channels);
    const uint32_t stride_width = bitmap.width;
    for (uint32_t local_y = 0; local_y < block_h; ++local_y)
    {
      for (uint32_t local_x = 0; local_x < block_w; ++local_x)
      {
        const uint32_t px = tile_origin_x + block_x + local_x;
        const uint32_t py = tile_origin_y + block_y + local_y;
        if (px >= bitmap.width || py >= bitmap.height)
          continue;
        const size_t pixel_index =
            (static_cast<size_t>(py) * stride_width + px) * bitmap.channels;
        const uint32_t value_index = local_y * block_w + local_x;
        for (uint32_t comp = 0; comp < used_components; ++comp)
        {
          const int16_t residual = values.values[comp][value_index];
          const double scaled = 128.0 + static_cast<double>(residual) * emphasis;
          int pixel_value = static_cast<int>(std::lround(scaled));
          if (pixel_value < 0)
            pixel_value = 0;
          else if (pixel_value > 255)
            pixel_value = 255;
          bitmap.data[pixel_index + comp] = static_cast<uint8_t>(pixel_value);
        }
      }
    }
  }
}

namespace tlg::v8::enc
{
  bool encode_for_tile(detail::bitio::BitWriter &writer,
                       const uint8_t *image_base,
                       uint32_t image_width,
                       uint32_t components,
                       uint32_t origin_x,
                       uint32_t origin_y,
                       uint32_t tile_w,
                       uint32_t tile_h,
                       FILE *dump_fp,
                       TlgOptions::DumpResidualsOrder dump_order,
                       PixelBuffer *residual_bitmap,
                       TlgOptions::DumpResidualsOrder residual_bitmap_order,
                       double residual_bitmap_emphasis,
                       std::string &err)
  {
    if (components == 0 || components > 4)
    {
      err = "tlg8: コンポーネント数が不正です";
      return false;
    }

    const bool dump_after_predictor = dump_fp &&
                                      (dump_order == TlgOptions::DumpResidualsOrder::AfterPredictor);
    const bool dump_after_color = dump_fp &&
                                  (dump_order == TlgOptions::DumpResidualsOrder::AfterColorFilter ||
                                   dump_order == TlgOptions::DumpResidualsOrder::BeforeHilbert);
    const bool dump_after_hilbert = dump_fp &&
                                    (dump_order == TlgOptions::DumpResidualsOrder::AfterHilbert);

    const auto &predictors = predictor_table();
    const auto &entropy_encoders = entropy_encoder_table();
    std::array<std::vector<int16_t>, kGolombRowCount> entropy_values;
    const size_t reserve_per_row = static_cast<size_t>(tile_w) * tile_h;
    for (auto &values : entropy_values)
      values.reserve(reserve_per_row);
    // 将来的にはここで並び替え候補も列挙し、
    // predictor × filter × reorder × entropy の全組み合わせを評価する。
    // 現状は reorder をヒルベルト固定とし、predictor × filter × entropy の
    // 組み合わせを探索している。
    tile_accessor accessor(image_base, image_width, components, origin_x, origin_y, tile_w, tile_h);
    const uint32_t block_cols = (tile_w + kBlockSize - 1) / kBlockSize;
    const uint32_t block_rows = (tile_h + kBlockSize - 1) / kBlockSize;
    std::vector<block_choice> block_choices;
    block_choices.reserve(static_cast<size_t>(block_cols) * block_rows);
    golomb_histogram histograms{};
    std::array<uint64_t, kGolombRowCount> sample_counts{};
    constexpr uint32_t kRowSmoothingThreshold = 256;
    const uint32_t filter_count = (components >= 3) ? static_cast<uint32_t>(kColorFilterCodeCount) : 1u;
    auto estimate_total_bits = [&](const std::array<std::vector<int16_t>, kGolombRowCount> &values) -> uint64_t
    {
      uint64_t total_bits = 0;
      for (uint32_t row = 0; row < kGolombRowCount; ++row)
      {
        const auto &row_values = values[row];
        if (row_values.empty())
          continue;
        if (row_values.size() > std::numeric_limits<uint32_t>::max())
          return std::numeric_limits<uint64_t>::max();
        const uint32_t count = static_cast<uint32_t>(row_values.size());
        const auto kind = golomb_row_kind(row);
        const uint32_t component = golomb_row_component(row);
        total_bits += estimate_row_bits(kind, component, row_values.data(), count);
      }
      return total_bits;
    };
    auto compute_energy = [](const component_colors &colors, uint32_t comp_count, uint32_t value_count)
    {
      double energy = 0.0;
      for (uint32_t comp = 0; comp < comp_count; ++comp)
      {
        const auto &channel = colors.values[comp];
        for (uint32_t index = 0; index < value_count; ++index)
        {
          const double value = static_cast<double>(channel[index]);
          energy += value * value;
        }
      }
      return energy;
    };

    for (uint32_t block_y = 0; block_y < tile_h; block_y += kBlockSize)
    {
      const uint32_t block_h = std::min(kBlockSize, tile_h - block_y);
      for (uint32_t block_x = 0; block_x < tile_w; block_x += kBlockSize)
      {
        const uint32_t block_w = std::min(kBlockSize, tile_w - block_x);
        const uint32_t value_count = block_w * block_h;

        component_colors best_after_interleave{};
        component_colors best_after_hilbert{};
        component_colors best_after_color{};
        component_colors best_after_predictor{};
        uint32_t best_predictor = 0;
        uint32_t best_filter = 0;
        uint32_t best_entropy = 0;
        uint32_t best_interleave = 0;
        uint64_t best_bits = std::numeric_limits<uint64_t>::max();

        component_colors candidate{};
        double best_residual_energy = std::numeric_limits<double>::infinity();
        double best_filtered_energy = std::numeric_limits<double>::infinity();
        for (uint32_t predictor_order_index = 0; predictor_order_index < kNumPredictors; ++predictor_order_index)
        {
          const uint32_t predictor_index = kPredictorTrialOrder[predictor_order_index];
          compute_residual_block(accessor,
                                 candidate,
                                 predictors[predictor_index],
                                 components,
                                 block_x,
                                 block_y,
                                 block_w,
                                 block_h);
          const double residual_energy = compute_energy(candidate, components, value_count);
          if (best_residual_energy < std::numeric_limits<double>::infinity() &&
              residual_energy > best_residual_energy * kEarlyExitGiveUpRate)
          {
            // 予測誤差の自乗和が閾値を超えた場合は、この predictor を早期に諦める。
            continue;
          }
          if (residual_energy < best_residual_energy)
            best_residual_energy = residual_energy;
          for (uint32_t filter_order_index = 0; filter_order_index < filter_count; ++filter_order_index)
          {
            const uint32_t filter_code =
                (components >= 3) ? kColorFilterTrialOrder[filter_order_index] : filter_order_index;
            component_colors filtered = candidate;
            if (components >= 3)
              apply_color_filter(static_cast<int>(filter_code), filtered, components, value_count);

            component_colors filtered_before_hilbert = filtered;
            const double filtered_energy = compute_energy(filtered, components, value_count);
            if (best_filtered_energy < std::numeric_limits<double>::infinity() &&
                filtered_energy > best_filtered_energy * kEarlyExitGiveUpRate)
            {
              // フィルター適用後の誤差エネルギーが大きすぎる候補は以降の処理へ進めない。
              continue;
            }
            if (filtered_energy < best_filtered_energy)
              best_filtered_energy = filtered_energy;

            component_colors reordered = filtered;
            reorder_to_hilbert(reordered, components, block_w, block_h);

            for (uint32_t interleave_index = 0; interleave_index < kNumInterleaveFilter; ++interleave_index)
            {
              component_colors interleaved = reordered;
              apply_interleave_filter(static_cast<InterleaveFilter>(interleave_index),
                                      interleaved,
                                      components,
                                      value_count);

              for (uint32_t entropy_index = 0; entropy_index < kNumEntropyEncoders; ++entropy_index)
              {
                const bool uses_interleave_candidate =
                    (interleave_index == static_cast<uint32_t>(InterleaveFilter::Interleave));
                const uint64_t estimated_bits = entropy_encoders[entropy_index].estimate_bits(interleaved,
                                                                                              components,
                                                                                              value_count,
                                                                                              uses_interleave_candidate);
                if (estimated_bits < best_bits)
                {
                  best_bits = estimated_bits;
                  best_predictor = predictor_index;
                  best_filter = filter_code;
                  best_entropy = entropy_index;
                  best_interleave = interleave_index;
                  best_after_interleave = interleaved;
                  best_after_hilbert = reordered;
                  best_after_color = filtered_before_hilbert;
                  best_after_predictor = candidate;
                }
              }
            }
          }
        }

        // 最小の推定ビット長を与えた組み合わせを採用する。
        // reorder は固定だが、filter やエントロピー符号化方式と同じ基準で
        // 比較する設計とし、将来的に並び替え候補を増やしても流用できるよう
        // にしている。
        block_choices.push_back(block_choice{best_predictor, best_filter, best_entropy, best_interleave});

        if (dump_fp)
        {
          if (dump_after_predictor)
            dump_residual_block(dump_fp,
                                origin_x,
                                origin_y,
                                block_x,
                                block_y,
                                block_w,
                                block_h,
                                components,
                                best_predictor,
                                best_filter,
                                best_entropy,
                                best_interleave,
                                best_bits,
                                best_after_predictor,
                                "after_predictor");
          if (dump_after_color)
            dump_residual_block(dump_fp,
                                origin_x,
                                origin_y,
                                block_x,
                                block_y,
                                block_w,
                                block_h,
                                components,
                                best_predictor,
                                best_filter,
                                best_entropy,
                                best_interleave,
                                best_bits,
                                best_after_color,
                                "after_color");
          if (dump_after_hilbert)
            dump_residual_block(dump_fp,
                                origin_x,
                                origin_y,
                                block_x,
                                block_y,
                                block_w,
                                block_h,
                                components,
                                best_predictor,
                                best_filter,
                                best_entropy,
                                best_interleave,
                                best_bits,
                                best_after_hilbert,
                                "after_hilbert");
        }

        if (residual_bitmap)
        {
          const component_colors *source = nullptr;
          component_colors reordered_values{};
          if (residual_bitmap_order == TlgOptions::DumpResidualsOrder::AfterPredictor)
          {
            source = &best_after_predictor;
          }
          else if (residual_bitmap_order == TlgOptions::DumpResidualsOrder::AfterColorFilter ||
                   residual_bitmap_order == TlgOptions::DumpResidualsOrder::BeforeHilbert)
          {
            source = &best_after_color;
          }
          else if (residual_bitmap_order == TlgOptions::DumpResidualsOrder::AfterHilbert)
          {
            reordered_values = best_after_hilbert;
            reorder_from_hilbert(reordered_values, components, block_w, block_h);
            source = &reordered_values;
          }

          if (source)
          {
            write_residual_block_to_bitmap(*residual_bitmap,
                                           components,
                                           origin_x,
                                           origin_y,
                                           block_x,
                                           block_y,
                                           block_w,
                                           block_h,
                                           *source,
                                           residual_bitmap_emphasis);
          }
        }

        const auto kind = entropy_encoders[best_entropy].kind;
        for (uint32_t comp = 0; comp < components; ++comp)
        {
          const int row = golomb_row_index(kind, comp);
          if (row < 0 || row >= static_cast<int>(kGolombRowCount))
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          uint64_t processed = 0;
          if (kind == GolombCodingKind::Plain)
            processed = accumulate_plain_histogram(histograms[static_cast<std::size_t>(row)],
                                                   best_after_interleave.values[comp].data(),
                                                   value_count);
          else
            processed = accumulate_run_length_histogram(histograms[static_cast<std::size_t>(row)],
                                                        best_after_interleave.values[comp].data(),
                                                        value_count);
          sample_counts[static_cast<std::size_t>(row)] += processed;
          auto &row_values = entropy_values[static_cast<std::size_t>(row)];
          row_values.insert(row_values.end(),
                            best_after_interleave.values[comp].begin(),
                            best_after_interleave.values[comp].begin() + value_count);
        }
      }
    }
    // ここからは、収集した残差統計をもとにゴロムテーブルを動的にリビルドする。
    // 具体的な手順は以下の通りである。
    //   1. 既存テーブルを基準とした推定ビット数を記録する。
    //   2. 行ごとのヒストグラムを `normalize_histogram_row_local` で正規化し、候補行を作る。
    //      サンプル数が少ない行は既定テーブルと線形補間したヒストグラムで安定化する。
    //   3. 行単位で改善量と 9×11bit の書き出しコストを比較し、改善が見込める行だけを更新する。
    //   4. 更新行が存在する場合はマスク付きで差分テーブルを送信し、総コストが改善するか確認する。
    const golomb_table_counts previous_table = current_golomb_table();
    golomb_table_counts candidate = previous_table;
    const uint64_t baseline_bits = estimate_total_bits(entropy_values);
    std::array<uint64_t, kGolombRowCount> baseline_row_bits{};
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      const auto kind = golomb_row_kind(row);
      const uint32_t component = golomb_row_component(row);
      const auto &values = entropy_values[row];
      if (!values.empty())
        baseline_row_bits[row] = estimate_row_bits(kind, component, values.data(), static_cast<uint32_t>(values.size()));
    }
    bool table_changed = false;
    std::array<bool, kGolombRowCount> row_changed{};
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      if (sample_counts[row] == 0)
        continue;
      auto blended_hist = histograms[row];
      if (sample_counts[row] < kRowSmoothingThreshold)
      {
        const uint64_t smoothing = static_cast<uint64_t>(kRowSmoothingThreshold - sample_counts[row]);
        for (uint32_t col = 0; col < kGolombColumnCount; ++col)
          blended_hist[col] += static_cast<uint64_t>(previous_table[row][col]) * smoothing;
      }
      GolombRow new_row{};
      if (!normalize_histogram_row_local(blended_hist, new_row))
        continue;
      if (new_row == candidate[row])
        continue;
      golomb_table_counts trial = candidate;
      trial[row] = new_row;
      apply_golomb_table(trial);
      const auto kind = golomb_row_kind(row);
      const uint32_t component = golomb_row_component(row);
      const auto &values = entropy_values[row];
      const uint64_t candidate_bits = values.empty()
                                          ? 0
                                          : estimate_row_bits(kind, component, values.data(), static_cast<uint32_t>(values.size()));
      const uint64_t row_overhead_bits = static_cast<uint64_t>(kGolombColumnCount) * 11u;
      if (candidate_bits + row_overhead_bits < baseline_row_bits[row])
      {
        candidate = trial;
        baseline_row_bits[row] = candidate_bits;
        row_changed[row] = true;
        table_changed = true;
      }
      else
      {
        apply_golomb_table(candidate);
      }
    }
    apply_golomb_table(candidate);
    uint32_t changed_rows = 0;
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      if (row_changed[row])
        ++changed_rows;
    }
    bool keep_dynamic_table = false;
    if (table_changed)
    {
      const uint64_t dynamic_bits = estimate_total_bits(entropy_values);
      const uint64_t table_overhead_bits = static_cast<uint64_t>(changed_rows) * kGolombColumnCount * 11u +
                                           static_cast<uint64_t>(kGolombRowCount);
      const uint64_t total_with_table = (dynamic_bits == std::numeric_limits<uint64_t>::max())
                                            ? std::numeric_limits<uint64_t>::max()
                                            : dynamic_bits + table_overhead_bits;
      if (total_with_table < baseline_bits)
        keep_dynamic_table = true;
    }
    if (!keep_dynamic_table)
    {
      apply_golomb_table(previous_table);
      table_changed = false;
      row_changed.fill(false);
      candidate = previous_table;
    }

    writer.put_upto8(table_changed ? 1u : 0u, 1);
    if (table_changed)
    {
      uint32_t mask = 0;
      for (uint32_t row = 0; row < kGolombRowCount; ++row)
      {
        if (row_changed[row])
          mask |= (1u << row);
      }
      writer.put_upto8(mask, kGolombRowCount);
      const auto &table = current_golomb_table();
      for (uint32_t row = 0; row < kGolombRowCount; ++row)
      {
        if (!row_changed[row])
          continue;
        uint32_t sum = 0;
        for (uint32_t col = 0; col < kGolombColumnCount; ++col)
        {
          const uint32_t value = static_cast<uint32_t>(table[row][col]);
          writer.put(value, 11);
          sum += value;
        }
        if (sum != kGolombRowSum)
        {
          err = "tlg8: ゴロムテーブル行の合計が不正です";
          return false;
        }
      }
    }

    const uint32_t predictor_bits = tlg::detail::bit_width(kNumPredictors);
    const uint32_t filter_bits = tlg::detail::bit_width(filter_count);
    const uint32_t entropy_bits = tlg::detail::bit_width(kNumEntropyEncoders);
    const uint32_t interleave_bits = tlg::detail::bit_width(kNumInterleaveFilter);

    std::vector<uint32_t> predictor_stream;
    std::vector<uint32_t> filter_stream;
    std::vector<uint32_t> entropy_stream;
    std::vector<uint32_t> interleave_stream;
    predictor_stream.reserve(block_choices.size());
    filter_stream.reserve(block_choices.size());
    entropy_stream.reserve(block_choices.size());
    interleave_stream.reserve(block_choices.size());
    for (const auto &choice : block_choices)
    {
      predictor_stream.push_back(choice.predictor);
      filter_stream.push_back(choice.filter);
      entropy_stream.push_back(choice.entropy);
      interleave_stream.push_back(choice.interleave);
    }

    const BlockChoiceEncoding predictor_mode = select_best_sequence_mode(predictor_stream, predictor_bits);
    const BlockChoiceEncoding filter_mode = select_best_sequence_mode(filter_stream, filter_bits);
    const BlockChoiceEncoding entropy_mode = select_best_sequence_mode(entropy_stream, entropy_bits);
    const BlockChoiceEncoding interleave_mode = select_best_sequence_mode(interleave_stream, interleave_bits);

    writer.put_upto8(static_cast<uint32_t>(predictor_mode), 2);
    writer.put_upto8(static_cast<uint32_t>(filter_mode), 2);
    writer.put_upto8(static_cast<uint32_t>(entropy_mode), 2);
    writer.put_upto8(static_cast<uint32_t>(interleave_mode), 2);

    if (!block_choices.empty())
    {
      encode_value_sequence(writer, predictor_mode, predictor_stream, predictor_bits);
      encode_value_sequence(writer, filter_mode, filter_stream, filter_bits);
      encode_value_sequence(writer, entropy_mode, entropy_stream, entropy_bits);
      encode_value_sequence(writer, interleave_mode, interleave_stream, interleave_bits);
    }

    writer.align_to_byte_zero();
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      const auto kind = golomb_row_kind(row);
      const uint32_t component = golomb_row_component(row);
      const auto &values = entropy_values[row];
      if (values.size() > std::numeric_limits<uint32_t>::max())
      {
        err = "tlg8: エントロピー値数が大きすぎます";
        return false;
      }
      const uint32_t count = static_cast<uint32_t>(values.size());
      if (count == 0)
        continue;
      if (!encode_values(writer,
                         kind,
                         component,
                         values.data(),
                         count,
                         err))
      {
        err = "tlg8: エントロピー書き込みに失敗しました";
        return false;
      }
    }
    return true;
  }

}
