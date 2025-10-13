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
  using tlg::v8::enc::GolombCodingKind;
  using tlg::v8::enc::golomb_histogram;
  using tlg::v8::enc::kColorFilterCodeCount;
  using tlg::v8::enc::kGolombColumnCount;
  using tlg::v8::enc::kGolombRowCount;
  using tlg::v8::enc::kGolombRowSum;
  using tlg::v8::enc::kNumPredictors;
  using tlg::v8::enc::component_colors;
  using tlg::v8::enc::adaptation::mix_a_m;
  using tlg::v8::enc::adaptation::reduce_a;
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

  // 値の配列を走査し、連続する同値のラン情報を列挙するユーティリティ。
  template <typename Callback>
  void for_each_run(const std::vector<uint32_t> &values, const Callback &callback)
  {
    std::size_t index = 0;
    while (index < values.size())
    {
      const uint32_t current = values[index];
      std::size_t run_end = index + 1;
      while (run_end < values.size() && values[run_end] == current)
        ++run_end;
      callback(current, static_cast<uint32_t>(run_end - index));
      index = run_end;
    }
  }

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
      for_each_run(values, [&](uint32_t value, uint32_t run_length) {
        static_cast<void>(value);
        bits += value_bits;
        bits += varuint_bits(run_length - 1);
      });
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
      for_each_run(values, [&](uint32_t current, uint32_t run_length) {
        writer.put_upto8(current, value_bits);
        put_varuint(writer, run_length - 1);
      });
      break;
    }

    default:
      break;
    }
  }

  // タイル全体を 8x8 ブロックへ分割し、カラー相関フィルター→予測→
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

  // 8x8 ブロック周辺の画素を一度だけ読み出し、予測器評価時に再利用するための
  // ワークスペース。
  struct block_workspace
  {
    uint32_t block_w = 0;
    uint32_t block_h = 0;
    uint32_t stride = 0;
    std::array<std::vector<uint8_t>, 4> padded{};

    void prepare(const tile_accessor &accessor,
                 uint32_t block_x,
                 uint32_t block_y,
                 uint32_t width,
                 uint32_t height,
                 uint32_t components)
    {
      block_w = width;
      block_h = height;
      stride = block_w + 2u;
      const uint32_t padded_height = block_h + 1u;
      const size_t required = static_cast<size_t>(stride) * padded_height;
      for (auto &plane : padded)
        plane.assign(required, 0);

      for (uint32_t comp = 0; comp < components; ++comp)
      {
        auto &plane = padded[comp];
        for (int32_t local_y = -1; local_y < static_cast<int32_t>(block_h); ++local_y)
        {
          for (int32_t local_x = -1; local_x <= static_cast<int32_t>(block_w); ++local_x)
          {
            const uint8_t sample = accessor.sample(static_cast<int32_t>(block_x) + local_x,
                                                   static_cast<int32_t>(block_y) + local_y,
                                                   comp);
            const size_t index = static_cast<size_t>(local_y + 1) * stride + static_cast<size_t>(local_x + 1);
            plane[index] = sample;
          }
        }
      }
    }
  };

  struct filtered_workspace
  {
    uint32_t stride = 0;
    std::array<std::vector<int16_t>, 4> padded{};
  };

  void prepare_filtered_workspace(const block_workspace &workspace,
                                  filtered_workspace &filtered,
                                  tlg::v8::enc::component_colors &filtered_block,
                                  uint32_t components,
                                  uint32_t block_w,
                                  uint32_t block_h,
                                  int filter_code)
  {
    filtered.stride = workspace.stride;
    const uint32_t padded_height = block_h + 1u;
    const size_t total = static_cast<size_t>(filtered.stride) * padded_height;
    for (auto &plane : filtered.padded)
      plane.assign(total, 0);
    for (auto &component : filtered_block.values)
      component.fill(0);

    component_colors pixel{};
    for (int32_t local_y = -1; local_y < static_cast<int32_t>(block_h); ++local_y)
    {
      for (int32_t local_x = -1; local_x <= static_cast<int32_t>(block_w); ++local_x)
      {
        const size_t index = static_cast<size_t>(local_y + 1) * filtered.stride + static_cast<size_t>(local_x + 1);
        for (uint32_t comp = 0; comp < pixel.values.size(); ++comp)
          pixel.values[comp][0] = 0;
        for (uint32_t comp = 0; comp < components; ++comp)
        {
          const auto &plane = workspace.padded[comp];
          const int value = (index < plane.size()) ? static_cast<int>(plane[index]) : 0;
          pixel.values[comp][0] = static_cast<int16_t>(value);
        }
        if (components >= 3)
          apply_color_filter(filter_code, pixel, components, 1);
        for (uint32_t comp = 0; comp < components; ++comp)
          filtered.padded[comp][index] = pixel.values[comp][0];
        for (uint32_t comp = components; comp < filtered.padded.size(); ++comp)
          filtered.padded[comp][index] = 0;

        if (local_x >= 0 && local_y >= 0)
        {
          const size_t block_index = static_cast<size_t>(local_y) * block_w + static_cast<size_t>(local_x);
          for (uint32_t comp = 0; comp < components; ++comp)
            filtered_block.values[comp][block_index] = pixel.values[comp][0];
          for (uint32_t comp = components; comp < filtered_block.values.size(); ++comp)
            filtered_block.values[comp][block_index] = 0;
        }
      }
    }
  }

  // カラー相関フィルター適用後の信号に predictor を適用し、予測誤差を
  // component_colors へ格納する。
  void compute_residual_block(const filtered_workspace &workspace,
                              tlg::v8::enc::component_colors &out,
                              tlg::v8::enc::predictor_fn predictor,
                              uint32_t components,
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
        for (uint32_t comp = 0; comp < components; ++comp)
        {
          const auto &plane = workspace.padded[comp];
          const size_t base = static_cast<size_t>(by + 1) * workspace.stride + static_cast<size_t>(bx + 1);
          const int16_t a = plane[base - 1];
          const int16_t b = plane[base - workspace.stride];
          const int16_t c = plane[base - workspace.stride - 1];
          const int16_t d = plane[base - workspace.stride + 1];
          const int16_t actual = plane[base];
          const int16_t predicted = predictor(a, b, c, d);
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
    const uint32_t filter_count = (components >= 3) ? static_cast<uint32_t>(kColorFilterCodeCount) : 1u;
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

    block_workspace workspace;
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
        workspace.prepare(accessor, block_x, block_y, block_w, block_h, components);
        double best_residual_energy = std::numeric_limits<double>::infinity();
        double best_filtered_energy = std::numeric_limits<double>::infinity();
        for (uint32_t filter_order_index = 0; filter_order_index < filter_count; ++filter_order_index)
        {
          const uint32_t filter_code =
              (components >= 3) ? kColorFilterTrialOrder[filter_order_index] : filter_order_index;
          filtered_workspace filtered_ws{};
          component_colors filtered_block{};
          prepare_filtered_workspace(workspace,
                                     filtered_ws,
                                     filtered_block,
                                     components,
                                     block_w,
                                     block_h,
                                     static_cast<int>(filter_code));

          const double filtered_energy = compute_energy(filtered_block, components, value_count);
          if (best_filtered_energy < std::numeric_limits<double>::infinity() &&
              filtered_energy > best_filtered_energy * kEarlyExitGiveUpRate)
          {
            // フィルター適用後の信号エネルギーが大きすぎる候補は以降の処理へ進めない。
            continue;
          }
          if (filtered_energy < best_filtered_energy)
            best_filtered_energy = filtered_energy;

          for (uint32_t predictor_order_index = 0; predictor_order_index < kNumPredictors; ++predictor_order_index)
          {
            const uint32_t predictor_index = kPredictorTrialOrder[predictor_order_index];
            compute_residual_block(filtered_ws,
                                   candidate,
                                   predictors[predictor_index],
                                   components,
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

            component_colors reordered = candidate;
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
                  best_after_color = filtered_block;
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
        const bool uses_interleave_filter =
            (best_interleave == static_cast<uint32_t>(InterleaveFilter::Interleave));
        const uint32_t used_components =
            std::min<uint32_t>(components, static_cast<uint32_t>(best_after_interleave.values.size()));
        if (uses_interleave_filter && used_components > 0)
        {
          const int row = golomb_row_index(kind, kInterleavedComponentIndex);
          if (row < 0 || row >= static_cast<int>(kGolombRowCount))
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          const std::size_t combined_count = static_cast<std::size_t>(value_count) * used_components;
          std::array<int16_t, kMaxBlockPixels * 4> combined{};
          std::size_t combined_offset = 0;
          for (uint32_t comp = 0; comp < used_components; ++comp)
          {
            std::copy_n(best_after_interleave.values[comp].begin(),
                        value_count,
                        combined.begin() + combined_offset);
            combined_offset += value_count;
          }
          auto &row_values = entropy_values[static_cast<std::size_t>(row)];
          row_values.insert(row_values.end(), combined.begin(), combined.begin() + combined_count);
        }
        else
        {
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            const int row = golomb_row_index(kind, comp);
            if (row < 0 || row >= static_cast<int>(kGolombRowCount))
            {
              err = "tlg8: 不正なゴロム行です";
              return false;
            }
            auto &row_values = entropy_values[static_cast<std::size_t>(row)];
            row_values.insert(row_values.end(),
                              best_after_interleave.values[comp].begin(),
                              best_after_interleave.values[comp].begin() + value_count);
          }
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
