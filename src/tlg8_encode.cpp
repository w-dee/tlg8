#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_color_filter.h"
#include "tlg8_entropy.h"
#include "tlg8_interleave.h"
#include "tlg8_reorder.h"
#include "tlg8_predictors.h"
#include "tlg_io_common.h"
#include "image_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{
  inline constexpr double EARLY_EXIT_GIVE_UP_RATE = 1.4;
  using tlg::v8::enc::kGolombColumnCount;
  using tlg::v8::enc::kGolombRowCount;
  using tlg::v8::enc::kGolombRowSum;

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

  using bucket_cost_matrix = std::array<std::array<uint64_t, kGolombRowSum>, kGolombColumnCount>;
  using GolombRow = std::array<uint16_t, kGolombColumnCount>;

  // ブロック毎に選ばれたメタデータを一時保持する構造体。
  struct block_choice
  {
    uint32_t predictor = 0;
    uint32_t filter = 0;
    uint32_t entropy = 0;
    uint32_t interleave = 0;
  };

  uint64_t accumulate_plain_bucket_cost(bucket_cost_matrix &matrix,
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
      const uint32_t bucket = std::min<uint32_t>(static_cast<uint32_t>(reduce_a(a)), kGolombRowSum - 1);
      for (uint32_t k = 0; k < kGolombColumnCount; ++k)
      {
        const uint32_t q = (k > 0) ? (m >> k) : m;
        matrix[k][bucket] += static_cast<uint64_t>(q + 1u + k);
      }
      a = mix_a_m(a, static_cast<int>(m));
      ++processed;
    }
    return processed;
  }

  uint64_t accumulate_run_length_bucket_cost(bucket_cost_matrix &matrix,
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
        const uint32_t bucket = std::min<uint32_t>(static_cast<uint32_t>(reduce_a(a)), kGolombRowSum - 1);
        for (uint32_t k = 0; k < kGolombColumnCount; ++k)
        {
          const uint32_t q = (k > 0) ? (m >> k) : m;
          matrix[k][bucket] += static_cast<uint64_t>(q + 1u + k);
        }
        a = mix_a_m(a, static_cast<int>(m));
        ++index;
        ++processed;
      }
    }
    return processed;
  }

  GolombRow optimize_row_from_cost(const bucket_cost_matrix &matrix)
  {
    constexpr uint64_t INF = std::numeric_limits<uint64_t>::max() / 4;
    std::array<std::array<uint64_t, kGolombRowSum + 1>, kGolombColumnCount> prefix{};
    for (uint32_t k = 0; k < kGolombColumnCount; ++k)
    {
      prefix[k][0] = 0;
      for (uint32_t i = 0; i < kGolombRowSum; ++i)
        prefix[k][i + 1] = prefix[k][i] + matrix[k][i];
    }

    std::array<std::array<uint64_t, kGolombRowSum + 1>, kGolombColumnCount> dp{};
    std::array<std::array<uint32_t, kGolombRowSum + 1>, kGolombColumnCount> prev{};
    for (uint32_t col = 0; col < kGolombColumnCount; ++col)
    {
      for (uint32_t count = 0; count <= kGolombRowSum; ++count)
      {
        dp[col][count] = INF;
        prev[col][count] = 0;
      }
    }

    for (uint32_t count = 0; count <= kGolombRowSum; ++count)
    {
      dp[0][count] = prefix[0][count];
      prev[0][count] = 0;
    }

    for (uint32_t col = 1; col < kGolombColumnCount; ++col)
    {
      for (uint32_t count = 0; count <= kGolombRowSum; ++count)
      {
        for (uint32_t prev_count = 0; prev_count <= count; ++prev_count)
        {
          const uint64_t prev_cost = dp[col - 1][prev_count];
          if (prev_cost == INF)
            continue;
          const uint64_t segment_cost = prefix[col][count] - prefix[col][prev_count];
          const uint64_t cost = prev_cost + segment_cost;
          if (cost < dp[col][count])
          {
            dp[col][count] = cost;
            prev[col][count] = prev_count;
          }
        }
      }
    }

    GolombRow result{};
    uint32_t end = kGolombRowSum;
    for (int col = static_cast<int>(kGolombColumnCount) - 1; col >= 0; --col)
    {
      const uint32_t start = prev[static_cast<std::size_t>(col)][end];
      result[static_cast<std::size_t>(col)] = static_cast<uint16_t>(end - start);
      end = start;
    }
    return result;
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
    std::array<bucket_cost_matrix, kGolombRowCount> bucket_costs{};
    std::array<uint64_t, kGolombRowCount> sample_counts{};
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
        for (uint32_t predictor_index = 0; predictor_index < kNumPredictors; ++predictor_index)
        {
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
              residual_energy > best_residual_energy * EARLY_EXIT_GIVE_UP_RATE)
          {
            // 予測誤差の自乗和が閾値を超えた場合は、この predictor を早期に諦める。
            continue;
          }
          if (residual_energy < best_residual_energy)
            best_residual_energy = residual_energy;
          for (uint32_t filter_code = 0; filter_code < filter_count; ++filter_code)
          {
            component_colors filtered = candidate;
            if (components >= 3)
              apply_color_filter(static_cast<int>(filter_code), filtered, components, value_count);

            component_colors filtered_before_hilbert = filtered;
            const double filtered_energy = compute_energy(filtered, components, value_count);
            if (best_filtered_energy < std::numeric_limits<double>::infinity() &&
                filtered_energy > best_filtered_energy * EARLY_EXIT_GIVE_UP_RATE)
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

        const auto kind = entropy_encoders[best_entropy].kind;
        const bool uses_interleave =
            (best_interleave == static_cast<uint32_t>(InterleaveFilter::Interleave));
        if (uses_interleave)
        {
          // インターリーブ後の値は振幅が大きくなることが多く、
          // 絶対値が大きな値を扱う 0 番/3 番行へ集約することで
          // 適切なパラメーターで Golomb-Rice 符号化できる。
          const uint32_t target_row = (kind == GolombCodingKind::Plain) ? 0u : 3u;
          if (target_row >= kGolombRowCount)
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          uint64_t processed = 0;
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            if (kind == GolombCodingKind::Plain)
              processed += accumulate_plain_bucket_cost(bucket_costs[target_row],
                                                        best_after_interleave.values[comp].data(),
                                                        value_count);
            else
              processed += accumulate_run_length_bucket_cost(bucket_costs[target_row],
                                                             best_after_interleave.values[comp].data(),
                                                             value_count);
          }
          sample_counts[target_row] += processed;
          auto &row_values = entropy_values[static_cast<std::size_t>(target_row)];
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            // インターリーブ時は全コンポーネント分の値をまとめて同一行へ投入する。
            row_values.insert(row_values.end(),
                              best_after_interleave.values[comp].begin(),
                              best_after_interleave.values[comp].begin() + value_count);
          }
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
            uint64_t processed = 0;
            if (kind == GolombCodingKind::Plain)
              processed = accumulate_plain_bucket_cost(bucket_costs[static_cast<std::size_t>(row)],
                                                        best_after_interleave.values[comp].data(),
                                                        value_count);
            else
              processed = accumulate_run_length_bucket_cost(bucket_costs[static_cast<std::size_t>(row)],
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
    }
    // ここからは、収集した残差統計をもとにゴロムテーブルを動的にリビルドする。
    // 具体的には以下の手順で実施している。
    //   1. 既存テーブルのまま推定したビット数 (baseline_bits) を基準として記録する。
    //   2. 行ごとに集計したバケットコストから DP 最適化 (`optimize_row_from_cost`) を実行し、
    //      各行の候補テーブルを生成する。サンプルが存在する行のみを更新対象とし、
    //      変更があった場合は一時的に `apply_golomb_table` で適用する。
    //   3. 新テーブルで再度ビット数を見積もり、テーブルを書き出すオーバーヘッド
    //      (1 スロット 11bit × 行数 × 列数) を加えた総コストを評価する。
    //      ベースラインより悪化した場合は元のテーブルへロールバックする。
    const golomb_table_counts previous_table = current_golomb_table();
    golomb_table_counts candidate = previous_table;
    const uint64_t baseline_bits = estimate_total_bits(entropy_values);
    bool table_changed = false;
    for (uint32_t row = 0; row < kGolombRowCount; ++row)
    {
      if (sample_counts[row] == 0)
        continue;
      const GolombRow optimized = optimize_row_from_cost(bucket_costs[row]);
      if (optimized != candidate[row])
      {
        candidate[row] = optimized;
        table_changed = true;
      }
    }
    if (table_changed)
      apply_golomb_table(candidate);
    if (table_changed)
    {
      const uint64_t dynamic_bits = estimate_total_bits(entropy_values);
      const uint64_t table_overhead_bits = static_cast<uint64_t>(kGolombColumnCount) * kGolombRowCount * 11u;
      const uint64_t total_with_table = (dynamic_bits == std::numeric_limits<uint64_t>::max())
                                            ? std::numeric_limits<uint64_t>::max()
                                            : dynamic_bits + table_overhead_bits;
      if (total_with_table == std::numeric_limits<uint64_t>::max() || total_with_table >= baseline_bits)
      {
        apply_golomb_table(previous_table);
        table_changed = false;
      }
    }

    writer.put_upto8(table_changed ? 1u : 0u, 1);
    if (table_changed)
    {
      const auto &table = current_golomb_table();
      for (uint32_t row = 0; row < kGolombRowCount; ++row)
      {
        for (uint32_t col = 0; col < kGolombColumnCount; ++col)
        {
          writer.put(static_cast<uint32_t>(table[row][col]), 11);
        }
      }
    }

    for (const auto &choice : block_choices)
    {
      writer.put_upto8(choice.predictor, tlg::detail::bit_width(kNumPredictors));
      writer.put_upto8(choice.filter, tlg::detail::bit_width(filter_count));
      writer.put_upto8(choice.entropy, tlg::detail::bit_width(kNumEntropyEncoders));
      writer.put_upto8(choice.interleave, tlg::detail::bit_width(kNumInterleaveFilter));
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
