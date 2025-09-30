#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_color_filter.h"
#include "tlg8_entropy.h"
#include "tlg8_reorder.h"
#include "tlg8_predictors.h"
#include "tlg_io_common.h"

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
                 "# predictor=%u filter=%u entropy=%u encoded_bit_size=%u\n",
                 predictor_index,
                 filter_code,
                 entropy_index,
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
                       bool dump_before_hilbert,
                       bool dump_after_hilbert,
                       std::string &err)
  {
    if (components == 0 || components > 4)
    {
      err = "tlg8: コンポーネント数が不正です";
      return false;
    }

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

        component_colors best_block{};
        component_colors best_filtered{};
        uint32_t best_predictor = 0;
        uint32_t best_filter = 0;
        uint32_t best_entropy = 0;
        uint64_t best_bits = std::numeric_limits<uint64_t>::max();

        component_colors candidate{};
        const uint32_t filter_count = (components >= 3) ? static_cast<uint32_t>(kColorFilterCodeCount) : 1u;
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

            for (uint32_t entropy_index = 0; entropy_index < kNumEntropyEncoders; ++entropy_index)
            {
              const uint64_t estimated_bits =
                  entropy_encoders[entropy_index].estimate_bits(reordered, components, value_count);
              if (estimated_bits < best_bits)
              {
                best_bits = estimated_bits;
                best_predictor = predictor_index;
                best_filter = filter_code;
                best_entropy = entropy_index;
                best_block = reordered;
                best_filtered = filtered_before_hilbert;
              }
            }
          }
        }

        // 最小の推定ビット長を与えた組み合わせを採用する。
        // reorder は固定だが、filter やエントロピー符号化方式と同じ基準で
        // 比較する設計とし、将来的に並び替え候補を増やしても流用できるよう
        // にしている。
        writer.put_upto8(best_predictor, tlg::detail::bit_width(kNumPredictors));
        writer.put_upto8(best_filter, tlg::detail::bit_width(filter_count));
        writer.put_upto8(best_entropy, tlg::detail::bit_width(kNumEntropyEncoders));

        if (dump_fp)
        {
          if (dump_before_hilbert)
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
                                best_bits,
                                best_filtered,
                                "before_hilbert");
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
                                best_bits,
                                best_block,
                                "after_hilbert");
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
          auto &row_values = entropy_values[static_cast<std::size_t>(row)];
          row_values.insert(row_values.end(),
                            best_block.values[comp].begin(),
                            best_block.values[comp].begin() + value_count);
        }
      }
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
