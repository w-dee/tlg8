#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_color_filter.h"
#include "tlg8_entropy.h"
#include "tlg8_interleave.h"
#include "tlg8_reorder.h"
#include "tlg8_predictors.h"
#include "tlg_io_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tlg::v8
{
  namespace
  {
    uint8_t sample_pixel(const std::vector<uint8_t> &decoded,
                         const std::vector<uint8_t> &decoded_mask,
                         uint32_t image_width,
                         uint32_t components,
                         uint32_t origin_x,
                         uint32_t origin_y,
                         uint32_t tile_w,
                         uint32_t tile_h,
                         int32_t tx,
                         int32_t ty,
                         uint32_t component)
    {
      if (tx < 0 || ty < 0)
        return 0;
      if (tx >= static_cast<int32_t>(tile_w) || ty >= static_cast<int32_t>(tile_h))
        return 0;
      const size_t tile_offset = (static_cast<size_t>(ty) * tile_w + static_cast<size_t>(tx)) * components + component;
      if (tile_offset >= decoded_mask.size())
        return 0;
      if (!decoded_mask[tile_offset])
        return 0;
      const uint32_t gx = origin_x + static_cast<uint32_t>(tx);
      const uint32_t gy = origin_y + static_cast<uint32_t>(ty);
      const size_t index = (static_cast<size_t>(gy) * image_width + gx) * components + component;
      if (index >= decoded.size())
        return 0;
      return decoded[index];
    }
  }

  bool decode_for_tile(detail::bitio::BitReader &reader,
                       uint32_t tile_w,
                       uint32_t tile_h,
                       uint32_t components,
                       uint32_t origin_x,
                       uint32_t origin_y,
                       uint32_t image_width,
                       std::vector<uint8_t> &decoded,
                       std::string &err)
  {
    if (components == 0 || components > 4)
    {
      err = "tlg8: コンポーネント数が不正です";
      return false;
    }

    const auto &predictors = enc::predictor_table();
    const auto &entropy_encoders = enc::entropy_encoder_table();
    const uint32_t filter_count = (components >= 3) ? static_cast<uint32_t>(tlg::v8::enc::kColorFilterCodeCount) : 1u;

    std::vector<uint8_t> decoded_mask(static_cast<size_t>(tile_w) * tile_h * components, 0);

    struct block_decode_state
    {
      uint32_t block_x;
      uint32_t block_w;
      uint32_t predictor_index;
      uint32_t filter_index;
      uint32_t entropy_index;
      uint32_t interleave_index;
      uint32_t value_count;
      enc::component_colors residuals;
    };

    struct block_row_state
    {
      uint32_t block_y;
      uint32_t block_h;
      std::vector<block_decode_state> blocks;
    };

    std::vector<block_row_state> block_rows;
    std::array<std::size_t, enc::kGolombRowCount> row_value_counts{};

    for (uint32_t block_y = 0; block_y < tile_h; block_y += enc::kBlockSize)
    {
      const uint32_t block_h = std::min(enc::kBlockSize, tile_h - block_y);
      block_row_state row_state{};
      row_state.block_y = block_y;
      row_state.block_h = block_h;

      for (uint32_t block_x = 0; block_x < tile_w; block_x += enc::kBlockSize)
      {
        const uint32_t block_w = std::min(enc::kBlockSize, tile_w - block_x);
        const uint32_t predictor_index = reader.get_upto8(tlg::detail::bit_width(tlg::v8::enc::kNumPredictors));
        if (predictor_index >= enc::kNumPredictors)
        {
          err = "tlg8: 不正な予測器インデックスです";
          return false;
        }

        const uint32_t filter_index = reader.get_upto8(tlg::detail::bit_width(filter_count));
        if (filter_index >= enc::kColorFilterCodeCount)
        {
          err = "tlg8: 不正なカラーフィルターインデックスです";
          return false;
        }

        const uint32_t entropy_index = reader.get_upto8(tlg::detail::bit_width(tlg::v8::enc::kNumEntropyEncoders));
        if (entropy_index >= enc::kNumEntropyEncoders)
        {
          err = "tlg8: 不正なエントロピーインデックスです";
          return false;
        }

        const uint32_t interleave_index = reader.get_upto8(tlg::detail::bit_width(tlg::v8::enc::kNumInterleaveFilter));
        if (interleave_index >= enc::kNumInterleaveFilter)
        {
          err = "tlg8: 不正なインターリーブフィルターです";
          return false;
        }

        const uint32_t value_count = block_w * block_h;
        block_decode_state state{};
        state.block_x = block_x;
        state.block_w = block_w;
        state.predictor_index = predictor_index;
        state.filter_index = filter_index;
        state.entropy_index = entropy_index;
        state.interleave_index = interleave_index;
        state.value_count = value_count;
        row_state.blocks.emplace_back(state);

        const auto kind = entropy_encoders[entropy_index].kind;
        const bool uses_interleave =
            (static_cast<enc::InterleaveFilter>(interleave_index) == enc::InterleaveFilter::Interleave);
        if (uses_interleave)
        {
          // インターリーブ後の値は振幅が大きくなるため、
          // エンコード側と同様に 0 番/3 番行へ集約された列から
          // 各コンポーネント分の値を順番に読み戻す。
          const uint32_t target_row = (kind == enc::GolombCodingKind::Plain) ? 0u : 3u;
          if (target_row >= enc::kGolombRowCount)
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          row_value_counts[target_row] += static_cast<std::size_t>(value_count) * components;
        }
        else
        {
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            const int row = enc::golomb_row_index(kind, comp);
            if (row < 0 || row >= static_cast<int>(enc::kGolombRowCount))
            {
              err = "tlg8: 不正なゴロム行です";
              return false;
            }
            row_value_counts[static_cast<std::size_t>(row)] += value_count;
          }
        }
      }

      block_rows.emplace_back(std::move(row_state));
    }

    reader.align_to_byte();

    std::array<std::vector<int16_t>, enc::kGolombRowCount> row_values;
    std::array<std::size_t, enc::kGolombRowCount> row_offsets{};

    for (uint32_t row = 0; row < enc::kGolombRowCount; ++row)
    {
      const std::size_t value_count = row_value_counts[row];
      auto &values = row_values[row];
      values.resize(value_count);
      if (value_count == 0)
        continue;
      if (value_count > std::numeric_limits<uint32_t>::max())
      {
        err = "tlg8: エントロピー値数が大きすぎます";
        return false;
      }
      if (!enc::decode_values(reader,
                              enc::golomb_row_kind(row),
                              enc::golomb_row_component(row),
                              static_cast<uint32_t>(value_count),
                              values.data(),
                              err))
      {
        err = "tlg8: エントロピー復号に失敗しました";
        return false;
      }
    }

    for (auto &row_state : block_rows)
    {
      for (auto &state : row_state.blocks)
      {
        const auto kind = entropy_encoders[state.entropy_index].kind;
        const bool uses_interleave = (static_cast<enc::InterleaveFilter>(state.interleave_index) ==
                                      enc::InterleaveFilter::Interleave);
        if (uses_interleave)
        {
          // エンコード側で 0 番/3 番行へ集約した値列を、
          // コンポーネント単位に分割してブロックへ復元する。
          const uint32_t target_row = (kind == enc::GolombCodingKind::Plain) ? 0u : 3u;
          if (target_row >= enc::kGolombRowCount)
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          auto &values = row_values[target_row];
          auto &offset = row_offsets[target_row];
          const std::size_t required = static_cast<std::size_t>(state.value_count) * components;
          if (offset + required > values.size())
          {
            err = "tlg8: エントロピー列が不足しています";
            return false;
          }
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            const std::size_t begin = offset + static_cast<std::size_t>(comp) * state.value_count;
            std::copy_n(values.data() + begin, state.value_count, state.residuals.values[comp].begin());
          }
          offset += required;
        }
        else
        {
          for (uint32_t comp = 0; comp < components; ++comp)
          {
            const int row = enc::golomb_row_index(kind, comp);
            if (row < 0 || row >= static_cast<int>(enc::kGolombRowCount))
            {
              err = "tlg8: 不正なゴロム行です";
              return false;
            }
            auto &values = row_values[static_cast<std::size_t>(row)];
            auto &offset = row_offsets[static_cast<std::size_t>(row)];
            if (offset + state.value_count > values.size())
            {
              err = "tlg8: エントロピー列が不足しています";
              return false;
            }
            std::copy_n(values.data() + offset, state.value_count, state.residuals.values[comp].begin());
            offset += state.value_count;
          }
        }
        enc::undo_interleave_filter(static_cast<enc::InterleaveFilter>(state.interleave_index),
                                    state.residuals,
                                    components,
                                    state.value_count);
        enc::reorder_from_hilbert(state.residuals, components, state.block_w, row_state.block_h);
        enc::undo_color_filter(state.filter_index, state.residuals, components, state.value_count);
      }
    }

    for (const auto &row_state : block_rows)
    {
      for (uint32_t local_y = 0; local_y < row_state.block_h; ++local_y)
      {
        for (const auto &state : row_state.blocks)
        {
          const auto predictor = predictors[state.predictor_index];
          for (uint32_t bx = 0; bx < state.block_w; ++bx)
          {
            const int32_t tx = static_cast<int32_t>(state.block_x + bx);
            const int32_t ty = static_cast<int32_t>(row_state.block_y + local_y);
            const uint32_t value_index = local_y * state.block_w + bx;
            for (uint32_t comp = 0; comp < components; ++comp)
            {
              const uint8_t a = sample_pixel(decoded, decoded_mask, image_width, components, origin_x, origin_y, tile_w, tile_h,
                                             tx - 1, ty, comp);
              const uint8_t b = sample_pixel(decoded, decoded_mask, image_width, components, origin_x, origin_y, tile_w, tile_h,
                                             tx, ty - 1, comp);
              const uint8_t c = sample_pixel(decoded, decoded_mask, image_width, components, origin_x, origin_y, tile_w, tile_h,
                                             tx - 1, ty - 1, comp);
              const uint8_t d = sample_pixel(decoded, decoded_mask, image_width, components, origin_x, origin_y, tile_w, tile_h,
                                             tx + 1, ty - 1, comp);
              const uint8_t predicted = predictor(a, b, c, d);
              const int16_t residual = state.residuals.values[comp][value_index];
              int32_t value = static_cast<int32_t>(predicted) + static_cast<int32_t>(residual);
              value = std::clamp(value, 0, 255);
              const size_t dst_index =
                  (static_cast<size_t>(origin_y + static_cast<uint32_t>(ty)) * image_width +
                   static_cast<size_t>(origin_x + static_cast<uint32_t>(tx))) *
                      components +
                  comp;
              if (dst_index >= decoded.size())
              {
                err = "tlg8: 出力バッファ範囲外に書き込もうとしました";
                return false;
              }
              decoded[dst_index] = static_cast<uint8_t>(value);
              const size_t tile_offset = (static_cast<size_t>(ty) * tile_w + static_cast<size_t>(tx)) * components + comp;
              if (tile_offset < decoded_mask.size())
                decoded_mask[tile_offset] = 1;
            }
          }
        }
      }
    }

    for (std::size_t row = 0; row < enc::kGolombRowCount; ++row)
    {
      if (row_offsets[row] != row_value_counts[row])
      {
        err = "tlg8: エントロピー列の消費量が一致しません";
        return false;
      }
    }

    return true;
  }
}
