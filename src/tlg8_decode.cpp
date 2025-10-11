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

#include <algorithm>
#include <array>
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
                       bool golomb_adaptive_update,
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

    if (golomb_adaptive_update)
    {
      const uint32_t table_flag = reader.get_upto8(1);
      if (table_flag)
      {
        enc::golomb_table_counts table = enc::current_golomb_table();
        const uint32_t mask = reader.get_upto8(enc::kGolombRowCount);
        for (uint32_t row = 0; row < enc::kGolombRowCount; ++row)
        {
          if (((mask >> row) & 1u) == 0u)
            continue;
          uint32_t sum = 0;
          for (uint32_t col = 0; col < enc::kGolombColumnCount; ++col)
          {
            const uint32_t value = reader.get(11);
            if (value > enc::kGolombRowSum)
            {
              err = "tlg8: ゴロムテーブル値が範囲外です";
              return false;
            }
            table[row][col] = static_cast<uint16_t>(value);
            sum += value;
          }
          if (sum != enc::kGolombRowSum)
          {
            err = "tlg8: ゴロムテーブルの合計が不正です";
            return false;
          }
        }
        enc::apply_golomb_table(table);
      }
    }

    struct block_choice
    {
      uint32_t predictor = 0;
      uint32_t filter = 0;
      uint32_t entropy = 0;
      uint32_t interleave = 0;
    };

    const uint32_t block_cols = (tile_w + enc::kBlockSize - 1) / enc::kBlockSize;
    const uint32_t block_rows_count = (tile_h + enc::kBlockSize - 1) / enc::kBlockSize;
    const uint32_t block_count = block_cols * block_rows_count;

    std::vector<block_choice> block_choices;
    block_choices.reserve(block_count);

    const uint32_t predictor_bits = tlg::detail::bit_width(enc::kNumPredictors);
    const uint32_t filter_bits = tlg::detail::bit_width(filter_count);
    const uint32_t entropy_bits = tlg::detail::bit_width(enc::kNumEntropyEncoders);
    const uint32_t interleave_bits = tlg::detail::bit_width(enc::kNumInterleaveFilter);

    std::array<enc::BlockChoiceEncoding, 4> field_modes{};
    for (auto &mode : field_modes)
    {
      const uint32_t mode_bits = reader.get_upto8(2);
      if (mode_bits >= static_cast<uint32_t>(enc::BlockChoiceEncoding::Count))
      {
        err = "tlg8: サイド情報のエンコード方式が不正です";
        return false;
      }
      mode = static_cast<enc::BlockChoiceEncoding>(mode_bits);
    }

    std::vector<uint32_t> predictor_stream;
    std::vector<uint32_t> filter_stream;
    std::vector<uint32_t> entropy_stream;
    std::vector<uint32_t> interleave_stream;
    predictor_stream.reserve(block_count);
    filter_stream.reserve(block_count);
    entropy_stream.reserve(block_count);
    interleave_stream.reserve(block_count);

    auto decode_stream = [&](enc::BlockChoiceEncoding mode,
                             uint32_t value_bits,
                             uint32_t max_value,
                             std::vector<uint32_t> &output,
                             const char *value_error) -> bool {
      output.clear();
      if (block_count == 0)
        return true;
      if (value_bits == 0)
      {
        if (mode != enc::BlockChoiceEncoding::Raw)
        {
          err = "tlg8: サイド情報のエンコード方式が不正です";
          return false;
        }
        output.assign(block_count, 0u);
        return true;
      }

      switch (mode)
      {
      case enc::BlockChoiceEncoding::Raw:
        for (uint32_t index = 0; index < block_count; ++index)
        {
          const uint32_t value = reader.get_upto8(value_bits);
          if (value >= max_value)
          {
            err = value_error;
            return false;
          }
          output.push_back(value);
        }
        return true;

      case enc::BlockChoiceEncoding::SameAsPrevious:
      {
        const uint32_t first = reader.get_upto8(value_bits);
        if (first >= max_value)
        {
          err = value_error;
          return false;
        }
        output.push_back(first);
        uint32_t previous = first;
        for (uint32_t index = 1; index < block_count; ++index)
        {
          const uint32_t same_flag = reader.get_upto8(1);
          if (same_flag)
          {
            output.push_back(previous);
          }
          else
          {
            const uint32_t value = reader.get_upto8(value_bits);
            if (value >= max_value)
            {
              err = value_error;
              return false;
            }
            output.push_back(value);
            previous = value;
          }
        }
        return true;
      }

      case enc::BlockChoiceEncoding::RunLength:
      {
        uint32_t decoded = 0;
        while (decoded < block_count)
        {
          const uint32_t value = reader.get_upto8(value_bits);
          if (value >= max_value)
          {
            err = value_error;
            return false;
          }
          uint32_t extra = 0;
          if (!tlg::v8::detail::bitio::get_varuint(reader, extra))
          {
            err = "tlg8: サイド情報のランレングスが不正です";
            return false;
          }
          const uint64_t run_length = static_cast<uint64_t>(extra) + 1u;
          if (run_length == 0 || decoded + run_length > block_count)
          {
            err = "tlg8: サイド情報のランレングスが範囲外です";
            return false;
          }
          output.insert(output.end(), static_cast<std::size_t>(run_length), value);
          decoded += static_cast<uint32_t>(run_length);
        }
        if (output.size() != block_count)
        {
          err = "tlg8: サイド情報の数が一致しません";
          return false;
        }
        return true;
      }

      default:
        break;
      }
      err = "tlg8: サイド情報のエンコード方式が不正です";
      return false;
    };

    if (!decode_stream(field_modes[0], predictor_bits, enc::kNumPredictors, predictor_stream, "tlg8: 不正な予測器インデックスです"))
      return false;
    if (!decode_stream(field_modes[1], filter_bits, filter_count, filter_stream, "tlg8: 不正なカラーフィルターインデックスです"))
      return false;
    if (!decode_stream(field_modes[2], entropy_bits, enc::kNumEntropyEncoders, entropy_stream, "tlg8: 不正なエントロピーインデックスです"))
      return false;
    if (!decode_stream(field_modes[3], interleave_bits, enc::kNumInterleaveFilter, interleave_stream, "tlg8: 不正なインターリーブフィルターです"))
      return false;

    for (uint32_t index = 0; index < block_count; ++index)
    {
      block_choice choice{};
      choice.predictor = (index < predictor_stream.size()) ? predictor_stream[index] : 0u;
      choice.filter = (index < filter_stream.size()) ? filter_stream[index] : 0u;
      choice.entropy = (index < entropy_stream.size()) ? entropy_stream[index] : 0u;
      choice.interleave = (index < interleave_stream.size()) ? interleave_stream[index] : 0u;
      block_choices.push_back(choice);
    }

    std::size_t block_choice_index = 0;

    for (uint32_t block_y = 0; block_y < tile_h; block_y += enc::kBlockSize)
    {
      const uint32_t block_h = std::min(enc::kBlockSize, tile_h - block_y);
      block_row_state row_state{};
      row_state.block_y = block_y;
      row_state.block_h = block_h;

      for (uint32_t block_x = 0; block_x < tile_w; block_x += enc::kBlockSize)
      {
        const uint32_t block_w = std::min(enc::kBlockSize, tile_w - block_x);
        if (block_choice_index >= block_choices.size())
        {
          err = "tlg8: サイド情報の数が不足しています";
          return false;
        }

        const auto &choice = block_choices[block_choice_index++];
        const uint32_t predictor_index = choice.predictor;
        const uint32_t filter_index = choice.filter;
        const uint32_t entropy_index = choice.entropy;
        const uint32_t interleave_index = choice.interleave;

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
        const bool uses_interleave_filter =
            (interleave_index == static_cast<uint32_t>(enc::InterleaveFilter::Interleave));
        const uint32_t used_components =
            std::min<uint32_t>(components, static_cast<uint32_t>(state.residuals.values.size()));
        if (uses_interleave_filter && used_components > 0)
        {
          const int row = enc::golomb_row_index(kind, enc::kInterleavedComponentIndex);
          if (row < 0 || row >= static_cast<int>(enc::kGolombRowCount))
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          row_value_counts[static_cast<std::size_t>(row)] +=
              static_cast<std::size_t>(value_count) * used_components;
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

    if (block_choice_index != block_choices.size())
    {
      err = "tlg8: サイド情報の数が余っています";
      return false;
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
        const bool uses_interleave_filter =
            (state.interleave_index == static_cast<uint32_t>(enc::InterleaveFilter::Interleave));
        const uint32_t used_components =
            std::min<uint32_t>(components, static_cast<uint32_t>(state.residuals.values.size()));
        if (uses_interleave_filter && used_components > 0)
        {
          const int row = enc::golomb_row_index(kind, enc::kInterleavedComponentIndex);
          if (row < 0 || row >= static_cast<int>(enc::kGolombRowCount))
          {
            err = "tlg8: 不正なゴロム行です";
            return false;
          }
          auto &values = row_values[static_cast<std::size_t>(row)];
          auto &offset = row_offsets[static_cast<std::size_t>(row)];
          const std::size_t combined_count = static_cast<std::size_t>(state.value_count) * used_components;
          if (offset + combined_count > values.size())
          {
            err = "tlg8: エントロピー列が不足しています";
            return false;
          }
          for (uint32_t comp = 0; comp < used_components; ++comp)
          {
            const std::size_t chunk_offset = offset + static_cast<std::size_t>(comp) * state.value_count;
            std::copy_n(values.data() + chunk_offset,
                        state.value_count,
                        state.residuals.values[comp].begin());
          }
          for (uint32_t comp = used_components; comp < components; ++comp)
            std::fill_n(state.residuals.values[comp].begin(), state.value_count, 0);
          offset += combined_count;
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
