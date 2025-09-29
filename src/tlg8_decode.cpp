#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_color_filter.h"
#include "tlg8_entropy.h"
#include "tlg8_reorder.h"
#include "tlg8_predictors.h"

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

    struct block_info
    {
      uint8_t predictor = 0;
      uint8_t filter = 0;
      uint8_t entropy = 0;
      uint8_t block_w = 0;
      uint8_t block_h = 0;
    };

    const uint32_t blocks_x = (tile_w + enc::kBlockSize - 1u) / enc::kBlockSize;
    const uint32_t blocks_y = (tile_h + enc::kBlockSize - 1u) / enc::kBlockSize;
    std::vector<block_info> blocks;
    blocks.reserve(static_cast<size_t>(blocks_x) * static_cast<size_t>(blocks_y));

    for (uint32_t block_y = 0; block_y < tile_h; block_y += enc::kBlockSize)
    {
      const uint32_t block_h = std::min(enc::kBlockSize, tile_h - block_y);
      for (uint32_t block_x = 0; block_x < tile_w; block_x += enc::kBlockSize)
      {
        const uint32_t block_w = std::min(enc::kBlockSize, tile_w - block_x);
        block_info info{};
        if (!reader.read_u8(info.predictor))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (info.predictor >= enc::kNumPredictors)
        {
          err = "tlg8: 不正な予測器インデックスです";
          return false;
        }

        if (!reader.read_u8(info.filter))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (info.filter >= enc::kColorFilterCodeCount)
        {
          err = "tlg8: 不正なカラーフィルターインデックスです";
          return false;
        }

        if (!reader.read_u8(info.entropy))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (info.entropy >= enc::kNumEntropyEncoders)
        {
          err = "tlg8: 不正なエントロピーインデックスです";
          return false;
        }

        if (!reader.read_u8(info.block_w) || !reader.read_u8(info.block_h))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (info.block_w == 0 || info.block_h == 0)
        {
          err = "tlg8: 不正なブロック寸法です";
          return false;
        }
        if (info.block_w != block_w || info.block_h != block_h)
        {
          err = "tlg8: ブロック寸法が一致しません";
          return false;
        }

        blocks.push_back(info);
      }
    }

    enc::entropy_decode_context entropy_ctx{};
    if (!enc::load_entropy_contexts(reader, entropy_ctx, err))
      return false;

    enc::component_colors residuals{};
    size_t block_index = 0;

    for (uint32_t block_y = 0; block_y < tile_h; block_y += enc::kBlockSize)
    {
      const uint32_t block_h = std::min(enc::kBlockSize, tile_h - block_y);
      for (uint32_t block_x = 0; block_x < tile_w; block_x += enc::kBlockSize)
      {
        const uint32_t block_w = std::min(enc::kBlockSize, tile_w - block_x);
        if (block_index >= blocks.size())
        {
          err = "tlg8: ブロック情報が不足しています";
          return false;
        }
        const block_info info = blocks[block_index++];
        if (info.block_w != block_w || info.block_h != block_h)
        {
          err = "tlg8: ブロック寸法が一致しません";
          return false;
        }

        const uint32_t value_count = static_cast<uint32_t>(info.block_w) * info.block_h;
        const auto predictor = predictors[info.predictor];
        const auto kind = entropy_encoders[info.entropy].kind;

        if (!enc::decode_block_from_context(entropy_ctx, kind, components, value_count, residuals, err))
          return false;

        enc::reorder_from_hilbert(residuals, components, block_w, block_h);

        enc::undo_color_filter(info.filter, residuals, components, value_count);

        for (uint32_t by = 0; by < block_h; ++by)
        {
          for (uint32_t bx = 0; bx < block_w; ++bx)
          {
            const int32_t tx = static_cast<int32_t>(block_x + bx);
            const int32_t ty = static_cast<int32_t>(block_y + by);
            const uint32_t value_index = by * block_w + bx;
            for (uint32_t comp = 0; comp < components; ++comp)
            {
              const uint8_t a = sample_pixel(decoded, image_width, components, origin_x, origin_y, tile_w, tile_h, tx - 1, ty, comp);
              const uint8_t b = sample_pixel(decoded, image_width, components, origin_x, origin_y, tile_w, tile_h, tx, ty - 1, comp);
              const uint8_t c = sample_pixel(decoded, image_width, components, origin_x, origin_y, tile_w, tile_h, tx - 1, ty - 1, comp);
              const uint8_t d = sample_pixel(decoded, image_width, components, origin_x, origin_y, tile_w, tile_h, tx + 1, ty - 1, comp);
              const uint8_t predicted = predictor(a, b, c, d);
              const int16_t residual = residuals.values[comp][value_index];
              int32_t value = static_cast<int32_t>(predicted) + static_cast<int32_t>(residual);
              value = std::clamp(value, 0, 255);
              const size_t dst_index =
                  (static_cast<size_t>(origin_y + static_cast<uint32_t>(ty)) * image_width +
                   static_cast<size_t>(origin_x + static_cast<uint32_t>(tx))) * components + comp;
              if (dst_index >= decoded.size())
              {
                err = "tlg8: 出力バッファ範囲外に書き込もうとしました";
                return false;
              }
              decoded[dst_index] = static_cast<uint8_t>(value);
            }
          }
        }
      }
    }

    return true;
  }
}
