#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_entropy.h"
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

    enc::component_colors residuals{};

    for (uint32_t block_y = 0; block_y < tile_h; block_y += enc::kBlockSize)
    {
      const uint32_t block_h = std::min(enc::kBlockSize, tile_h - block_y);
      for (uint32_t block_x = 0; block_x < tile_w; block_x += enc::kBlockSize)
      {
        const uint32_t block_w = std::min(enc::kBlockSize, tile_w - block_x);

        uint8_t predictor_index = 0;
        if (!reader.read_u8(predictor_index))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (predictor_index >= enc::kNumPredictors)
        {
          err = "tlg8: 不正な予測器インデックスです";
          return false;
        }

        uint8_t entropy_index = 0;
        if (!reader.read_u8(entropy_index))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (entropy_index >= enc::kNumEntropyEncoders)
        {
          err = "tlg8: 不正なエントロピーインデックスです";
          return false;
        }

        uint8_t stored_w = 0;
        uint8_t stored_h = 0;
        if (!reader.read_u8(stored_w) || !reader.read_u8(stored_h))
        {
          err = "tlg8: タイルデータが不足しています";
          return false;
        }
        if (stored_w == 0 || stored_h == 0)
        {
          err = "tlg8: 不正なブロック寸法です";
          return false;
        }
        if (stored_w != block_w || stored_h != block_h)
        {
          err = "tlg8: ブロック寸法が一致しません";
          return false;
        }

        const uint32_t value_count = static_cast<uint32_t>(stored_w) * stored_h;
        for (auto &component : residuals.values)
          component.fill(0);

        for (uint32_t comp = 0; comp < components; ++comp)
        {
          for (uint32_t i = 0; i < value_count; ++i)
          {
            uint16_t raw = 0;
            if (!reader.read_u16_le(raw))
            {
              err = "tlg8: タイルデータが不足しています";
              return false;
            }
            int16_t residual = 0;
            std::memcpy(&residual, &raw, sizeof(residual));
            residuals.values[comp][i] = residual;
          }
        }

        const auto predictor = predictors[predictor_index];

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
