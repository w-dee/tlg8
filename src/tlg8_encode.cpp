#include "tlg8_bit_io.h"
#include "tlg8_block.h"
#include "tlg8_entropy.h"
#include "tlg8_predictors.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace
{
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
          const uint8_t a = accessor.sample(tx - 1, ty, comp);
          const uint8_t b = accessor.sample(tx, ty - 1, comp);
          const uint8_t c = accessor.sample(tx - 1, ty - 1, comp);
          const uint8_t d = accessor.sample(tx + 1, ty - 1, comp);
          const uint8_t predicted = predictor(a, b, c, d);
          const uint8_t actual = accessor.sample(tx, ty, comp);
          out.values[comp][index] = static_cast<int16_t>(static_cast<int32_t>(actual) - static_cast<int32_t>(predicted));
        }
        ++index;
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
                       std::string &err)
  {
    if (components == 0 || components > 4)
    {
      err = "tlg8: コンポーネント数が不正です";
      return false;
    }

    const auto &predictors = predictor_table();
    const auto &entropy_encoders = entropy_encoder_table();
    tile_accessor accessor(image_base, image_width, components, origin_x, origin_y, tile_w, tile_h);

    for (uint32_t block_y = 0; block_y < tile_h; block_y += kBlockSize)
    {
      const uint32_t block_h = std::min(kBlockSize, tile_h - block_y);
      for (uint32_t block_x = 0; block_x < tile_w; block_x += kBlockSize)
      {
        const uint32_t block_w = std::min(kBlockSize, tile_w - block_x);
        const uint32_t value_count = block_w * block_h;

        component_colors best_block{};
        uint32_t best_predictor = 0;
        uint32_t best_entropy = 0;
        uint64_t best_bits = std::numeric_limits<uint64_t>::max();

        component_colors candidate{};
        for (uint32_t predictor_index = 0; predictor_index < kNumPredictors; ++predictor_index)
        {
          compute_residual_block(accessor, candidate, predictors[predictor_index], components, block_x, block_y, block_w, block_h);
          for (uint32_t entropy_index = 0; entropy_index < kNumEntropyEncoders; ++entropy_index)
          {
            const uint64_t estimated_bits = entropy_encoders[entropy_index].estimate_bits(candidate, components, value_count);
            if (estimated_bits < best_bits)
            {
              best_bits = estimated_bits;
              best_predictor = predictor_index;
              best_entropy = entropy_index;
              best_block = candidate;
            }
          }
        }

        if (!writer.write_u8(static_cast<uint8_t>(best_predictor)))
        {
          err = "tlg8: 予測器インデックスの書き込みに失敗しました";
          return false;
        }
        if (!writer.write_u8(static_cast<uint8_t>(best_entropy)))
        {
          err = "tlg8: エントロピーインデックスの書き込みに失敗しました";
          return false;
        }
        if (!writer.write_u8(static_cast<uint8_t>(block_w)))
        {
          err = "tlg8: ブロック幅の書き込みに失敗しました";
          return false;
        }
        if (!writer.write_u8(static_cast<uint8_t>(block_h)))
        {
          err = "tlg8: ブロック高さの書き込みに失敗しました";
          return false;
        }
        if (!entropy_encoders[best_entropy].encode_block(writer, best_block, components, value_count, err))
          return false;
      }
    }

    return true;
  }
}
