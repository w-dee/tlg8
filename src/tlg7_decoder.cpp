#include "tlg7_common.h"
#include "tlg7_entropy_codec.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "tlg_io_common.h"

namespace tlg::v7
{

  bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err)
  {
    err.clear();
    detail::Header hdr{};
    if (!detail::read_header(fp, hdr, err))
      return false;

    const std::size_t width = hdr.width;
    const std::size_t height = hdr.height;
    const std::size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const std::size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (static_cast<uint64_t>(blocks_x) * static_cast<uint64_t>(blocks_y) != hdr.block_count)
    {
      err = "tlg7: block count mismatch";
      return false;
    }

    std::vector<uint8_t> filter_indices(hdr.block_count);
    if (!filter_indices.empty())
    {
      if (!tlg::detail::read_exact(fp, filter_indices.data(), filter_indices.size()))
      {
        err = "tlg7: read filter indices";
        return false;
      }
    }

    const std::size_t component_count = (hdr.colors == 4) ? 4u : (hdr.colors == 3 ? 3u : 1u);

    std::vector<detail::image<uint8_t>> filtered_planes;
    filtered_planes.reserve(component_count);
    std::vector<detail::image<uint8_t>> output_planes;
    output_planes.reserve(component_count);
    for (std::size_t i = 0; i < component_count; ++i)
    {
      filtered_planes.emplace_back(width, height, 0);
      output_planes.emplace_back(width, height, 0);
    }

#ifdef TLG7_USE_MED_PREDICTOR
    ActivePredictor predictor;
#else
    ActivePredictor::Config mwr_cfg;
    ActivePredictor predictor(mwr_cfg, 0, 255);
#endif

    GolombResidualEntropyDecoder entropy_decoder;
    std::vector<uint8_t> encoded_stream;

    const std::size_t chunk_rows = (height + CHUNK_SCAN_LINES - 1) / CHUNK_SCAN_LINES;

    for (std::size_t chunk_y = 0; chunk_y < chunk_rows; ++chunk_y)
    {
      const std::size_t chunk_y0 = chunk_y * CHUNK_SCAN_LINES;
      const std::size_t chunk_height = std::min<std::size_t>(CHUNK_SCAN_LINES, height - chunk_y0);
      const std::size_t chunk_pixels = chunk_height * width;

      std::vector<std::vector<int16_t>> chunk_residuals(component_count);

      for (std::size_t c = 0; c < component_count; ++c)
      {
        const uint32_t bit_length = tlg::detail::read_u32le(fp);
        const std::size_t byte_count = (bit_length + 7u) / 8u;
        encoded_stream.resize(byte_count);
        if (byte_count && !tlg::detail::read_exact(fp, encoded_stream.data(), byte_count))
        {
          err = "tlg7: read residual stream";
          return false;
        }
        entropy_decoder.set_component_index(c);
        if (!entropy_decoder.decode(encoded_stream.data(), encoded_stream.size(), chunk_pixels, chunk_residuals[c]))
        {
          err = "tlg7: residual decode error";
          return false;
        }
      }

      std::vector<ActivePredictor::State> states(component_count);
      std::vector<std::size_t> residual_cursor(component_count, 0);

      const std::size_t chunk_block_row_start = chunk_y0 / BLOCK_SIZE;
      const std::size_t chunk_block_row_end = std::min<std::size_t>((chunk_y0 + chunk_height + BLOCK_SIZE - 1) / BLOCK_SIZE, blocks_y);

      for (std::size_t by = chunk_block_row_start; by < chunk_block_row_end; ++by)
      {
        const std::size_t y0 = by * BLOCK_SIZE;
        if (y0 >= height)
          break;
        for (std::size_t bx = 0; bx < blocks_x; ++bx)
        {
          const BlockContext ctx = make_block_context(bx, by, width, height, blocks_x);
          const std::size_t pixel_count = ctx.bw * ctx.bh;
          const bool is_full_block = (ctx.bw == BLOCK_SIZE && ctx.bh == BLOCK_SIZE);

          std::vector<std::vector<int16_t>> block_residuals(component_count);
          for (std::size_t c = 0; c < component_count; ++c)
          {
            block_residuals[c].assign(chunk_residuals[c].begin() + residual_cursor[c],
                                      chunk_residuals[c].begin() + residual_cursor[c] + pixel_count);
            residual_cursor[c] += pixel_count;
            if (is_full_block)
              reorder_from_hilbert(block_residuals[c]);
          }

          if (component_count >= 3)
          {
            undo_color_filter(filter_indices[ctx.index], block_residuals[0], block_residuals[1], block_residuals[2]);
          }

          std::vector<std::vector<uint8_t>> block_pixels(component_count);
          for (std::size_t c = 0; c < component_count; ++c)
          {
            block_pixels[c].resize(pixel_count);

            std::size_t idx = 0;
            for (std::size_t y = 0; y < ctx.bh; ++y)
            {
              for (std::size_t x = 0; x < ctx.bw; ++x)
              {
                const int gx = static_cast<int>(ctx.x0 + x);
                const int gy = static_cast<int>(ctx.y0 + y);

                const int a = sample_pixel(filtered_planes[c], gx - 1, gy);
                const int b = sample_pixel(filtered_planes[c], gx, gy - 1);
                const int cdiag = sample_pixel(filtered_planes[c], gx - 1, gy - 1);
                const int d = sample_pixel(filtered_planes[c], gx + 1, gy - 1);
                const int f = sample_pixel(filtered_planes[c], gx, gy - 2);

                auto [pred, pid] = predictor.predict_and_choose<uint8_t>(a, b, cdiag, d, f, states[c]);
                int recon = pred + block_residuals[c][idx];
                if (recon < 0)
                  recon = 0;
                else if (recon > 255)
                  recon = 255;

                filtered_planes[c].row_ptr(ctx.y0 + y)[ctx.x0 + x] = static_cast<uint8_t>(recon);
                block_pixels[c][idx] = static_cast<uint8_t>(recon);

                const int Dh = std::abs(a - b);
                const int Dv = std::abs(b - cdiag);
                predictor.update_state(states[c], pid, std::abs(recon - pred), Dh, Dv);
                ++idx;
              }
            }
          }

          for (std::size_t c = 0; c < component_count; ++c)
            store_block_to_plane(output_planes[c], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_pixels[c]);
        }
      }

      for (std::size_t c = 0; c < component_count; ++c)
      {
        if (residual_cursor[c] != chunk_pixels)
        {
          err = "tlg7: residual cursor mismatch";
          return false;
        }
      }
    }

    out = assemble_pixelbuffer(output_planes, hdr.width, hdr.height, hdr.colors == 1 ? 3 : hdr.colors);
    if (hdr.colors == 1)
    {
      const std::size_t pixels = static_cast<std::size_t>(hdr.width) * hdr.height;
      for (std::size_t i = 0; i < pixels; ++i)
      {
        const uint8_t g = out.data[i * 3 + 0];
        out.data[i * 3 + 1] = g;
        out.data[i * 3 + 2] = g;
      }
    }
    return true;
  }

} // namespace tlg::v7
