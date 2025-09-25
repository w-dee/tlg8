#include "tlg7_common.h"
#include "tlg7_entropy_codec.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tlg_io_common.h"

namespace tlg::v7
{
  namespace
  {

    uint64_t estimate_residual_bits(const std::vector<int16_t> &residuals, std::size_t component_index)
    {
      static thread_local GolombResidualEntropyEncoder encoder;
      encoder.set_component_index(component_index);
      return encoder.estimate_bits(residuals);
    }

    std::int64_t estimate_sequence_cost(const std::vector<int16_t> &seq)
    {
      if (seq.empty())
        return 0;
      std::int64_t cost = 0;
      std::int64_t prev = static_cast<std::int64_t>(seq[0]);
      for (std::size_t i = 1; i < seq.size(); ++i)
      {
        const std::int64_t cur = static_cast<std::int64_t>(seq[i]);
        const std::int64_t diff = cur - prev;
        cost += diff * diff;
        prev = cur;
      }
      return cost;
    }

    int choose_filter_fast(const std::vector<int16_t> &src_b,
                           const std::vector<int16_t> &src_g,
                           const std::vector<int16_t> &src_r,
                           std::vector<int16_t> &dst_b,
                           std::vector<int16_t> &dst_g,
                           std::vector<int16_t> &dst_r)
    {
      int best_code = 0;
      std::int64_t best_score = std::numeric_limits<std::int64_t>::max();
      std::vector<int16_t> tb, tg, tr;
      tb.reserve(src_b.size());
      tg.reserve(src_g.size());
      tr.reserve(src_r.size());
      for (int perm = 0; perm < COLOR_FILTER_PERMUTATIONS; ++perm)
      {
        for (int primary = 0; primary < COLOR_FILTER_PRIMARY_PREDICTORS; ++primary)
        {
          for (int secondary = 0; secondary < COLOR_FILTER_SECONDARY_PREDICTORS; ++secondary)
          {
            const int code = (perm << 4) | (primary << 2) | secondary;
            tb.assign(src_b.begin(), src_b.end());
            tg.assign(src_g.begin(), src_g.end());
            tr.assign(src_r.begin(), src_r.end());
            apply_color_filter(code, tb, tg, tr);
            const std::int64_t score = estimate_sequence_cost(tb) + estimate_sequence_cost(tg) + estimate_sequence_cost(tr);
            if (score < best_score)
            {
              best_score = score;
              best_code = code;
              dst_b = tb;
              dst_g = tg;
              dst_r = tr;
            }
          }
        }
      }
      return best_code;
    }

    struct FilterSelectionResult
    {
      int code = 0;
      std::array<std::vector<int16_t>, 3> filtered;
    };

    struct FileCloser
    {
      void operator()(FILE *fp) const noexcept
      {
        if (fp)
          std::fclose(fp);
      }
    };

    void dump_residual_block(FILE *fp,
                             std::size_t component_index,
                             const BlockContext &ctx,
                             const std::vector<int16_t> &residuals)
    {
      if (!fp)
        return;

      const std::size_t block_x = ctx.x0 / BLOCK_SIZE;
      const std::size_t block_y = ctx.y0 / BLOCK_SIZE;
      std::fprintf(fp, "# Color component %zu at block %zu,%zu\n",
                   component_index,
                   block_x,
                   block_y);

      for (std::size_t i = 0; i < residuals.size(); ++i)
      {
        std::fprintf(fp, "%4d,", static_cast<int>(residuals[i]));
        if ((i + 1) % BLOCK_SIZE == 0)
          std::fputc('\n', fp);
      }

      if (residuals.size() % BLOCK_SIZE != 0)
        std::fputc('\n', fp);

      std::fputc('\n', fp);
    }

    FilterSelectionResult choose_filter_optimal(const BlockContext &ctx,
                                                const std::vector<int16_t> &residual_b,
                                                const std::vector<int16_t> &residual_g,
                                                const std::vector<int16_t> &residual_r)
    {
      FilterSelectionResult result;
      const bool is_full_block = (ctx.bw == BLOCK_SIZE && ctx.bh == BLOCK_SIZE);

      uint64_t best_bits = std::numeric_limits<uint64_t>::max();

      for (int perm = 0; perm < COLOR_FILTER_PERMUTATIONS; ++perm)
      {
        for (int primary = 0; primary < COLOR_FILTER_PRIMARY_PREDICTORS; ++primary)
        {
          for (int secondary = 0; secondary < COLOR_FILTER_SECONDARY_PREDICTORS; ++secondary)
          {
            const int code = (perm << 4) | (primary << 2) | secondary;
            std::array<std::vector<int16_t>, 3> candidate = {residual_b, residual_g, residual_r};
            apply_color_filter(code, candidate[0], candidate[1], candidate[2]);

            uint64_t total_bits = 0;
            for (std::size_t c = 0; c < candidate.size(); ++c)
            {
              std::vector<int16_t> tmp = candidate[c];
              if (is_full_block)
                reorder_to_hilbert(tmp);
              total_bits += estimate_residual_bits(tmp, c);
              if (total_bits >= best_bits)
                break;
            }

            if (total_bits < best_bits)
            {
              best_bits = total_bits;
              result.code = code;
              result.filtered = std::move(candidate);
            }
          }
        }
      }

      return result;
    }

    void compute_block_residuals(const BlockContext &ctx,
                                 const std::vector<uint8_t> &block_values,
                                 const detail::image<uint8_t> &reference_plane,
                                 PredictorMode mode,
                                 std::vector<int16_t> &residual_out)
    {
      const std::size_t pixel_count = ctx.bw * ctx.bh;
      residual_out.resize(pixel_count);
      std::vector<uint8_t> local_plane(pixel_count, 0);
      std::size_t idx = 0;
      for (std::size_t y = 0; y < ctx.bh; ++y)
      {
        for (std::size_t x = 0; x < ctx.bw; ++x)
        {
          const int gx = static_cast<int>(ctx.x0 + x);
          const int gy = static_cast<int>(ctx.y0 + y);
          const uint8_t value = block_values[idx];

          const int a = (x > 0) ? local_plane[y * ctx.bw + (x - 1)]
                                : sample_pixel(reference_plane, gx - 1, gy);
          const int b = (y > 0) ? local_plane[(y - 1) * ctx.bw + x]
                                : sample_pixel(reference_plane, gx, gy - 1);
          const int cdiag = (x > 0 && y > 0) ? local_plane[(y - 1) * ctx.bw + (x - 1)]
                                             : sample_pixel(reference_plane, gx - 1, gy - 1);

          const int pred = apply_predictor<uint8_t>(mode, a, b, cdiag);
          residual_out[idx] = static_cast<int16_t>(static_cast<int>(value) - pred);
          local_plane[y * ctx.bw + x] = value;
          ++idx;
        }
      }
    }

    inline uint8_t pack_filter_predictor(int filter_code, PredictorMode mode)
    {
      const int predictor_bit = (mode == PredictorMode::AVG) ? 1 : 0;
      return static_cast<uint8_t>((filter_code << 1) | predictor_bit);
    }

    struct BlockCandidate
    {
      PredictorMode mode = PredictorMode::MED;
      int filter_code = 0;
      uint64_t bit_cost = std::numeric_limits<uint64_t>::max();
      std::vector<std::vector<int16_t>> residuals;
    };

  } // namespace

  namespace enc
  {

    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int colors,
                   bool fast_mode,
                   const std::string &dump_residuals_path,
                   TlgOptions::DumpResidualsOrder dump_residuals_order,
                   std::string &err)
    {
      err.clear();
      if (!(colors == 1 || colors == 3 || colors == 4))
      {
        err = "tlg7: unsupported color count";
        return false;
      }
      if (!(src.channels == 3 || src.channels == 4))
      {
        err = "tlg7: unsupported source format";
        return false;
      }
      if (src.width == 0 || src.height == 0)
      {
        err = "tlg7: empty image";
        return false;
      }

      const std::size_t width = src.width;
      const std::size_t height = src.height;
      const std::size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const std::size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const std::size_t block_count = blocks_x * blocks_y;
      const std::size_t chunk_rows = (height + CHUNK_SCAN_LINES - 1) / CHUNK_SCAN_LINES;

      if (block_count == 0 || chunk_rows == 0)
      {
        err = "tlg7: invalid block geometry";
        return false;
      }
      if (block_count > std::numeric_limits<uint32_t>::max() || chunk_rows > std::numeric_limits<uint32_t>::max())
      {
        err = "tlg7: image too large";
        return false;
      }

      detail::Header hdr;
      hdr.colors = static_cast<uint8_t>(colors);
      hdr.width = src.width;
      hdr.height = src.height;
      hdr.block_count = static_cast<uint32_t>(block_count);
      hdr.chunk_count = static_cast<uint32_t>(chunk_rows);

      std::unique_ptr<FILE, FileCloser> dump_file;
      if (!dump_residuals_path.empty())
      {
        FILE *dump_fp = std::fopen(dump_residuals_path.c_str(), "w");
        if (!dump_fp)
        {
          err = "tlg7: cannot open residual dump file: " + dump_residuals_path;
          return false;
        }
        dump_file.reset(dump_fp);
      }

      const bool dump_before_hilbert = (dump_residuals_order == TlgOptions::DumpResidualsOrder::BeforeHilbert);
      const bool dump_after_hilbert = (dump_residuals_order == TlgOptions::DumpResidualsOrder::AfterHilbert);

      if (!detail::write_header(fp, hdr))
      {
        err = "tlg7: write header";
        return false;
      }

      const std::size_t component_count = (colors == 4) ? 4u : (colors == 3 ? 3u : 1u);

      std::vector<uint8_t> filter_indices(block_count, 0);
      if (!filter_indices.empty())
      {
        std::vector<uint8_t> zero(block_count, 0);
        if (!zero.empty() && fwrite(zero.data(), 1, zero.size(), fp) != zero.size())
        {
          err = "tlg7: write filter placeholder";
          return false;
        }
      }
      const long filter_pos = std::ftell(fp) - static_cast<long>(filter_indices.size());

      auto planes = extract_planes(src, colors);
      std::vector<detail::image<uint8_t>> filtered_planes;
      filtered_planes.reserve(component_count);
      for (std::size_t i = 0; i < component_count; ++i)
        filtered_planes.emplace_back(width, height, 0);

      GolombResidualEntropyEncoder entropy_encoder;
      std::vector<uint8_t> encoded_stream;

      for (std::size_t chunk_y = 0; chunk_y < chunk_rows; ++chunk_y)
      {
        const std::size_t chunk_y0 = chunk_y * CHUNK_SCAN_LINES;
        const std::size_t chunk_height = std::min<std::size_t>(CHUNK_SCAN_LINES, height - chunk_y0);
        const std::size_t chunk_pixels = chunk_height * width;

        std::vector<std::vector<int16_t>> chunk_residuals(component_count);
        for (auto &vec : chunk_residuals)
          vec.reserve(chunk_pixels);

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
            const bool is_full_block = (ctx.bw == BLOCK_SIZE && ctx.bh == BLOCK_SIZE);

            std::vector<std::vector<uint8_t>> block_values(component_count);
            for (std::size_t c = 0; c < component_count; ++c)
              copy_block_from_plane(planes[c], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_values[c]);

            BlockCandidate best_candidate;
            bool has_candidate = false;

            for (PredictorMode mode : PREDICTOR_CANDIDATES)
            {
              BlockCandidate cand;
              cand.mode = mode;
              cand.residuals.resize(component_count);

              for (std::size_t c = 0; c < component_count; ++c)
                compute_block_residuals(ctx, block_values[c], filtered_planes[c], mode, cand.residuals[c]);

              int filter_code = 0;
              if (component_count >= 3)
              {
                if (fast_mode)
                {
                  std::vector<int16_t> filtered_b;
                  std::vector<int16_t> filtered_g;
                  std::vector<int16_t> filtered_r;
                  filter_code = choose_filter_fast(cand.residuals[0], cand.residuals[1], cand.residuals[2], filtered_b, filtered_g, filtered_r);
                  cand.residuals[0] = std::move(filtered_b);
                  cand.residuals[1] = std::move(filtered_g);
                  cand.residuals[2] = std::move(filtered_r);
                }
                else
                {
                  auto selection = choose_filter_optimal(ctx,
                                                         cand.residuals[0],
                                                         cand.residuals[1],
                                                         cand.residuals[2]);
                  filter_code = selection.code;
                  cand.residuals[0] = std::move(selection.filtered[0]);
                  cand.residuals[1] = std::move(selection.filtered[1]);
                  cand.residuals[2] = std::move(selection.filtered[2]);
                }
              }

              cand.filter_code = filter_code;

              uint64_t total_bits = 0;
              for (std::size_t c = 0; c < component_count; ++c)
              {
                std::vector<int16_t> tmp = cand.residuals[c];
                if (is_full_block)
                  reorder_to_hilbert(tmp);
                total_bits += estimate_residual_bits(tmp, c);
                if (total_bits >= std::numeric_limits<uint64_t>::max())
                  break;
              }
              cand.bit_cost = total_bits;

              if (!has_candidate || cand.bit_cost < best_candidate.bit_cost)
              {
                best_candidate = std::move(cand);
                has_candidate = true;
              }
            }

            if (!has_candidate)
            {
              err = "tlg7: predictor evaluation failed";
              return false;
            }

            filter_indices[ctx.index] = pack_filter_predictor(best_candidate.filter_code, best_candidate.mode);

            for (std::size_t c = 0; c < component_count; ++c)
            {
              std::vector<int16_t> residual = std::move(best_candidate.residuals[c]);
              if (dump_file && dump_before_hilbert)
                dump_residual_block(dump_file.get(), c, ctx, residual);
              if (is_full_block)
                reorder_to_hilbert(residual);
              if (dump_file && dump_after_hilbert)
                dump_residual_block(dump_file.get(), c, ctx, residual);
              chunk_residuals[c].insert(chunk_residuals[c].end(), residual.begin(), residual.end());
            }

            for (std::size_t c = 0; c < component_count; ++c)
              store_block_to_plane(filtered_planes[c], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_values[c]);
          }
        }

        for (std::size_t c = 0; c < component_count; ++c)
        {
          if (chunk_residuals[c].size() != chunk_pixels)
          {
            err = "tlg7: residual size mismatch";
            return false;
          }
        }

        for (std::size_t c = 0; c < component_count; ++c)
        {
          uint32_t bit_length = 0;
          entropy_encoder.set_component_index(c);
          entropy_encoder.encode(chunk_residuals[c], encoded_stream, bit_length);
          tlg::detail::write_u32le(fp, bit_length);
          const std::size_t byte_count = (bit_length + 7u) / 8u;
          if (encoded_stream.size() != byte_count)
            encoded_stream.resize(byte_count);
          if (byte_count && fwrite(encoded_stream.data(), 1, byte_count, fp) != byte_count)
          {
            err = "tlg7: write residual data";
            return false;
          }
        }
      }

      if (!filter_indices.empty())
      {
        if (std::fseek(fp, filter_pos, SEEK_SET) != 0)
        {
          err = "tlg7: seek filter table";
          return false;
        }
        if (!filter_indices.empty() && fwrite(filter_indices.data(), 1, filter_indices.size(), fp) != filter_indices.size())
        {
          err = "tlg7: write filter table";
          return false;
        }
        if (std::fseek(fp, 0, SEEK_END) != 0)
        {
          err = "tlg7: seek end";
          return false;
        }
      }

      return true;
    }

  } // namespace enc

} // namespace tlg::v7
