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
    constexpr std::array<const char *, COLOR_FILTER_PERMUTATIONS> kPermutationNames = {
        "BGR",
        "BRG",
        "GBR",
        "GRB",
        "RBG",
        "RGB"};

    constexpr std::array<const char *, COLOR_FILTER_PRIMARY_PREDICTORS> kPrimaryDescriptions = {
        "0",
        "c0",
        "c0/2",
        "3*c0/2"};

    constexpr std::array<const char *, COLOR_FILTER_SECONDARY_PREDICTORS> kSecondaryDescriptions = {
        "0",
        "c0",
        "r1",
        "(c0+r1)/2"};

    std::size_t wrap_index(int value, int modulo)
    {
      if (modulo <= 0)
        return 0;
      const int mod = value % modulo;
      return static_cast<std::size_t>((mod < 0) ? mod + modulo : mod);
    }

    const char *describe_mode(PredictorMode mode)
    {
      switch (mode)
      {
      case PredictorMode::AVG:
        return "AVG";
      case PredictorMode::MED:
      default:
        return "MED";
      }
    }

    const char *describe_diff(int diff_index)
    {
      switch (static_cast<DiffFilterType>(diff_index))
      {
      case DiffFilterType::NWSE:
        return "NWSE";
      case DiffFilterType::NESW:
        return "NESW";
      case DiffFilterType::HORZ:
        return "HORZ";
      case DiffFilterType::VERT:
        return "VERT";
      case DiffFilterType::None:
        return "None";
      case DiffFilterType::Count:
      default:
        return "Unknown";
      }
    }

    uint64_t estimate_residual_bits(const std::vector<int16_t> &residuals, std::size_t component_index)
    {
      static thread_local GolombResidualEntropyEncoder encoder;
      encoder.set_component_index(component_index);
      return encoder.estimate_bits(residuals);
    }

    double compute_squared_entropy(const std::vector<std::vector<int16_t>> &residuals)
    {
      long double sum = 0.0;
      for (const auto &component : residuals)
      {
        for (const auto value : component)
        {
          const long double v = static_cast<long double>(value);
          sum += v * v;
        }
      }
      return static_cast<double>(sum);
    }

    struct FilterSelectionResult
    {
      int code = 0;
      int total_bits = 0;
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
                             const std::vector<int16_t> &residuals,
                             const side_info &info,
                             uint64_t bit_cost)
    {
      if (!fp)
        return;

      const std::size_t block_x = ctx.x0 / BLOCK_SIZE;
      const std::size_t block_y = ctx.y0 / BLOCK_SIZE;
      const uint16_t packed = pack_block_sideinfo(info);
      const int filter_code = info.filter_code;
      const int perm_raw = (filter_code >> 4) & 0x7;
      const int primary_idx = (filter_code >> 2) & 0x3;
      const int secondary_idx = filter_code & 0x3;
      const std::size_t perm_idx = wrap_index(perm_raw, COLOR_FILTER_PERMUTATIONS);
      const char *perm_name = kPermutationNames[perm_idx];
      const char *p0_desc = kPrimaryDescriptions[static_cast<std::size_t>(primary_idx) % kPrimaryDescriptions.size()];
      const char *p1_desc = kSecondaryDescriptions[static_cast<std::size_t>(secondary_idx) % kSecondaryDescriptions.size()];
      const char *mode_desc = describe_mode(info.mode);
      const char *diff_desc = describe_diff(info.diff_index);

      std::fprintf(fp, "# Color component %zu at block %zu,%zu\n",
                   component_index,
                   block_x,
                   block_y);
      std::fprintf(fp, "# filter_code: 0x%02X(perm=%s, P0=%s, P1=%s)\n",
                   static_cast<unsigned>(filter_code & 0xFF),
                   perm_name,
                   p0_desc,
                   p1_desc);
      std::fprintf(fp, "# mode: %s\n", mode_desc);
      std::fprintf(fp, "# diff: %s (packed=0x%03X, est_size=%d)\n",
                   diff_desc,
                   static_cast<unsigned>(packed),
                   (int)bit_cost);

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
              result.total_bits = static_cast<int>(total_bits);
              result.filtered = std::move(candidate);
            }
          }
        }
      }

      return result;
    }

    template <typename SampleT>
    void compute_per_block_prediction(const BlockContext &ctx,
                                      const std::vector<SampleT> &block_values,
                                      const detail::image<SampleT> &reference_plane,
                                      PredictorMode mode,
                                      std::vector<int16_t> &residual_out)
    {
      const std::size_t pixel_count = ctx.bw * ctx.bh;
      residual_out.resize(pixel_count);
      std::vector<SampleT> local_plane(pixel_count, 0);
      std::size_t idx = 0;
      for (std::size_t y = 0; y < ctx.bh; ++y)
      {
        for (std::size_t x = 0; x < ctx.bw; ++x)
        {
          const int gx = static_cast<int>(ctx.x0 + x);
          const int gy = static_cast<int>(ctx.y0 + y);
          const SampleT value = block_values[idx];

          const int a = (x > 0) ? static_cast<int>(local_plane[y * ctx.bw + (x - 1)])
                                : sample_pixel(reference_plane, gx - 1, gy);
          const int b = (y > 0) ? static_cast<int>(local_plane[(y - 1) * ctx.bw + x])
                                : sample_pixel(reference_plane, gx, gy - 1);
          const int cdiag = (x > 0 && y > 0) ? static_cast<int>(local_plane[(y - 1) * ctx.bw + (x - 1)])
                                             : sample_pixel(reference_plane, gx - 1, gy - 1);

          const int pred = apply_predictor<SampleT>(mode, a, b, cdiag);
          residual_out[idx] = static_cast<int16_t>(static_cast<int>(value) - pred);
          local_plane[y * ctx.bw + x] = value;
          ++idx;
        }
      }
    }

    struct BlockCandidate
    {
      PredictorMode mode = PredictorMode::MED;
      int filter_code = 0;
      uint64_t bit_cost = std::numeric_limits<uint64_t>::max();
      std::vector<std::vector<int16_t>> residuals;
      int diff_filter_index = 0;
      std::vector<std::vector<int16_t>> predictor_signal;
    };

  } // namespace

  namespace enc
  {

    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int colors,
                   TlgOptions::Tlg7PipelineOrder pipeline_order,
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

      const PipelineOrder order = (pipeline_order == TlgOptions::Tlg7PipelineOrder::PredictorThenFilter)
                                      ? PipelineOrder::PredictorThenFilter
                                      : PipelineOrder::FilterThenPredictor;

      detail::Header hdr;
      hdr.colors = static_cast<uint8_t>(colors);
      hdr.flags = static_cast<uint8_t>(order == PipelineOrder::FilterThenPredictor ? 1 : 0);
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

      auto planes = extract_planes(src, colors);
      std::vector<detail::image<uint8_t>> predictor_planes_u8;
      std::vector<detail::image<int16_t>> predictor_planes_i16;
      if (order == PipelineOrder::PredictorThenFilter)
      {
        predictor_planes_u8.reserve(component_count);
        for (std::size_t i = 0; i < component_count; ++i)
          predictor_planes_u8.emplace_back(width, height, 0);
      }
      else
      {
        predictor_planes_i16.reserve(component_count);
        for (std::size_t i = 0; i < component_count; ++i)
          predictor_planes_i16.emplace_back(width, height, static_cast<int16_t>(0));
      }

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
        const std::size_t chunk_block_capacity = (chunk_block_row_end - chunk_block_row_start) * blocks_x;

        std::vector<uint16_t> chunk_sideinfo;
        chunk_sideinfo.reserve(chunk_block_capacity);

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
            double best_prediction_entropy = std::numeric_limits<double>::infinity();
            double best_color_filter_entropy = std::numeric_limits<double>::infinity();

            for (PredictorMode mode : PREDICTOR_CANDIDATES)
            {
              BlockCandidate cand;
              cand.mode = mode;
              cand.residuals.resize(component_count);

              if (order == PipelineOrder::PredictorThenFilter)
              {
                for (std::size_t c = 0; c < component_count; ++c)
                  compute_per_block_prediction(ctx, block_values[c], predictor_planes_u8[c], mode, cand.residuals[c]);

                const double predictor_entropy = compute_squared_entropy(cand.residuals);
                if (best_prediction_entropy < std::numeric_limits<double>::infinity() &&
                    predictor_entropy > best_prediction_entropy * PER_BLOCK_PREDICTION_EARLY_EXIT_RATIO)
                  continue;
                if (predictor_entropy < best_prediction_entropy)
                  best_prediction_entropy = predictor_entropy;

                int filter_code = 0;
                if (component_count >= 3)
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

                cand.filter_code = filter_code;
                const double color_filter_entropy = compute_squared_entropy(cand.residuals);
                if (best_color_filter_entropy < std::numeric_limits<double>::infinity() &&
                    color_filter_entropy > best_color_filter_entropy * COLOR_FILTER_EARLY_EXIT_RATIO)
                  continue;
                if (color_filter_entropy < best_color_filter_entropy)
                  best_color_filter_entropy = color_filter_entropy;
              }
              else
              {
                std::vector<std::vector<int16_t>> base_signal(component_count);
                for (std::size_t c = 0; c < component_count; ++c)
                {
                  base_signal[c].resize(block_values[c].size());
                  std::transform(block_values[c].begin(),
                                 block_values[c].end(),
                                 base_signal[c].begin(),
                                 [](uint8_t v)
                                 { return static_cast<int16_t>(v); });
                }

                if (component_count >= 3)
                {
                  uint64_t best_bits_estimate = std::numeric_limits<uint64_t>::max();
                  int best_filter_code = 0;
                  std::vector<std::vector<int16_t>> best_signal(component_count);
                  std::vector<std::vector<int16_t>> best_residuals(component_count);
                  bool found_candidate = false;

                  for (int perm = 0; perm < COLOR_FILTER_PERMUTATIONS; ++perm)
                  {
                    for (int primary = 0; primary < COLOR_FILTER_PRIMARY_PREDICTORS; ++primary)
                    {
                      for (int secondary = 0; secondary < COLOR_FILTER_SECONDARY_PREDICTORS; ++secondary)
                      {
                        const int code = (perm << 4) | (primary << 2) | secondary;
                        std::array<std::vector<int16_t>, 3> filtered = {base_signal[0],
                                                                        base_signal[1],
                                                                        base_signal[2]};
                        apply_color_filter(code, filtered[0], filtered[1], filtered[2]);

                        std::vector<std::vector<int16_t>> candidate_signal(component_count);
                        candidate_signal[0] = filtered[0];
                        candidate_signal[1] = filtered[1];
                        candidate_signal[2] = filtered[2];
                        for (std::size_t extra = 3; extra < component_count; ++extra)
                        {
                          candidate_signal[extra] = base_signal[extra];
                        }

                        const double color_filter_entropy = compute_squared_entropy(candidate_signal);
                        if (best_color_filter_entropy < std::numeric_limits<double>::infinity() &&
                            color_filter_entropy > best_color_filter_entropy * COLOR_FILTER_EARLY_EXIT_RATIO)
                          continue;

                        std::vector<std::vector<int16_t>> candidate_residuals(component_count);
                        for (std::size_t c = 0; c < component_count; ++c)
                          compute_per_block_prediction(ctx, candidate_signal[c], predictor_planes_i16[c], mode, candidate_residuals[c]);

                        const double predictor_entropy = compute_squared_entropy(candidate_residuals);
                        if (best_prediction_entropy < std::numeric_limits<double>::infinity() &&
                            predictor_entropy > best_prediction_entropy * PER_BLOCK_PREDICTION_EARLY_EXIT_RATIO)
                          continue;
                        if (predictor_entropy < best_prediction_entropy)
                          best_prediction_entropy = predictor_entropy;

                        if (color_filter_entropy < best_color_filter_entropy)
                          best_color_filter_entropy = color_filter_entropy;

                        uint64_t total_bits = 0;
                        for (std::size_t c = 0; c < component_count; ++c)
                        {
                          std::vector<int16_t> tmp = candidate_residuals[c];
                          if (is_full_block)
                            reorder_to_hilbert(tmp);
                          total_bits += estimate_residual_bits(tmp, c);
                          if (total_bits >= best_bits_estimate)
                            break;
                        }

                        if (total_bits < best_bits_estimate)
                        {
                          best_bits_estimate = total_bits;
                          best_filter_code = code;
                          best_signal = std::move(candidate_signal);
                          best_residuals = std::move(candidate_residuals);
                          found_candidate = true;
                        }
                      }
                    }
                  }

                  if (!found_candidate)
                    continue;

                  cand.filter_code = best_filter_code;
                  cand.predictor_signal = std::move(best_signal);
                  cand.residuals = std::move(best_residuals);
                }
                else
                {
                  cand.filter_code = 0;
                  const double color_filter_entropy = compute_squared_entropy(base_signal);
                  if (best_color_filter_entropy < std::numeric_limits<double>::infinity() &&
                      color_filter_entropy > best_color_filter_entropy * COLOR_FILTER_EARLY_EXIT_RATIO)
                    continue;

                  cand.predictor_signal = std::move(base_signal);
                  for (std::size_t c = 0; c < component_count; ++c)
                    compute_per_block_prediction(ctx, cand.predictor_signal[c], predictor_planes_i16[c], mode, cand.residuals[c]);

                  const double predictor_entropy = compute_squared_entropy(cand.residuals);
                  if (best_prediction_entropy < std::numeric_limits<double>::infinity() &&
                      predictor_entropy > best_prediction_entropy * PER_BLOCK_PREDICTION_EARLY_EXIT_RATIO)
                    continue;
                  if (predictor_entropy < best_prediction_entropy)
                    best_prediction_entropy = predictor_entropy;

                  if (color_filter_entropy < best_color_filter_entropy)
                    best_color_filter_entropy = color_filter_entropy;
                }
              }

              const auto base_residuals = cand.residuals;
              std::vector<std::vector<int16_t>> best_residuals;
              int best_diff_index = 0;
              uint64_t best_bits = std::numeric_limits<uint64_t>::max();

              for (int diff_idx = 0; diff_idx < static_cast<int>(DiffFilterType::Count); ++diff_idx)
              {
                const DiffFilterType diff_type = static_cast<DiffFilterType>(diff_idx);
                std::vector<std::vector<int16_t>> diff_residuals(component_count);
                for (std::size_t c = 0; c < component_count; ++c)
                {
                  if (diff_type == DiffFilterType::None)
                    diff_residuals[c] = base_residuals[c];
                  else
                    diff_residuals[c] = apply_diff_filter(ctx, diff_type, base_residuals[c]);
                }

                uint64_t total_bits = 0;
                for (std::size_t c = 0; c < component_count; ++c)
                {
                  std::vector<int16_t> tmp = diff_residuals[c];
                  if (is_full_block)
                    reorder_to_hilbert(tmp);
                  total_bits += estimate_residual_bits(tmp, c);
                  if (total_bits >= best_bits)
                    break;
                }

                if (total_bits < best_bits)
                {
                  best_bits = total_bits;
                  best_diff_index = diff_idx;
                  best_residuals = std::move(diff_residuals);
                }
              }

              cand.residuals = std::move(best_residuals);
              cand.bit_cost = best_bits;
              cand.diff_filter_index = best_diff_index;

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

            const side_info block_sideinfo{
                best_candidate.filter_code,
                best_candidate.mode,
                best_candidate.diff_filter_index};
            const uint16_t packed_sideinfo = pack_block_sideinfo(block_sideinfo);
            chunk_sideinfo.push_back(packed_sideinfo);

            for (std::size_t c = 0; c < component_count; ++c)
            {
              std::vector<int16_t> residual = std::move(best_candidate.residuals[c]);
              if (dump_file && dump_before_hilbert)
                dump_residual_block(dump_file.get(), c, ctx, residual, block_sideinfo, best_candidate.bit_cost);
              if (is_full_block)
                reorder_to_hilbert(residual);
              if (dump_file && dump_after_hilbert)
                dump_residual_block(dump_file.get(), c, ctx, residual, block_sideinfo, best_candidate.bit_cost);
              chunk_residuals[c].insert(chunk_residuals[c].end(), residual.begin(), residual.end());
            }

            if (order == PipelineOrder::PredictorThenFilter)
            {
              for (std::size_t c = 0; c < component_count; ++c)
                store_block_to_plane(predictor_planes_u8[c], ctx.x0, ctx.y0, ctx.bw, ctx.bh, block_values[c]);
            }
            else
            {
              if (best_candidate.predictor_signal.size() != component_count)
              {
                err = "tlg7: missing predictor signal";
                return false;
              }
              for (std::size_t c = 0; c < component_count; ++c)
                store_block_to_plane(predictor_planes_i16[c], ctx.x0, ctx.y0, ctx.bw, ctx.bh, best_candidate.predictor_signal[c]);
            }
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

        const uint32_t sideinfo_bit_length = static_cast<uint32_t>(chunk_sideinfo.size() * SIDEINFO_BITS_PER_BLOCK);
        tlg::detail::write_u32le(fp, sideinfo_bit_length);
        if (sideinfo_bit_length)
        {
          const std::size_t sideinfo_byte_count = (sideinfo_bit_length + 7u) / 8u;
          std::vector<uint8_t> sideinfo_bytes(sideinfo_byte_count, 0);
          std::size_t byte_pos = 0;
          int bit_pos = 0;
          for (std::size_t i = 0; i < chunk_sideinfo.size(); ++i)
          {
            const uint16_t value = chunk_sideinfo[i];
            for (int bit = 0; bit < SIDEINFO_BITS_PER_BLOCK; ++bit)
            {
              if (byte_pos >= sideinfo_bytes.size())
              {
                err = "tlg7: sideinfo buffer overflow";
                return false;
              }
              const int bit_value = (value >> bit) & 1;
              if (bit_value)
                sideinfo_bytes[byte_pos] |= static_cast<uint8_t>(1u << bit_pos);
              ++bit_pos;
              if (bit_pos == 8)
              {
                bit_pos = 0;
                ++byte_pos;
              }
            }
          }
          if (!sideinfo_bytes.empty() && fwrite(sideinfo_bytes.data(), 1, sideinfo_bytes.size(), fp) != sideinfo_bytes.size())
          {
            err = "tlg7: write sideinfo stream";
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

      return true;
    }

  } // namespace enc

} // namespace tlg::v7
