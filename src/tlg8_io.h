#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "image_io.h"

namespace tlg::v8
{
  struct DumpContext
  {
    struct TrainingJsonState
    {
      FILE *file = nullptr;
      std::string path;

      bool enabled() const noexcept
      {
        return file != nullptr;
      }
    } training_dump;

    std::string image_tag;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t components = 0;
    struct LabelCacheState
    {
      FILE *file = nullptr;
      std::string bin_path;
      std::string meta_path;
      std::vector<std::string> input_paths;
      uint64_t record_count = 0;
    } label_cache;
    struct FeatureStatsState
    {
      std::string path;
      std::vector<double> sum;
      std::vector<double> sumsq;
      uint64_t count = 0;

      bool enabled() const noexcept
      {
        return !path.empty();
      }
    } feature_stats;

    bool enable_features = false; // DumpMode に基づき特徴量系を有効化するか
    bool enable_labels = false;   // DumpMode に基づきラベル系を有効化するか

    bool wants_training_dump() const noexcept
    {
      return training_dump.enabled();
    }

    bool wants_feature_stats() const noexcept
    {
      return enable_features && feature_stats.enabled();
    }

    bool wants_feature_pixels() const noexcept
    {
      return wants_training_dump() || wants_feature_stats();
    }

    bool wants_label_cache() const noexcept
    {
      return enable_labels && label_cache.file != nullptr;
    }

    bool has_any_output() const noexcept
    {
      return training_dump.enabled() || label_cache.file != nullptr || feature_stats.enabled();
    }
  };

  constexpr std::size_t kLabelRecordSize = 128;
  constexpr std::size_t kFeatureVectorSize = 4 * 8 * 8 + 3;

  bool configure_golomb_table(const std::string &path, std::string &err);
  bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err);

  namespace enc
  {
    bool write_raw(FILE *fp,
                   const PixelBuffer &src,
                   int desired_colors,
                   const std::string &dump_residuals_path,
                   TlgOptions::DumpResidualsOrder dump_residuals_order,
                   const std::string &dump_golomb_prediction_path,
                   const std::string &reorder_histogram_path,
                   const std::string &residual_bmp_path,
                   TlgOptions::DumpResidualsOrder residual_bmp_order,
                   double residual_bmp_emphasis,
                   TlgOptions::DumpMode dump_mode,
                   const std::string &training_dump_path,
                   const std::string &training_image_tag,
                   const std::string &training_stats_path,
                   const std::string &label_cache_bin_path,
                   const std::string &label_cache_meta_path,
                   bool force_hilbert_reorder,
                   int force_entropy,
                   std::string &err,
                   uint64_t *out_entropy_bits = nullptr);
  }
} // namespace tlg::v8
