#pragma once

#include <cstdio>
#include <string>

#include "image_io.h"

namespace tlg::v8
{
  struct TrainingDumpContext
  {
    FILE *file = nullptr;
    std::string image_tag;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t components = 0;
  };

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
                   const std::string &training_dump_path,
                   const std::string &training_image_tag,
                   bool force_hilbert_reorder,
                   std::string &err,
                   uint64_t *out_entropy_bits = nullptr);
  }
} // namespace tlg::v8

