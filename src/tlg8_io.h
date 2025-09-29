#pragma once

#include <cstdio>
#include <string>

#include "image_io.h"

namespace tlg::v8
{
  bool configure_golomb_table(const std::string &path, std::string &err);
  bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err);

  namespace enc
  {
    bool write_raw(FILE *fp, const PixelBuffer &src, int desired_colors, std::string &err);
  }
} // namespace tlg::v8

