#pragma once

#include <cstdio>
#include <string>

#include "image_io.h"

namespace tlg::v5
{

    bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err);
    bool write_raw(FILE *fp, const PixelBuffer &src, int desired_colors, std::string &err);

} // namespace tlg::v5
