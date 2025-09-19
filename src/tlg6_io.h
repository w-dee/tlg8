#pragma once

#include <cstdio>
#include <string>

#include "image_io.h"

namespace tlg::v6 {

bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err);

namespace enc {
bool write_raw(FILE *fp, const PixelBuffer &src, int colors, std::string &err);
}

} // namespace tlg::v6

