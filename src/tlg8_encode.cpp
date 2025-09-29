#include "tlg8_bit_io.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tlg::v8::enc
{
  bool encode_for_tile(detail::bitio::BitWriter &writer,
                       const uint8_t *row_ptr,
                       uint32_t tile_w,
                       uint32_t components,
                       std::string &err)
  {
    for (uint32_t dx = 0; dx < tile_w; ++dx)
    {
      const size_t pixel_base = static_cast<size_t>(dx) * components;
      for (uint32_t c = 0; c < components; ++c)
      {
        if (!writer.write_u8(row_ptr[pixel_base + c]))
        {
          err = "tlg8: tile buffer overflow";
          return false;
        }
      }
    }
    return true;
  }
}
