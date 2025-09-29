#include "tlg8_bit_io.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tlg::v8
{
  bool decode_for_tile(detail::bitio::BitReader &reader,
                       uint32_t tile_w,
                       uint32_t components,
                       size_t row_offset,
                       std::vector<uint8_t> &decoded,
                       std::string &err)
  {
    for (uint32_t dx = 0; dx < tile_w; ++dx)
    {
      for (uint32_t c = 0; c < components; ++c)
      {
        uint8_t value = 0;
        if (!reader.read_u8(value))
        {
          err = "tlg8: tile payload truncated";
          return false;
        }
        decoded[row_offset + static_cast<size_t>(dx) * components + c] = value;
      }
    }
    return true;
  }
}
