#pragma once

#include "tlg8_bit_io.h"

#include <cstdint>

namespace tlg::v8::detail::bitio
{
  // 7ビット単位で値を符号化する可変長符号。上位ビットは継続フラグとする。
  inline void put_varuint(BitWriter &writer, uint32_t value)
  {
    while (true)
    {
      const uint32_t chunk = value & 0x7Fu;
      value >>= 7;
      if (value != 0)
      {
        writer.put_upto8(chunk | 0x80u, 8);
      }
      else
      {
        writer.put_upto8(chunk, 8);
        break;
      }
    }
  }

  // put_varuint と対になる読み出し。失敗した場合は false を返す。
  inline bool get_varuint(BitReader &reader, uint32_t &value)
  {
    value = 0;
    uint32_t shift = 0;
    for (int i = 0; i < 5; ++i)
    {
      const uint32_t byte = reader.get(8);
      value |= (byte & 0x7Fu) << shift;
      if ((byte & 0x80u) == 0)
        return true;
      shift += 7;
      if (shift >= 32)
        return false;
    }
    return false;
  }

  // varuint を書き出したときに必要となるビット数を返す。
  inline uint64_t varuint_bits(uint32_t value)
  {
    uint64_t bits = 0;
    do
    {
      bits += 8;
      value >>= 7;
    } while (value != 0);
    return bits;
  }
}

