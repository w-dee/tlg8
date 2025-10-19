#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct LabelRecord
{
  uint32_t magic = 0x4C424C38u;
  uint16_t version = 1u;
  uint16_t reserved = 0u;
  int16_t labels[12] = {};
  uint32_t crc32 = 0u;
  uint8_t padding[92] = {};
};
#pragma pack(pop)

static_assert(sizeof(LabelRecord) == 128, "LabelRecord は 128 バイトである必要があります");

inline void split_filter(int code, int16_t &perm, int16_t &primary, int16_t &secondary)
{
  if (code < 0)
  {
    perm = static_cast<int16_t>(-1);
    primary = static_cast<int16_t>(-1);
    secondary = static_cast<int16_t>(-1);
    return;
  }
  perm = static_cast<int16_t>(((code >> 4) & 0x7) % 6);
  primary = static_cast<int16_t>(((code >> 2) & 0x3) % 4);
  secondary = static_cast<int16_t>((code & 0x3) % 4);
}
