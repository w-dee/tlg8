#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "tlg8_block.h"
#include "tlg8_bit_io.h"

namespace tlg::v8::enc
{
  // ゴロム符号はプレーン／ランレングスの 2 種類を使用する。
  enum class GolombCodingKind : uint8_t
  {
    Plain = 0,
    RunLength = 1,
  };

  constexpr uint32_t kNumEntropyEncoders = 2;

  struct entropy_encoder
  {
    using estimate_fn = uint64_t (*)(const component_colors &, uint32_t components, uint32_t value_count);
    using encode_fn = bool (*)(detail::bitio::BitWriter &,
                               const component_colors &,
                               uint32_t components,
                               uint32_t value_count,
                               std::string &err);

    GolombCodingKind kind;
    estimate_fn estimate_bits;
    encode_fn encode_block;
  };

  const std::array<entropy_encoder, kNumEntropyEncoders> &entropy_encoder_table();

  bool decode_block(detail::bitio::BitReader &reader,
                    GolombCodingKind kind,
                    uint32_t components,
                    uint32_t value_count,
                    component_colors &out,
                    std::string &err);
}
