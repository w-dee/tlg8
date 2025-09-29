#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "tlg8_block.h"
#include "tlg8_bit_io.h"

namespace tlg::v8::enc
{
  constexpr uint32_t kNumEntropyEncoders = 2;

  // 本来は「ランレングス＋ゴロム」など別種のエントロピー符号器を
  // 2 系統実装する予定だが、現段階ではダミー実装でビット列をそのまま
  // 吐き出すのみとし、インターフェースだけを先行して定義している。
  struct entropy_encoder
  {
    using estimate_fn = uint64_t (*)(const component_colors &, uint32_t components, uint32_t value_count);
    using encode_fn = bool (*)(detail::bitio::BitWriter &, const component_colors &, uint32_t components, uint32_t value_count, std::string &err);

    estimate_fn estimate_bits;
    encode_fn encode_block;
  };

  const std::array<entropy_encoder, kNumEntropyEncoders> &entropy_encoder_table();
}
