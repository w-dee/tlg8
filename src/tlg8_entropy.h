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
  constexpr uint32_t kGolombRowCount = 6;

  struct entropy_encoder
  {
    using estimate_fn = uint64_t (*)(const component_colors &,
                                     uint32_t components,
                                     uint32_t value_count,
                                     bool uses_interleave);
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

  int golomb_row_index(GolombCodingKind kind, uint32_t component);
  GolombCodingKind golomb_row_kind(uint32_t row);
  uint32_t golomb_row_component(uint32_t row);

  bool encode_values(detail::bitio::BitWriter &writer,
                     GolombCodingKind kind,
                     uint32_t component,
                     const int16_t *values,
                     uint32_t count,
                     std::string &err);

  bool decode_values(detail::bitio::BitReader &reader,
                     GolombCodingKind kind,
                     uint32_t component,
                     uint32_t count,
                     int16_t *dst,
                     std::string &err);

  bool decode_block(detail::bitio::BitReader &reader,
                    GolombCodingKind kind,
                    uint32_t components,
                    uint32_t value_count,
                    component_colors &out,
                    std::string &err);
}
