#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

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
  constexpr uint32_t kEntropyContextCount = 8; // Plain×RGBA(4) + RunLength×RGBA(4)

  // プレーンなビット列を保持するコンテキスト。ブロック毎ではなく、
  // タイル全体で連結して使用する。
  struct entropy_context_stream
  {
    std::vector<uint8_t> data{};
    uint8_t current_byte = 0;
    uint8_t bit_count = 0;
  };

  struct entropy_encode_context
  {
    std::array<entropy_context_stream, kEntropyContextCount> streams{};
  };

  struct entropy_decode_stream
  {
    std::vector<uint8_t> data{};
    size_t byte_pos = 0;
    uint32_t bit_buffer = 0;
    uint8_t bits_available = 0;
  };

  struct entropy_decode_context
  {
    std::array<entropy_decode_stream, kEntropyContextCount> streams{};
  };

  struct entropy_encoder
  {
    using estimate_fn = uint64_t (*)(const component_colors &, uint32_t components, uint32_t value_count);
    using encode_fn = bool (*)(entropy_encode_context &, const component_colors &, uint32_t components, uint32_t value_count,
                               std::string &err);

    GolombCodingKind kind;
    estimate_fn estimate_bits;
    encode_fn encode_block;
  };

  const std::array<entropy_encoder, kNumEntropyEncoders> &entropy_encoder_table();

  bool flush_entropy_contexts(entropy_encode_context &ctx, detail::bitio::BitWriter &writer, std::string &err);
  bool load_entropy_contexts(detail::bitio::BitReader &reader, entropy_decode_context &ctx, std::string &err);
  bool decode_block_from_context(entropy_decode_context &ctx,
                                 GolombCodingKind kind,
                                 uint32_t components,
                                 uint32_t value_count,
                                 component_colors &out,
                                 std::string &err);
}

