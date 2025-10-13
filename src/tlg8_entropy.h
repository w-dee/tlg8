#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

#include "tlg8_block.h"
#include "tlg8_bit_io.h"

namespace tlg::v8::enc
{
  namespace adaptation
  {
    // 適応パラメータを共有するための定数と関数。
    inline constexpr int kAParameterShift = 2;
    inline constexpr int kAParameterBias = 1 << (kAParameterShift - 1);

    inline constexpr int reduce_a(int a)
    {
      return (a + kAParameterBias) >> kAParameterShift;
    }

    inline constexpr int mix_a_m(int a, int m, int m_raise_ratio, int m_fall_ratio)
    {
      if (a < m)
        return ((m << kAParameterShift) * m_raise_ratio + a * (16 - m_raise_ratio) + 8) >> 4;
      return ((m << kAParameterShift) * m_fall_ratio + a * (16 - m_fall_ratio) + 8) >> 4;
    }
  }

  // ゴロム符号はプレーン／ランレングスの 2 種類を使用する。
  enum class GolombCodingKind : uint8_t
  {
    Plain = 0,
    RunLength = 1,
  };

  constexpr uint32_t kNumEntropyEncoders = 2;
  constexpr uint32_t kGolombRowCount = 8;
  constexpr uint32_t kGolombColumnCount = 9;
  constexpr uint32_t kGolombRowSum = 1024;

  constexpr uint32_t kInterleavedComponentIndex = 0xFFFFFFFFu;
  constexpr uint32_t kInterleavedPlainRow = 6;
  constexpr uint32_t kInterleavedRunLengthRow = 7;

  using golomb_histogram = std::array<std::array<uint64_t, kGolombColumnCount>, kGolombRowCount>;
  using golomb_table_counts = std::array<std::array<uint16_t, kGolombColumnCount>, kGolombRowCount>;
  using golomb_ratio_array = std::array<uint8_t, kGolombRowCount>;

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

  // ゴロム予測ダンプ用のファイルポインタを設定する。
  void set_golomb_prediction_dump_file(FILE *fp);

  bool rebuild_golomb_table_from_histogram(const golomb_histogram &histogram);
  const golomb_table_counts &current_golomb_table();
  const golomb_ratio_array &current_m_raise_ratios();
  const golomb_ratio_array &current_m_fall_ratios();
  bool apply_golomb_table(const golomb_table_counts &table,
                          const golomb_ratio_array &raise_ratios,
                          const golomb_ratio_array &fall_ratios);
  bool apply_golomb_table(const golomb_table_counts &table);
  bool is_golomb_table_overridden();
  uint64_t estimate_row_bits(GolombCodingKind kind,
                             uint32_t component,
                             const int16_t *values,
                             uint32_t count);
}
