#pragma once

#include <array>
#include <cstdint>

#include "tlg8_block.h"

namespace tlg::v8::enc
{
  enum class ReorderPattern : uint32_t
  {
    Hilbert = 0,    // ヒルベルト曲線
    ZigzagDiag,     // 左上から右下へのジグザグ
    ZigzagAntiDiag, // 右上から左下へのジグザグ
    ZigzagHorz,     // 横方向ジグザグ
    ZigzagVert,     // 縦方向ジグザグ
    ZigzagNNESSW,   // ジグザグ：北北東から南南西
    ZigzagNEESWW,   // ジグザグ：北東東から南西西
    ZigzagNWWSEE,   // ジグザグ：北西西から南西西
    Count,
  };

  constexpr uint32_t kReorderPatternCount = static_cast<uint32_t>(ReorderPattern::Count);

  // 8x8 ブロックの輝度（Luma）64値に対して、各リオーダーパターンの
  // 「スキャン順の滑らかさ（Total Variation の平均）」を計算する。
  //
  // 出力 reorder_tv[i] は、ReorderPattern の enum 値（0..7）順に対応する。
  // 値は「隣接差分 |Δ| の平均」（= sum(|Δ|)/63）で、スケールは入力 luma と同じ。
  std::array<float, kReorderPatternCount> compute_reorder_tv_mean(
      const std::array<float, kMaxBlockPixels> &luma);

  // カラーフィルター後の係数を指定した走査順へ並び替える。
  void reorder_to_scan(component_colors &colors,
                       uint32_t components,
                       uint32_t block_w,
                       uint32_t block_h,
                       ReorderPattern pattern);

  // エントロピー復号後の係数を元の走査順へ戻す。
  void reorder_from_scan(component_colors &colors,
                         uint32_t components,
                         uint32_t block_w,
                         uint32_t block_h,
                         ReorderPattern pattern);
}
