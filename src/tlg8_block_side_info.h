#pragma once

#include <cstdint>

namespace tlg::v8::enc
{
  // ブロック単位のサイド情報を送る際のエンコード方式。
  enum class BlockChoiceEncoding : uint32_t
  {
    Raw = 0,              // 各ブロックごとにそのまま書き出す
    SameAsPrevious = 1,   // 直前と同じかどうかをビットで示す
    RunLength = 2,        // ランレングス圧縮を適用する
    Count
  };
}

