#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

// SHA-256 ハッシュを計算するための簡易クラス。
class Sha256
{
public:
  Sha256();

  // データをハッシュへ追加する。
  void update(const void *data, std::size_t length);

  // バイト配列としてハッシュ値を取得する。finish を複数回呼んでも同じ値を返す。
  std::array<uint8_t, 32> finish();

  // ハッシュ値を 16 進文字列として取得する。
  std::string hexdigest();

private:
  void transform(const uint8_t block[64]);

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> buffer_{};
  std::uint64_t bit_length_ = 0;
  std::size_t buffer_size_ = 0;
  bool finished_ = false;
  std::array<uint8_t, 32> digest_{};
};
