#include "sha256.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace
{
  constexpr std::array<uint32_t, 64> kRoundConstants = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

  inline uint32_t rotr(uint32_t value, uint32_t bits)
  {
    return (value >> bits) | (value << (32u - bits));
  }
}

Sha256::Sha256()
{
  state_[0] = 0x6a09e667u;
  state_[1] = 0xbb67ae85u;
  state_[2] = 0x3c6ef372u;
  state_[3] = 0xa54ff53au;
  state_[4] = 0x510e527fu;
  state_[5] = 0x9b05688cu;
  state_[6] = 0x1f83d9abu;
  state_[7] = 0x5be0cd19u;
}

void Sha256::update(const void *data, std::size_t length)
{
  if (finished_)
    throw std::logic_error("SHA-256 は finish 後に update できません");

  const uint8_t *ptr = static_cast<const uint8_t *>(data);
  bit_length_ += static_cast<std::uint64_t>(length) * 8u;

  while (length > 0)
  {
    const std::size_t to_copy = std::min<std::size_t>(length, buffer_.size() - buffer_size_);
    std::memcpy(buffer_.data() + buffer_size_, ptr, to_copy);
    buffer_size_ += to_copy;
    ptr += to_copy;
    length -= to_copy;
    if (buffer_size_ == buffer_.size())
    {
      transform(buffer_.data());
      buffer_size_ = 0;
    }
  }
}

std::array<uint8_t, 32> Sha256::finish()
{
  if (!finished_)
  {
    buffer_[buffer_size_] = 0x80u;
    ++buffer_size_;

    if (buffer_size_ > 56)
    {
      std::fill(buffer_.begin() + buffer_size_, buffer_.end(), 0u);
      transform(buffer_.data());
      buffer_size_ = 0;
    }

    std::fill(buffer_.begin() + buffer_size_, buffer_.begin() + 56, 0u);

    std::array<uint8_t, 8> length_bytes{};
    for (int i = 0; i < 8; ++i)
      length_bytes[7 - i] = static_cast<uint8_t>((bit_length_ >> (i * 8)) & 0xffu);
    std::memcpy(buffer_.data() + 56, length_bytes.data(), length_bytes.size());
    transform(buffer_.data());

    for (std::size_t i = 0; i < state_.size(); ++i)
    {
      digest_[i * 4 + 0] = static_cast<uint8_t>((state_[i] >> 24) & 0xffu);
      digest_[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xffu);
      digest_[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xffu);
      digest_[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xffu);
    }
    finished_ = true;
  }
  return digest_;
}

std::string Sha256::hexdigest()
{
  auto bytes = finish();
  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result;
  result.resize(bytes.size() * 2);
  for (std::size_t i = 0; i < bytes.size(); ++i)
  {
    result[i * 2 + 0] = kHexDigits[(bytes[i] >> 4) & 0x0fu];
    result[i * 2 + 1] = kHexDigits[bytes[i] & 0x0fu];
  }
  return result;
}

void Sha256::transform(const uint8_t block[64])
{
  std::array<uint32_t, 64> w{};
  for (int i = 0; i < 16; ++i)
  {
    w[static_cast<std::size_t>(i)] =
        (static_cast<uint32_t>(block[i * 4 + 0]) << 24) |
        (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
        (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
        (static_cast<uint32_t>(block[i * 4 + 3]));
  }
  for (int i = 16; i < 64; ++i)
  {
    const uint32_t s0 = rotr(w[static_cast<std::size_t>(i - 15)], 7) ^
                        rotr(w[static_cast<std::size_t>(i - 15)], 18) ^
                        (w[static_cast<std::size_t>(i - 15)] >> 3);
    const uint32_t s1 = rotr(w[static_cast<std::size_t>(i - 2)], 17) ^
                        rotr(w[static_cast<std::size_t>(i - 2)], 19) ^
                        (w[static_cast<std::size_t>(i - 2)] >> 10);
    w[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i - 16)] + s0 + w[static_cast<std::size_t>(i - 7)] + s1;
  }

  uint32_t a = state_[0];
  uint32_t b = state_[1];
  uint32_t c = state_[2];
  uint32_t d = state_[3];
  uint32_t e = state_[4];
  uint32_t f = state_[5];
  uint32_t g = state_[6];
  uint32_t h = state_[7];

  for (int i = 0; i < 64; ++i)
  {
    const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + s1 + ch + kRoundConstants[static_cast<std::size_t>(i)] + w[static_cast<std::size_t>(i)];
    const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = s0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}
