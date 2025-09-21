#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tlg::v7
{

class ResidualEntropyEncoder
{
public:
  virtual ~ResidualEntropyEncoder() = default;

  virtual void encode(const std::vector<int16_t> &residuals,
                      std::vector<uint8_t> &out,
                      uint32_t &bit_length) = 0;
};

class ResidualEntropyDecoder
{
public:
  virtual ~ResidualEntropyDecoder() = default;

  virtual bool decode(const uint8_t *data,
                      std::size_t size,
                      std::size_t expected_count,
                      std::vector<int16_t> &out) = 0;
};

class GolombResidualEntropyEncoder : public ResidualEntropyEncoder
{
public:
  GolombResidualEntropyEncoder() = default;

  void encode(const std::vector<int16_t> &residuals,
              std::vector<uint8_t> &out,
              uint32_t &bit_length) override;

  [[nodiscard]] std::uint64_t estimate_bits(const std::vector<int16_t> &residuals) const;
};

class GolombResidualEntropyDecoder : public ResidualEntropyDecoder
{
public:
  GolombResidualEntropyDecoder() = default;

  bool decode(const uint8_t *data,
              std::size_t size,
              std::size_t expected_count,
              std::vector<int16_t> &out) override;
};

bool configure_golomb_table(const std::string &path, std::string &err);

} // namespace tlg::v7

