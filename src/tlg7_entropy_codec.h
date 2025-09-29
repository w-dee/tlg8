#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tlg::v7
{

enum class GolombCodingKind : std::uint8_t
{
  RunLength = 0,
  Plain = 1,
};

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

  void set_component_index(std::size_t index) noexcept { component_index_ = index; }
  [[nodiscard]] std::size_t component_index() const noexcept { return component_index_; }

  void encode(const std::vector<int16_t> &residuals,
              std::vector<uint8_t> &out,
              uint32_t &bit_length) override;

  [[nodiscard]] std::uint64_t estimate_bits(const std::vector<int16_t> &residuals) const;

  [[nodiscard]] GolombCodingKind last_mode() const noexcept { return last_mode_; }

private:
  std::size_t component_index_ = 0;
  GolombCodingKind last_mode_ = GolombCodingKind::RunLength;
};

class GolombResidualEntropyDecoder : public ResidualEntropyDecoder
{
public:
  GolombResidualEntropyDecoder();
  ~GolombResidualEntropyDecoder() override;

  void set_component_index(std::size_t index) noexcept { component_index_ = index; }
  [[nodiscard]] std::size_t component_index() const noexcept { return component_index_; }

  bool decode(const uint8_t *data,
              std::size_t size,
              std::size_t expected_count,
              std::vector<int16_t> &out) override;

  void set_coding_kind(GolombCodingKind kind) noexcept { coding_kind_ = kind; }
  [[nodiscard]] GolombCodingKind coding_kind() const noexcept { return coding_kind_; }

  void init_stream(const uint8_t *data, std::size_t size);
  bool decode_next(std::size_t expected_count, std::vector<int16_t> &out);
  [[nodiscard]] bool stream_consumed() const noexcept;

private:
  std::size_t component_index_ = 0;
  GolombCodingKind coding_kind_ = GolombCodingKind::RunLength;
  bool decoder_initialized_ = false;
  struct Impl;
  std::unique_ptr<Impl> decoder_impl_;
};

bool configure_golomb_table(const std::string &path, std::string &err);

} // namespace tlg::v7
