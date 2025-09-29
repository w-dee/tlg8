#include "tlg8_entropy.h"

#include <array>

namespace
{
  uint64_t estimate_dumb(const tlg::v8::enc::component_colors &colors,
                         uint32_t components,
                         uint32_t value_count)
  {
    (void)colors;
    return static_cast<uint64_t>(components) * static_cast<uint64_t>(value_count) * 16u;
  }

  bool encode_dumb(tlg::v8::detail::bitio::BitWriter &writer,
                   const tlg::v8::enc::component_colors &colors,
                   uint32_t components,
                   uint32_t value_count,
                   std::string &err)
  {
    for (uint32_t comp = 0; comp < components; ++comp)
    {
      for (uint32_t i = 0; i < value_count; ++i)
      {
        const int16_t v = colors.values[comp][i];
        if (!writer.write_u16_le(static_cast<uint16_t>(v)))
        {
          err = "tlg8: ビットストリームへの書き込みに失敗しました";
          return false;
        }
      }
    }
    return true;
  }

  constexpr std::array<tlg::v8::enc::entropy_encoder, tlg::v8::enc::kNumEntropyEncoders> kEncoders = {
      tlg::v8::enc::entropy_encoder{&estimate_dumb, &encode_dumb},
      tlg::v8::enc::entropy_encoder{&estimate_dumb, &encode_dumb}};
}

namespace tlg::v8::enc
{
  const std::array<entropy_encoder, kNumEntropyEncoders> &entropy_encoder_table()
  {
    return kEncoders;
  }
}
