#include "tlg8_interleave.h"

#include <array>

namespace tlg::v8::enc
{
  namespace
  {
    using buffer_type = std::array<int16_t, kMaxBlockPixels * 4>;

    buffer_type make_component_buffer(const component_colors &colors,
                                      uint32_t used_components,
                                      uint32_t value_count)
    {
      buffer_type buffer{};
      for (uint32_t comp = 0; comp < used_components; ++comp)
      {
        for (uint32_t i = 0; i < value_count; ++i)
          buffer[comp * value_count + i] = colors.values[comp][i];
      }
      return buffer;
    }
  }

  void apply_interleave_filter(InterleaveFilter filter,
                               component_colors &colors,
                               uint32_t components,
                               uint32_t value_count)
  {
    if (filter != InterleaveFilter::Interleave)
      return;
    if (value_count == 0)
      return;

    const uint32_t used_components = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    if (used_components <= 1)
      return;

    const auto source = make_component_buffer(colors, used_components, value_count);
    std::array<int16_t, kMaxBlockPixels * 4> interleaved{};

    for (uint32_t i = 0; i < value_count; ++i)
    {
      for (uint32_t comp = 0; comp < used_components; ++comp)
      {
        const uint32_t src_index = comp * value_count + i;
        const uint32_t dst_index = i * used_components + comp;
        interleaved[dst_index] = source[src_index];
      }
    }

    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      for (uint32_t i = 0; i < value_count; ++i)
      {
        const uint32_t index = comp * value_count + i;
        colors.values[comp][i] = interleaved[index];
      }
      for (uint32_t i = value_count; i < kMaxBlockPixels; ++i)
        colors.values[comp][i] = 0;
    }
  }

  void undo_interleave_filter(InterleaveFilter filter,
                              component_colors &colors,
                              uint32_t components,
                              uint32_t value_count)
  {
    if (filter != InterleaveFilter::Interleave)
      return;
    if (value_count == 0)
      return;

    const uint32_t used_components = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    if (used_components <= 1)
      return;

    const auto interleaved = make_component_buffer(colors, used_components, value_count);

    for (uint32_t comp = 0; comp < used_components; ++comp)
    {
      for (uint32_t i = 0; i < value_count; ++i)
      {
        const uint32_t src_index = i * used_components + comp;
        colors.values[comp][i] = interleaved[src_index];
      }
      for (uint32_t i = value_count; i < kMaxBlockPixels; ++i)
        colors.values[comp][i] = 0;
    }
  }
}
