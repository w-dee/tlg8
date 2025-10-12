#include "tlg8_interleave.h"

#include <algorithm>
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

    bool should_interleave(InterleaveFilter filter,
                           uint32_t value_count,
                           uint32_t used_components)
    {
      return filter == InterleaveFilter::Interleave && value_count > 0 && used_components > 1;
    }

    template <typename IndexProvider>
    void write_components(component_colors &colors,
                          const buffer_type &buffer,
                          uint32_t used_components,
                          uint32_t value_count,
                          const IndexProvider &provider)
    {
      for (uint32_t comp = 0; comp < used_components; ++comp)
      {
        auto &component = colors.values[comp];
        for (uint32_t i = 0; i < value_count; ++i)
          component[i] = buffer[provider(comp, i)];
        std::fill(component.begin() + value_count, component.end(), 0);
      }
    }
  }

  void apply_interleave_filter(InterleaveFilter filter,
                               component_colors &colors,
                               uint32_t components,
                               uint32_t value_count)
  {
    const uint32_t used_components = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    if (!should_interleave(filter, value_count, used_components))
      return;

    const auto source = make_component_buffer(colors, used_components, value_count);
    write_components(
        colors,
        source,
        used_components,
        value_count,
        [used_components, value_count](uint32_t comp, uint32_t i) {
          const uint32_t index = comp * value_count + i;
          const uint32_t src_component = index % used_components;
          const uint32_t src_offset = index / used_components;
          return src_component * value_count + src_offset;
        });
  }

  void undo_interleave_filter(InterleaveFilter filter,
                              component_colors &colors,
                              uint32_t components,
                              uint32_t value_count)
  {
    const uint32_t used_components = std::min<uint32_t>(components, static_cast<uint32_t>(colors.values.size()));
    if (!should_interleave(filter, value_count, used_components))
      return;

    const auto source = make_component_buffer(colors, used_components, value_count);
    write_components(
        colors,
        source,
        used_components,
        value_count,
        [used_components](uint32_t comp, uint32_t i) {
          return i * used_components + comp;
        });
  }
}
