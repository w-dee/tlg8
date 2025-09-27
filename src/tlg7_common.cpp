#include "tlg7_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tlg_io_common.h"

namespace tlg::v7::detail
{

  std::vector<image<uint8_t>> split_components_from_gray(const image<uint8_t> &gray)
  {
    std::vector<image<uint8_t>> planes;
    if (!gray.empty())
      planes.push_back(gray);
    return planes;
  }

  std::vector<image<uint8_t>> split_components_from_packed(const image<uint32_t> &packed,
                                                           std::size_t component_count)
  {
    std::vector<image<uint8_t>> planes;
    if (packed.empty() || component_count == 0)
      return planes;

    const std::size_t width = packed.get_width();
    const std::size_t height = packed.get_height();
    planes.reserve(component_count);
    for (std::size_t i = 0; i < component_count; ++i)
      planes.emplace_back(width, height);

    for (std::size_t y = 0; y < height; ++y)
    {
      const uint32_t *src_row = packed.row_ptr(y);
      for (std::size_t x = 0; x < width; ++x)
      {
        const uint32_t v = src_row[x];
        if (component_count >= 1)
          planes[0].row_ptr(y)[x] = static_cast<uint8_t>(v & 0xFF); // B
        if (component_count >= 2)
          planes[1].row_ptr(y)[x] = static_cast<uint8_t>((v >> 8) & 0xFF); // G
        if (component_count >= 3)
          planes[2].row_ptr(y)[x] = static_cast<uint8_t>((v >> 16) & 0xFF); // R
        if (component_count >= 4)
          planes[3].row_ptr(y)[x] = static_cast<uint8_t>((v >> 24) & 0xFF); // A
      }
    }
    return planes;
  }

} // namespace tlg::v7::detail

namespace tlg::v7
{

  BlockContext make_block_context(std::size_t block_x,
                                  std::size_t block_y,
                                  std::size_t width,
                                  std::size_t height,
                                  std::size_t blocks_x)
  {
    BlockContext ctx;
    ctx.x0 = block_x * BLOCK_SIZE;
    ctx.y0 = block_y * BLOCK_SIZE;
    ctx.bw = std::min<std::size_t>(BLOCK_SIZE, width - ctx.x0);
    ctx.bh = std::min<std::size_t>(BLOCK_SIZE, height - ctx.y0);
    ctx.index = block_y * blocks_x + block_x;
    return ctx;
  }

  namespace
  {
    constexpr std::array<std::array<uint8_t, 3>, COLOR_FILTER_PERMUTATIONS> kColorPermutations = {
        std::array<uint8_t, 3>{0, 1, 2},
        std::array<uint8_t, 3>{0, 2, 1},
        std::array<uint8_t, 3>{1, 0, 2},
        std::array<uint8_t, 3>{1, 2, 0},
        std::array<uint8_t, 3>{2, 0, 1},
        std::array<uint8_t, 3>{2, 1, 0},
    };

    struct ColorFilterParams
    {
      int perm = 0;
      int primary = 0;
      int secondary = 0;
    };

    [[maybe_unused]] constexpr int make_color_filter_code(int perm, int primary, int secondary)
    {
      return (perm << 4) | (primary << 2) | secondary;
    }

    inline ColorFilterParams decode_color_filter_code(int code)
    {
      if (code < 0)
        code = 0;
      if (code >= COLOR_FILTER_CODE_COUNT)
        code %= COLOR_FILTER_CODE_COUNT;

      const int perm_raw = (code >> 4) & 0x7;
      const int primary = (code >> 2) & 0x3;
      const int secondary = code & 0x3;
      const int perm = (perm_raw < COLOR_FILTER_PERMUTATIONS)
                           ? perm_raw
                           : (perm_raw % COLOR_FILTER_PERMUTATIONS);

      return {perm,
              primary % COLOR_FILTER_PRIMARY_PREDICTORS,
              secondary % COLOR_FILTER_SECONDARY_PREDICTORS};
    }

    inline int predict_primary(int mode, int c0)
    {
      switch (mode & 0x3)
      {
      case 0:
        return 0;
      case 1:
        return c0;
      case 2:
        return c0 / 2;
      case 3:
        return (3 * c0) / 2;
      default:
        return 0;
      }
    }

    inline int predict_secondary(int mode, int c0, int reference1)
    {
      switch (mode & 0x3)
      {
      case 0:
        return 0;
      case 1:
        return c0;
      case 2:
        return reference1;
      case 3:
        return (c0 + reference1) / 2;
      default:
        return 0;
      }
    }
  } // namespace

  void apply_color_filter(int code,
                          std::vector<int16_t> &b,
                          std::vector<int16_t> &g,
                          std::vector<int16_t> &r)
  {
    const std::size_t n = b.size();
    if (n == 0 || g.size() != n || r.size() != n)
      return;

    const ColorFilterParams params = decode_color_filter_code(code);
    const auto &perm = kColorPermutations[static_cast<std::size_t>(params.perm)];

    for (std::size_t i = 0; i < n; ++i)
    {
      const std::array<int, 3> source = {
          static_cast<int>(b[i]),
          static_cast<int>(g[i]),
          static_cast<int>(r[i])};

      const int d0 = source[perm[0]];
      const int d1 = source[perm[1]];
      const int d2 = source[perm[2]];

      const int predicted1 = predict_primary(params.primary, d0);
      const int predicted2 = predict_secondary(params.secondary, d0, d1);

      const int residual1 = d1 - predicted1;
      const int residual2 = d2 - predicted2;

      b[i] = static_cast<int16_t>(d0);
      g[i] = static_cast<int16_t>(residual1);
      r[i] = static_cast<int16_t>(residual2);
    }
  }

  void undo_color_filter(int code,
                         std::vector<int16_t> &b,
                         std::vector<int16_t> &g,
                         std::vector<int16_t> &r)
  {
    const std::size_t n = b.size();
    if (n == 0 || g.size() != n || r.size() != n)
      return;

    const ColorFilterParams params = decode_color_filter_code(code);
    const auto &perm = kColorPermutations[static_cast<std::size_t>(params.perm)];

    for (std::size_t i = 0; i < n; ++i)
    {
      const int c0 = static_cast<int>(b[i]);
      const int c1 = static_cast<int>(g[i]);
      const int c2 = static_cast<int>(r[i]);

      const int restored0 = c0;
      const int restored1 = c1 + predict_primary(params.primary, restored0);
      const int restored2 = c2 + predict_secondary(params.secondary, restored0, restored1);

      std::array<int, 3> destination = {0, 0, 0};
      destination[perm[0]] = restored0;
      destination[perm[1]] = restored1;
      destination[perm[2]] = restored2;

      b[i] = static_cast<int16_t>(destination[0]);
      g[i] = static_cast<int16_t>(destination[1]);
      r[i] = static_cast<int16_t>(destination[2]);
    }
  }

  namespace
  {
    struct DiffDirection
    {
      int dx = 0;
      int dy = 0;
    };

    DiffDirection get_diff_direction(DiffFilterType type)
    {
      switch (type)
      {
      case DiffFilterType::NWSE:
        return {-1, -1};
      case DiffFilterType::NESW:
        return {1, -1};
      case DiffFilterType::HORZ:
        return {1, 0};
      case DiffFilterType::VERT:
        return {0, -1};
      case DiffFilterType::None:
      case DiffFilterType::Count:
      default:
        return {0, 0};
      }
    }
  } // namespace

  uint16_t pack_block_sideinfo(const side_info &info)
  {
    const uint16_t filter_bits = static_cast<uint16_t>(info.filter_code & 0x7F);
    const uint16_t predictor_bit = (info.mode == PredictorMode::AVG) ? 1u : 0u;
    const uint16_t diff_bits = static_cast<uint16_t>(info.diff_index & 0x7);
    return static_cast<uint16_t>(filter_bits | (predictor_bit << 7) | (diff_bits << 8));
  }

  int unpack_filter_code(uint16_t sideinfo)
  {
    return static_cast<int>(sideinfo & 0x7F);
  }

  PredictorMode unpack_predictor_mode(uint16_t sideinfo)
  {
    return (sideinfo & 0x80) ? PredictorMode::AVG : PredictorMode::MED;
  }

  DiffFilterType unpack_diff_filter(uint16_t sideinfo)
  {
    const int diff = (sideinfo >> 8) & 0x7;
    if (diff < 0 || diff >= static_cast<int>(DiffFilterType::Count))
      return DiffFilterType::None;
    return static_cast<DiffFilterType>(diff);
  }

  std::vector<int16_t> apply_diff_filter(const BlockContext &ctx,
                                         DiffFilterType type,
                                         const std::vector<int16_t> &input)
  {
    if (type == DiffFilterType::None || input.empty())
      return input;

    const std::size_t width = ctx.bw;
    const std::size_t height = ctx.bh;
    const std::size_t expected = width * height;
    if (width == 0 || height == 0 || input.size() != expected)
      return input;

    const DiffDirection dir = get_diff_direction(type);
    if (dir.dx == 0 && dir.dy == 0)
      return input;

    std::vector<int16_t> output(input.size());
    for (std::size_t y = 0; y < height; ++y)
    {
      for (std::size_t x = 0; x < width; ++x)
      {
        const std::size_t idx = y * width + x;
        const int cur = static_cast<int>(input[idx]);
        const int nx = static_cast<int>(x) + dir.dx;
        const int ny = static_cast<int>(y) + dir.dy;
        if (nx >= 0 && nx < static_cast<int>(width) && ny >= 0 && ny < static_cast<int>(height))
        {
          const std::size_t nidx = static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
          const int neigh = static_cast<int>(input[nidx]);
          output[idx] = static_cast<int16_t>(cur - neigh);
        }
        else
        {
          output[idx] = static_cast<int16_t>(cur);
        }
      }
    }

    return output;
  }

  std::vector<int16_t> undo_diff_filter(const BlockContext &ctx,
                                        DiffFilterType type,
                                        const std::vector<int16_t> &input)
  {
    if (type == DiffFilterType::None || input.empty())
      return input;

    const std::size_t width = ctx.bw;
    const std::size_t height = ctx.bh;
    const std::size_t expected = width * height;
    if (width == 0 || height == 0 || input.size() != expected)
      return input;

    const DiffDirection dir = get_diff_direction(type);
    if (dir.dx == 0 && dir.dy == 0)
      return input;

    std::vector<int16_t> output(input.size());
    const bool reverse_y = dir.dy > 0;
    const bool reverse_x = dir.dx > 0;

    for (std::size_t yy = 0; yy < height; ++yy)
    {
      const std::size_t y = reverse_y ? (height - 1 - yy) : yy;
      for (std::size_t xx = 0; xx < width; ++xx)
      {
        const std::size_t x = reverse_x ? (width - 1 - xx) : xx;
        const std::size_t idx = y * width + x;
        int value = static_cast<int>(input[idx]);

        const int nx = static_cast<int>(x) + dir.dx;
        const int ny = static_cast<int>(y) + dir.dy;
        if (nx >= 0 && nx < static_cast<int>(width) && ny >= 0 && ny < static_cast<int>(height))
        {
          const std::size_t nidx = static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
          value += static_cast<int>(output[nidx]);
        }

        output[idx] = static_cast<int16_t>(value);
      }
    }

    return output;
  }

  std::vector<detail::image<uint8_t>> extract_planes(const PixelBuffer &src, int colors)
  {
    const std::size_t width = src.width;
    const std::size_t height = src.height;
    if (colors == 1)
    {
      detail::image<uint8_t> gray(width, height);
      for (std::size_t y = 0; y < height; ++y)
      {
        const std::size_t row_idx = y * width;
        for (std::size_t x = 0; x < width; ++x)
        {
          const std::size_t idx = row_idx + x;
          uint8_t r = 0;
          uint8_t g = 0;
          uint8_t b = 0;
          if (src.channels == 4)
          {
            r = src.data[idx * 4 + 1];
            g = src.data[idx * 4 + 2];
            b = src.data[idx * 4 + 3];
          }
          else
          {
            r = src.data[idx * 3 + 0];
            g = src.data[idx * 3 + 1];
            b = src.data[idx * 3 + 2];
          }
          const uint16_t gray_val = static_cast<uint16_t>(r) * 299u + static_cast<uint16_t>(g) * 587u + static_cast<uint16_t>(b) * 114u;
          gray.row_ptr(y)[x] = static_cast<uint8_t>(gray_val / 1000u);
        }
      }
      return detail::split_components_from_gray(gray);
    }

    detail::image<uint32_t> packed(width, height);
    for (std::size_t y = 0; y < height; ++y)
    {
      const std::size_t row_idx = y * width;
      uint32_t *dst = packed.row_ptr(y);
      for (std::size_t x = 0; x < width; ++x)
      {
        const std::size_t idx = row_idx + x;
        uint8_t a = 255;
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        if (src.channels == 4)
        {
          a = src.data[idx * 4 + 0];
          r = src.data[idx * 4 + 1];
          g = src.data[idx * 4 + 2];
          b = src.data[idx * 4 + 3];
        }
        else
        {
          r = src.data[idx * 3 + 0];
          g = src.data[idx * 3 + 1];
          b = src.data[idx * 3 + 2];
        }
        dst[x] = static_cast<uint32_t>(b) | (static_cast<uint32_t>(g) << 8) |
                 (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(a) << 24);
      }
    }

    const std::size_t component_count = (colors == 4) ? 4u : 3u;
    auto planes = detail::split_components_from_packed(packed, component_count);

    if (colors == 4 && planes.size() < 4)
    {
      planes.resize(4, detail::image<uint8_t>(width, height, 255));
    }
    if (colors == 3 && planes.size() < 3)
    {
      planes.resize(3, detail::image<uint8_t>(width, height, 0));
    }
    return planes;
  }

  PixelBuffer assemble_pixelbuffer(const std::vector<detail::image<uint8_t>> &planes,
                                   uint32_t width,
                                   uint32_t height,
                                   int colors)
  {
    PixelBuffer out;
    out.width = width;
    out.height = height;
    if (colors == 4)
      out.channels = 4;
    else
      out.channels = 3;
    out.data.resize(static_cast<std::size_t>(out.width) * out.height * out.channels);

    for (std::size_t y = 0; y < height; ++y)
    {
      const std::size_t row_idx = y * width;
      for (std::size_t x = 0; x < width; ++x)
      {
        const std::size_t idx = row_idx + x;
        if (colors == 4)
        {
          out.data[idx * 4 + 0] = planes[3].row_ptr(y)[x];
          out.data[idx * 4 + 1] = planes[2].row_ptr(y)[x];
          out.data[idx * 4 + 2] = planes[1].row_ptr(y)[x];
          out.data[idx * 4 + 3] = planes[0].row_ptr(y)[x];
        }
        else
        {
          out.data[idx * 3 + 0] = planes[2].row_ptr(y)[x];
          out.data[idx * 3 + 1] = planes[1].row_ptr(y)[x];
          out.data[idx * 3 + 2] = planes[0].row_ptr(y)[x];
        }
      }
    }
    return out;
  }

  namespace detail
  {

    bool write_header(FILE *fp, const Header &hdr)
    {
      const unsigned char mark[11] = {'T', 'L', 'G', '7', '.', '0', 0, 'r', 'a', 'w', 0x1a};
      if (fwrite(mark, 1, sizeof(mark), fp) != sizeof(mark))
        return false;
      if (std::fputc(hdr.colors, fp) == EOF)
        return false;
      if (std::fputc(hdr.flags, fp) == EOF)
        return false;
      if (std::fputc(hdr.reserved[0], fp) == EOF)
        return false;
      if (std::fputc(hdr.reserved[1], fp) == EOF)
        return false;
      tlg::detail::write_u32le(fp, hdr.width);
      tlg::detail::write_u32le(fp, hdr.height);
      tlg::detail::write_u32le(fp, hdr.block_count);
      tlg::detail::write_u32le(fp, hdr.chunk_count);
      return !std::ferror(fp);
    }

    bool read_header(FILE *fp, Header &hdr, std::string &err)
    {
      unsigned char colors = 0;
      unsigned char flags = 0;
      unsigned char r1 = 0, r2 = 0;
      int c0 = std::fgetc(fp);
      int c1 = std::fgetc(fp);
      int c2 = std::fgetc(fp);
      int c3 = std::fgetc(fp);
      if (c0 == EOF || c1 == EOF || c2 == EOF || c3 == EOF)
      {
        err = "tlg7: read header";
        return false;
      }
      colors = static_cast<unsigned char>(c0);
      flags = static_cast<unsigned char>(c1);
      r1 = static_cast<unsigned char>(c2);
      r2 = static_cast<unsigned char>(c3);

      hdr.colors = colors;
      hdr.flags = flags;
      hdr.reserved[0] = r1;
      hdr.reserved[1] = r2;

      hdr.width = tlg::detail::read_u32le(fp);
      hdr.height = tlg::detail::read_u32le(fp);
      hdr.block_count = tlg::detail::read_u32le(fp);
      hdr.chunk_count = tlg::detail::read_u32le(fp);

      if (!(hdr.colors == 1 || hdr.colors == 3 || hdr.colors == 4))
      {
        err = "tlg7: unsupported color count";
        return false;
      }
      if (hdr.reserved[0] || hdr.reserved[1])
      {
        err = "tlg7: reserved flags not zero";
        return false;
      }
      if (hdr.flags > 1)
      {
        err = "tlg7: unsupported pipeline order";
        return false;
      }
      if (hdr.width == 0 || hdr.height == 0)
      {
        err = "tlg7: invalid dimensions";
        return false;
      }
      if (hdr.block_count == 0 || hdr.chunk_count == 0)
      {
        err = "tlg7: inconsistent block metadata";
        return false;
      }
      return true;
    }

  } // namespace detail

} // namespace tlg::v7
