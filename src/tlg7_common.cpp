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

  int sample_pixel(const detail::image<uint8_t> &img, int x, int y)
  {
    if (x < 0 || y < 0)
      return 0;
    const std::size_t ux = static_cast<std::size_t>(x);
    const std::size_t uy = static_cast<std::size_t>(y);
    if (ux >= img.get_width() || uy >= img.get_height())
      return 0;
    return img.row_ptr(uy)[ux];
  }

  void apply_color_filter(int code,
                          std::vector<int16_t> &b,
                          std::vector<int16_t> &g,
                          std::vector<int16_t> &r)
  {
    const std::size_t n = b.size();
    switch (code)
    {
    case 0:
      return;
    case 1:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - g[i]);
        b[i] = static_cast<int16_t>(b[i] - g[i]);
      }
      break;
    case 2:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - g[i]);
        g[i] = static_cast<int16_t>(g[i] - b[i]);
      }
      break;
    case 3:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] - g[i]);
        g[i] = static_cast<int16_t>(g[i] - r[i]);
      }
      break;
    case 4:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - g[i]);
        g[i] = static_cast<int16_t>(g[i] - b[i]);
        b[i] = static_cast<int16_t>(b[i] - r[i]);
      }
      break;
    case 5:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] - b[i]);
        b[i] = static_cast<int16_t>(b[i] - r[i]);
      }
      break;
    case 6:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] - g[i]);
      break;
    case 7:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] - b[i]);
      break;
    case 8:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] - g[i]);
      break;
    case 9:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] - g[i]);
        g[i] = static_cast<int16_t>(g[i] - r[i]);
        r[i] = static_cast<int16_t>(r[i] - b[i]);
      }
      break;
    case 10:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] - r[i]);
        b[i] = static_cast<int16_t>(b[i] - r[i]);
      }
      break;
    case 11:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - b[i]);
        g[i] = static_cast<int16_t>(g[i] - b[i]);
      }
      break;
    case 12:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] - r[i]);
        r[i] = static_cast<int16_t>(r[i] - b[i]);
      }
      break;
    case 13:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] - r[i]);
        r[i] = static_cast<int16_t>(r[i] - b[i]);
        b[i] = static_cast<int16_t>(b[i] - g[i]);
      }
      break;
    case 14:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - b[i]);
        b[i] = static_cast<int16_t>(b[i] - g[i]);
        g[i] = static_cast<int16_t>(g[i] - r[i]);
      }
      break;
    case 15:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(b[i] << 1);
        r[i] = static_cast<int16_t>(r[i] - t);
        g[i] = static_cast<int16_t>(g[i] - t);
      }
      break;
    case 16:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] - r[i]);
      break;
    case 17:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - b[i]);
        b[i] = static_cast<int16_t>(b[i] - g[i]);
      }
      break;
    case 18:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] - b[i]);
      break;
    case 19:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] - r[i]);
        r[i] = static_cast<int16_t>(r[i] - g[i]);
      }
      break;
    case 20:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] - r[i]);
      break;
    case 21:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] - (g[i] >> 1));
      break;
    case 22:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] - (b[i] >> 1));
      break;
    case 23:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] - g[i]);
        b[i] = static_cast<int16_t>(b[i] - r[i]);
        g[i] = static_cast<int16_t>(g[i] - b[i]);
      }
      break;
    case 24:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] - b[i]);
        r[i] = static_cast<int16_t>(r[i] - g[i]);
        b[i] = static_cast<int16_t>(b[i] - r[i]);
      }
      break;
    case 25:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] - (r[i] >> 1));
      break;
    case 26:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] - (g[i] >> 1));
      break;
    case 27:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(r[i] >> 1);
        b[i] = static_cast<int16_t>(b[i] - t);
        g[i] = static_cast<int16_t>(g[i] - t);
      }
      break;
    case 28:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] - (b[i] >> 1));
      break;
    case 29:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(g[i] >> 1);
        b[i] = static_cast<int16_t>(b[i] - t);
        r[i] = static_cast<int16_t>(r[i] - t);
      }
      break;
    case 30:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(b[i] >> 1);
        g[i] = static_cast<int16_t>(g[i] - t);
        r[i] = static_cast<int16_t>(r[i] - t);
      }
      break;
    case 31:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] - (r[i] >> 1));
      break;
    default:
      break;
    }
  }

  void undo_color_filter(int code,
                         std::vector<int16_t> &b,
                         std::vector<int16_t> &g,
                         std::vector<int16_t> &r)
  {
    const std::size_t n = b.size();
    switch (code)
    {
    case 0:
      return;
    case 1:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + g[i]);
        r[i] = static_cast<int16_t>(r[i] + g[i]);
      }
      break;
    case 2:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] + b[i]);
        r[i] = static_cast<int16_t>(r[i] + g[i]);
      }
      break;
    case 3:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] + r[i]);
        b[i] = static_cast<int16_t>(b[i] + g[i]);
      }
      break;
    case 4:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + r[i]);
        g[i] = static_cast<int16_t>(g[i] + b[i]);
        r[i] = static_cast<int16_t>(r[i] + g[i]);
      }
      break;
    case 5:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + r[i]);
        g[i] = static_cast<int16_t>(g[i] + b[i]);
      }
      break;
    case 6:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] + g[i]);
      break;
    case 7:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] + b[i]);
      break;
    case 8:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] + g[i]);
      break;
    case 9:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] + b[i]);
        g[i] = static_cast<int16_t>(g[i] + r[i]);
        b[i] = static_cast<int16_t>(b[i] + g[i]);
      }
      break;
    case 10:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + r[i]);
        g[i] = static_cast<int16_t>(g[i] + r[i]);
      }
      break;
    case 11:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] + b[i]);
        r[i] = static_cast<int16_t>(r[i] + b[i]);
      }
      break;
    case 12:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] + b[i]);
        g[i] = static_cast<int16_t>(g[i] + r[i]);
      }
      break;
    case 13:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + g[i]);
        r[i] = static_cast<int16_t>(r[i] + b[i]);
        g[i] = static_cast<int16_t>(g[i] + r[i]);
      }
      break;
    case 14:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] + r[i]);
        b[i] = static_cast<int16_t>(b[i] + g[i]);
        r[i] = static_cast<int16_t>(r[i] + b[i]);
      }
      break;
    case 15:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(b[i] << 1);
        g[i] = static_cast<int16_t>(g[i] + t);
        r[i] = static_cast<int16_t>(r[i] + t);
      }
      break;
    case 16:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] + r[i]);
      break;
    case 17:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + g[i]);
        r[i] = static_cast<int16_t>(r[i] + b[i]);
      }
      break;
    case 18:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] + b[i]);
      break;
    case 19:
      for (std::size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<int16_t>(r[i] + g[i]);
        b[i] = static_cast<int16_t>(b[i] + r[i]);
      }
      break;
    case 20:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] + r[i]);
      break;
    case 21:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] + (g[i] >> 1));
      break;
    case 22:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] + (b[i] >> 1));
      break;
    case 23:
      for (std::size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<int16_t>(g[i] + b[i]);
        b[i] = static_cast<int16_t>(b[i] + r[i]);
        r[i] = static_cast<int16_t>(r[i] + g[i]);
      }
      break;
    case 24:
      for (std::size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<int16_t>(b[i] + r[i]);
        r[i] = static_cast<int16_t>(r[i] + g[i]);
        g[i] = static_cast<int16_t>(g[i] + b[i]);
      }
      break;
    case 25:
      for (std::size_t i = 0; i < n; ++i)
        g[i] = static_cast<int16_t>(g[i] + (r[i] >> 1));
      break;
    case 26:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] + (g[i] >> 1));
      break;
    case 27:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(r[i] >> 1);
        b[i] = static_cast<int16_t>(b[i] + t);
        g[i] = static_cast<int16_t>(g[i] + t);
      }
      break;
    case 28:
      for (std::size_t i = 0; i < n; ++i)
        r[i] = static_cast<int16_t>(r[i] + (b[i] >> 1));
      break;
    case 29:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(g[i] >> 1);
        b[i] = static_cast<int16_t>(b[i] + t);
        r[i] = static_cast<int16_t>(r[i] + t);
      }
      break;
    case 30:
      for (std::size_t i = 0; i < n; ++i)
      {
        const int16_t t = static_cast<int16_t>(b[i] >> 1);
        g[i] = static_cast<int16_t>(g[i] + t);
        r[i] = static_cast<int16_t>(r[i] + t);
      }
      break;
    case 31:
      for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<int16_t>(b[i] + (r[i] >> 1));
      break;
    default:
      break;
    }
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
      if (std::fputc(hdr.reserved[0], fp) == EOF)
        return false;
      if (std::fputc(hdr.reserved[1], fp) == EOF)
        return false;
      if (std::fputc(hdr.reserved[2], fp) == EOF)
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
      unsigned char r1 = 0, r2 = 0, r3 = 0;
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
      r1 = static_cast<unsigned char>(c1);
      r2 = static_cast<unsigned char>(c2);
      r3 = static_cast<unsigned char>(c3);

      hdr.colors = colors;
      hdr.reserved[0] = r1;
      hdr.reserved[1] = r2;
      hdr.reserved[2] = r3;

      hdr.width = tlg::detail::read_u32le(fp);
      hdr.height = tlg::detail::read_u32le(fp);
      hdr.block_count = tlg::detail::read_u32le(fp);
      hdr.chunk_count = tlg::detail::read_u32le(fp);

      if (!(hdr.colors == 1 || hdr.colors == 3 || hdr.colors == 4))
      {
        err = "tlg7: unsupported color count";
        return false;
      }
      if (hdr.reserved[0] || hdr.reserved[1] || hdr.reserved[2])
      {
        err = "tlg7: reserved flags not zero";
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
