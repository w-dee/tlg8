#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "image_io.h"

namespace tlg::v7
{
namespace detail
{

template <class PixelT>
class raster_subimage;

// -----------------------------------------------------------------------------
// image<T>: light-weight 2D raster container with shared storage
// -----------------------------------------------------------------------------
template <class PixelT>
class image
{
public:
  using pixel_type = PixelT;
  using storage_type = std::vector<pixel_type>;
  using storage_ptr = std::shared_ptr<storage_type>;

  image() = default;

  image(std::size_t width, std::size_t height)
      : width_(width), height_(height), stride_(width),
        storage_(std::make_shared<storage_type>(width * height)), offset_(0)
  {
  }

  image(std::size_t width, std::size_t height, const pixel_type &value)
      : width_(width), height_(height), stride_(width),
        storage_(std::make_shared<storage_type>(width * height, value)), offset_(0)
  {
  }

  bool empty() const noexcept { return width_ == 0 || height_ == 0; }

  std::size_t get_width() const noexcept { return width_; }
  std::size_t get_height() const noexcept { return height_; }
  std::size_t get_stride() const noexcept { return stride_; }

  pixel_type *data() noexcept
  {
    return storage_ ? storage_->data() + offset_ : nullptr;
  }

  const pixel_type *data() const noexcept
  {
    return storage_ ? storage_->data() + offset_ : nullptr;
  }

  pixel_type *row_ptr(std::size_t y) noexcept
  {
    return data() ? data() + y * stride_ : nullptr;
  }

  const pixel_type *row_ptr(std::size_t y) const noexcept
  {
    return data() ? data() + y * stride_ : nullptr;
  }

  pixel_type &at(std::size_t x, std::size_t y)
  {
    return storage_->at(offset_ + y * stride_ + x);
  }

  const pixel_type &at(std::size_t x, std::size_t y) const
  {
    return storage_->at(offset_ + y * stride_ + x);
  }

  void fill(const pixel_type &value)
  {
    if (!storage_)
      return;
    pixel_type *base = storage_->data() + offset_;
    for (std::size_t y = 0; y < height_; ++y)
    {
      std::fill(base + y * stride_, base + y * stride_ + width_, value);
    }
  }

  storage_ptr storage() const noexcept { return storage_; }
  std::size_t storage_offset() const noexcept { return offset_; }

private:
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::size_t stride_ = 0;
  storage_ptr storage_;
  std::size_t offset_ = 0;

  template <class>
  friend class raster_subimage;
};

// Alias for clarity when referring to planar rasters
template <class PixelT>
using raster_image = image<PixelT>;

// -----------------------------------------------------------------------------
// raster_subimage<T>: non-owning clipped view into an image<T>
// -----------------------------------------------------------------------------
template <class PixelT>
class raster_subimage
{
public:
  using image_type = raster_image<PixelT>;
  using pixel_type = typename image_type::pixel_type;

  raster_subimage() = default;

  raster_subimage(image_type &img,
                  std::size_t sx,
                  std::size_t sy,
                  std::size_t w,
                  std::size_t h)
  {
    if (img.empty())
      return;

    const std::size_t iw = img.get_width();
    const std::size_t ih = img.get_height();

    if (sx >= iw || sy >= ih)
      return; // fully clipped outside → empty view

    width_ = std::min(w, iw - sx);
    height_ = std::min(h, ih - sy);
    stride_ = img.get_stride();
    storage_ = img.storage();
    offset_ = img.storage_offset() + sy * stride_ + sx;
  }

  bool empty() const noexcept { return width_ == 0 || height_ == 0; }

  std::size_t get_width() const noexcept { return width_; }
  std::size_t get_height() const noexcept { return height_; }

  pixel_type *row_ptr(std::size_t y)
  {
    return (storage_ && y < height_) ? storage_->data() + offset_ + y * stride_ : nullptr;
  }

  const pixel_type *row_ptr(std::size_t y) const
  {
    return (storage_ && y < height_) ? storage_->data() + offset_ + y * stride_ : nullptr;
  }

  image_type get_copied_raster_image() const
  {
    image_type out(width_, height_);
    if (!storage_ || out.empty())
      return out;
    const pixel_type *src = storage_->data() + offset_;
    for (std::size_t y = 0; y < height_; ++y)
    {
      std::copy(src + y * stride_, src + y * stride_ + width_, out.row_ptr(y));
    }
    return out;
  }

private:
  typename image_type::storage_ptr storage_;
  std::size_t stride_ = 0;
  std::size_t offset_ = 0;
  std::size_t width_ = 0;
  std::size_t height_ = 0;
};

std::vector<image<uint8_t>> split_components_from_gray(const image<uint8_t> &gray);
std::vector<image<uint8_t>> split_components_from_packed(const image<uint32_t> &packed,
                                                         std::size_t component_count);

} // namespace detail

bool configure_golomb_table(const std::string &path, std::string &err);

bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err);

namespace enc
{

bool write_raw(FILE *fp, const PixelBuffer &src, int colors, bool fast_mode, std::string &err);

} // namespace enc

} // namespace tlg::v7
