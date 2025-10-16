#include "image_io.h"
#include "tlg_io_common.h"
#include "tlg5_io.h"
#include "tlg6_io.h"
#include "tlg7_io.h"
#include "tlg8_io.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace
{
  using tlg::detail::read_exact;
  using tlg::detail::read_u32le;

  bool decode_raw(FILE *fp, const char *mark11, PixelBuffer &out, std::string &err)
  {
    if (std::memcmp(mark11, "TLG5.0\x00raw\x1a\x00", 11) == 0)
    {
      return tlg::v5::decode_stream(fp, out, err);
    }
    if (std::memcmp(mark11, "TLG6.0\x00raw\x1a\x00", 11) == 0)
    {
      return tlg::v6::decode_stream(fp, out, err);
    }
    if (std::memcmp(mark11, "TLG7.0\x00raw\x1a\x00", 11) == 0)
    {
      return tlg::v7::decode_stream(fp, out, err);
    }
    if (std::memcmp(mark11, "TLG8.0\x00raw\x1a\x00", 11) == 0)
    {
      return tlg::v8::decode_stream(fp, out, err);
    }
    err = "invalid tlg raw header";
    return false;
  }

} // namespace

bool load_tlg(const std::string &path, PixelBuffer &out, std::string &err)
{
  err.clear();
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp)
  {
    err = "cannot open file";
    return false;
  }

  char mark[11];
  if (!read_exact(fp, mark, sizeof(mark)))
  {
    std::fclose(fp);
    err = "read error";
    return false;
  }

  bool ok = false;
  if (std::memcmp(mark, "TLG0.0\x00sds\x1a\x00", 11) == 0)
  {
    uint32_t rawlen = read_u32le(fp);
    (void)rawlen;
    char rawmark[11];
    if (!read_exact(fp, rawmark, sizeof(rawmark)))
    {
      std::fclose(fp);
      err = "read error";
      return false;
    }
    ok = decode_raw(fp, rawmark, out, err);
  }
  else
  {
    ok = decode_raw(fp, mark, out, err);
  }

  std::fclose(fp);
  return ok;
}

bool save_tlg(const std::string &path,
              const PixelBuffer &src,
              const TlgOptions &opt,
              std::string &err,
              uint64_t *out_entropy_bits)
{
  err.clear();
  if (out_entropy_bits)
    *out_entropy_bits = 0;
  if (!(src.channels == 3 || src.channels == 4))
  {
    err = "unsupported pixel channels";
    return false;
  }

  int desired_colors;
  if (opt.fmt == ImageFormat::A8R8G8B8)
    desired_colors = 4;
  else if (opt.fmt == ImageFormat::R8G8B8)
    desired_colors = 3;
  else
    desired_colors = src.has_alpha() ? 4 : 3;

  FILE *fp = std::fopen(path.c_str(), "wb");
  if (!fp)
  {
    err = "cannot open file";
    return false;
  }

  bool ok = false;
  if (opt.version == 7)
  {
    std::string cfg_err;
    if (!tlg::v7::configure_golomb_table(opt.tlg7_golomb_table_path, cfg_err))
    {
      err = cfg_err;
      std::fclose(fp);
      return false;
    }
    ok = tlg::v7::enc::write_raw(fp,
                                 src,
                                 desired_colors,
                                 opt.tlg7_pipeline_order,
                                 opt.tlg7_dump_residuals_path,
                                 opt.tlg7_dump_residuals_order,
                                 err);
  }
  else if (opt.version == 8)
  {
    std::string cfg_err;
    if (!tlg::v8::configure_golomb_table(opt.tlg8_golomb_table_path, cfg_err))
    {
      err = cfg_err;
      std::fclose(fp);
      return false;
    }
    ok = tlg::v8::enc::write_raw(fp,
                                 src,
                                 desired_colors,
                                 opt.tlg8_dump_residuals_path,
                                 opt.tlg8_dump_residuals_order,
                                 opt.tlg8_dump_golomb_prediction_path,
                                 opt.tlg8_dump_reorder_histogram_path,
                                 opt.tlg8_write_residuals_bmp_path,
                                 opt.tlg8_write_residuals_order,
                                 opt.tlg8_write_residuals_emphasis,
                                 opt.tlg8_force_hilbert_reorder,
                                 err,
                                 out_entropy_bits);
  }
  else if (opt.version == 6)
  {
    ok = tlg::v6::enc::write_raw(fp, src, desired_colors, err);
  }
  else if (opt.version == 5)
  {
    ok = tlg::v5::write_raw(fp, src, desired_colors, err);
  }
  else
  {
    err = "unsupported TLG version";
    std::fclose(fp);
    return false;
  }

  std::fclose(fp);
  return ok;
}
