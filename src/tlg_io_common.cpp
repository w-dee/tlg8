#include "tlg_io_common.h"

#include <cstring>

namespace tlg::detail {

bool read_exact(FILE *fp, void *buf, size_t n) {
  return std::fread(buf, 1, n, fp) == n;
}

uint32_t read_u32le(FILE *fp) {
  uint8_t b[4];
  if (!read_exact(fp, b, 4)) return 0;  // caller checks error state
  return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
         (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

void write_u32le(FILE *fp, uint32_t v) {
  uint8_t b[4] = {
      static_cast<uint8_t>(v & 0xff),
      static_cast<uint8_t>((v >> 8) & 0xff),
      static_cast<uint8_t>((v >> 16) & 0xff),
      static_cast<uint8_t>((v >> 24) & 0xff)};
  std::fwrite(b, 1, 4, fp);
}

int tlg5_lzss_decompress(uint8_t *out, const uint8_t *in, int insize, uint8_t *text, int r) {
  const uint8_t *inlim = in + insize;
  if (in >= inlim) return r;
  uint32_t flags;
getmore:
  flags = static_cast<uint32_t>(*in++) | 0x100u;
loop:
  {
    bool backref = (flags & 1u) != 0;
    flags >>= 1;
    if (!flags) {
      if (in >= inlim) return r;
      goto getmore;
    }
    if (backref) {
      uint16_t word = static_cast<uint16_t>(in[0] | (static_cast<uint16_t>(in[1]) << 8));
      in += 2;
      int mpos = word & 0x0fff;
      int mlen = word >> 12;
      if (mlen == 15) mlen += *in++;
      mlen += 3;
      do {
        uint8_t c = text[mpos];
        mpos = (mpos + 1) & (4096 - 1);
        text[r] = c;
        *out++ = c;
        r = (r + 1) & (4096 - 1);
      } while (--mlen);
    } else {
      uint8_t c = *in++;
      text[r] = c;
      *out++ = c;
      r = (r + 1) & (4096 - 1);
    }
  }
  if (in < inlim) goto loop;
  return r;
}

} // namespace tlg::detail

