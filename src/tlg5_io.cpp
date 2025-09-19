#include "tlg5_io.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "tlg_io_common.h"

namespace {
using tlg::detail::read_exact;
using tlg::detail::read_u32le;
using tlg::detail::tlg5_lzss_decompress;
using tlg::detail::write_u32le;
}

namespace tlg::v5 {

bool decode_stream(FILE* fp, PixelBuffer &out, std::string &err) {
  // We assume the 11-byte raw header "TLG5.0\0raw\x1a\0" has already been consumed.
  uint8_t colors_byte;
  if (!read_exact(fp, &colors_byte, 1)) { err = "tlg5: read colors"; return false; }
  int colors = colors_byte;
  uint32_t width = read_u32le(fp);
  uint32_t height = read_u32le(fp);
  uint32_t blockheight = read_u32le(fp);
  if (!(colors == 3 || colors == 4)) { err = "tlg5: unsupported color count"; return false; }
  if (width == 0 || height == 0 || blockheight == 0) { err = "tlg5: invalid dimensions"; return false; }
  int blockcount = (int)((height - 1) / blockheight) + 1;

  // skip block sizes (we will still read blocks sequentially as they appear)
  std::vector<uint32_t> blocksizes(blockcount);
  for (int i = 0; i < blockcount; ++i) blocksizes[i] = read_u32le(fp);

  // prepare output ARGB32
  out.width = width; out.height = height; out.channels = 4;
  out.data.assign(static_cast<size_t>(width) * height * 4, 0);

  std::vector<uint8_t> inbuf(blockheight * width + 16);
  std::vector<std::vector<uint8_t>> outbuf(colors, std::vector<uint8_t>(blockheight * width + 16));
  std::vector<uint8_t> upper(width * 4, 0);
  std::vector<uint8_t> curline(width * 4, 0);
  std::vector<uint8_t> prevcl(colors, 0);
  std::vector<int16_t> val(colors, 0);
  std::vector<uint8_t*> cmpin(colors);
  for (int c = 0; c < colors; ++c) cmpin[c] = outbuf[c].data();

  std::vector<uint8_t> text(4096, 0);
  int r = 0;

  for (uint32_t yblk = 0; yblk < height; yblk += blockheight) {
    uint32_t h = (yblk + blockheight <= height) ? blockheight : (height - yblk);

    // read each color plane block, decompress if needed
    for (int c = 0; c < colors; ++c) {
      uint8_t method;
      if (!read_exact(fp, &method, 1)) { err = "tlg5: read method"; return false; }
      uint32_t size = read_u32le(fp);
      if (size > inbuf.size()) inbuf.resize(size);
      if (!read_exact(fp, inbuf.data(), size)) { err = "tlg5: read block"; return false; }
      if (method == 0) {
        // compressed via LZSS
        int new_r = tlg5_lzss_decompress(cmpin[c], inbuf.data(), (int)size, text.data(), r);
        if (new_r < 0) {
          err = "tlg5: lzss error (block=" + std::to_string(yblk) + ", plane=" + std::to_string(c) + ")";
          return false;
        }
        r = new_r;
      } else if (method == 1) {
        // stored
        if (size > h * width) { err = "tlg5: stored size too big"; return false; }
        memcpy(cmpin[c], inbuf.data(), size);
      } else {
        err = "tlg5: unknown block method"; return false;
      }
    }

    // Compose into ARGB using the same algorithm as the core
    std::vector<uint8_t*> outbufp(colors);
    for (int c = 0; c < colors; ++c) outbufp[c] = cmpin[c];

    for (uint32_t y = yblk; y < yblk + h; ++y) {
      uint8_t* outp = &out.data[(static_cast<size_t>(y) * width) * 4];
      if (y > 0) {
        // not first line: add upper line
        const uint8_t* up = &out.data[(static_cast<size_t>(y - 1) * width) * 4];
        if (colors == 3) {
          int pb = 0, pg = 0, pr = 0;
          for (uint32_t x = 0; x < width; ++x) {
            int b = outbufp[0][x];
            int g = outbufp[1][x];
            int r = outbufp[2][x];
            b += g; r += g;
            pb += b; pg += g; pr += r;
            uint8_t B = (uint8_t)((pb + up[x*4 + 3]) & 0xff);
            uint8_t G = (uint8_t)((pg + up[x*4 + 2]) & 0xff);
            uint8_t R = (uint8_t)((pr + up[x*4 + 1]) & 0xff);
            outp[0] = 255; outp[1] = R; outp[2] = G; outp[3] = B; outp += 4;
          }
          outbufp[0] += width; outbufp[1] += width; outbufp[2] += width;
        } else { // colors == 4
          int pb = 0, pg = 0, pr = 0, pa = 0;
          for (uint32_t x = 0; x < width; ++x) {
            int b = outbufp[0][x];
            int g = outbufp[1][x];
            int r = outbufp[2][x];
            int a = outbufp[3][x];
            b += g; r += g;
            pb += b; pg += g; pr += r; pa += a;
            uint8_t B = (uint8_t)((pb + up[x*4 + 3]) & 0xff);
            uint8_t G = (uint8_t)((pg + up[x*4 + 2]) & 0xff);
            uint8_t R = (uint8_t)((pr + up[x*4 + 1]) & 0xff);
            uint8_t A = (uint8_t)((pa + up[x*4 + 0]) & 0xff);
            outp[0] = A; outp[1] = R; outp[2] = G; outp[3] = B; outp += 4;
          }
          outbufp[0] += width; outbufp[1] += width; outbufp[2] += width; outbufp[3] += width;
        }
      } else {
        // first line: no upper
        if (colors == 3) {
          int pb = 0, pg = 0, pr = 0;
          for (uint32_t x = 0; x < width; ++x) {
            int b = outbufp[0][x];
            int g = outbufp[1][x];
            int r = outbufp[2][x];
            b += g; r += g;
            pb += b; pg += g; pr += r;
            outp[0] = 255; outp[1] = (uint8_t)(pr & 0xff); outp[2] = (uint8_t)(pg & 0xff); outp[3] = (uint8_t)(pb & 0xff); outp += 4;
          }
          outbufp[0] += width; outbufp[1] += width; outbufp[2] += width;
        } else { // colors == 4
          int pb = 0, pg = 0, pr = 0, pa = 0;
          for (uint32_t x = 0; x < width; ++x) {
            int b = outbufp[0][x];
            int g = outbufp[1][x];
            int r = outbufp[2][x];
            int a = outbufp[3][x];
            b += g; r += g;
            pb += b; pg += g; pr += r; pa += a;
            outp[0] = (uint8_t)(pa & 0xff); outp[1] = (uint8_t)(pr & 0xff); outp[2] = (uint8_t)(pg & 0xff); outp[3] = (uint8_t)(pb & 0xff); outp += 4;
          }
          outbufp[0] += width; outbufp[1] += width; outbufp[2] += width; outbufp[3] += width;
        }
      }
    }
  }
  return true;
}
bool write_raw(FILE* fp, const PixelBuffer &src, int desired_colors, std::string &err) {
  (void)err;
  // header
  unsigned char mark[11] = { 'T','L','G','5','.','0',0,'r','a','w',0x1a };
  fwrite(mark, 1, sizeof(mark), fp);
  uint8_t colors_u8 = (uint8_t)desired_colors;
  fwrite(&colors_u8, 1, 1, fp);
  write_u32le(fp, src.width);
  write_u32le(fp, src.height);
  const uint32_t blockheight = 4;
  write_u32le(fp, blockheight);

  const uint32_t blockcount = (src.height + blockheight - 1) / blockheight;
  long blocksizepos = ftell(fp);
  for (uint32_t i = 0; i < blockcount; ++i) write_u32le(fp, 0); // placeholder

  std::vector<uint8_t> cmpbuf0(blockheight * src.width);
  std::vector<uint8_t> cmpbuf1(blockheight * src.width);
  std::vector<uint8_t> cmpbuf2(blockheight * src.width);
  std::vector<uint8_t> cmpbuf3(blockheight * src.width);
  std::vector<uint32_t> blocksizes(blockcount, 0);

  // previous line (for upper). Stored as ARGB components per pixel
  std::vector<uint8_t> upper_line(src.width * 4, 0);

  uint32_t block_index = 0;
  for (uint32_t by = 0; by < src.height; by += blockheight, ++block_index) {
    const uint32_t h = std::min(blockheight, src.height - by);
    size_t inp = 0;
    for (uint32_t y = 0; y < h; ++y) {
      const uint8_t* s = &src.data[(static_cast<size_t>(by + y) * src.width) * src.channels];
      // prevcl reset per line
      int prevclB = 0, prevclG = 0, prevclR = 0, prevclA = 0;
      for (uint32_t x = 0; x < src.width; ++x) {
        uint8_t A,R,G,B;
        if (src.channels == 4) { A = s[x*4 + 0]; R = s[x*4 + 1]; G = s[x*4 + 2]; B = s[x*4 + 3]; }
        else { R = s[x*3 + 0]; G = s[x*3 + 1]; B = s[x*3 + 2]; A = 255; }
        uint8_t upA = upper_line[x*4 + 0];
        uint8_t upR = upper_line[x*4 + 1];
        uint8_t upG = upper_line[x*4 + 2];
        uint8_t upB = upper_line[x*4 + 3];

        int clB = (int)B - (int)upB; int valB = (clB - prevclB) & 0xff; prevclB = clB & 0xff;
        int clG = (int)G - (int)upG; int valG = (clG - prevclG) & 0xff; prevclG = clG & 0xff;
        int clR = (int)R - (int)upR; int valR = (clR - prevclR) & 0xff; prevclR = clR & 0xff;
        if (desired_colors == 3) {
          cmpbuf0[inp] = static_cast<uint8_t>((valB - valG) & 0xff);
          cmpbuf1[inp] = static_cast<uint8_t>(valG);
          cmpbuf2[inp] = static_cast<uint8_t>((valR - valG) & 0xff);
        } else {
          int clA = (int)A - (int)upA; int valA = (clA - prevclA) & 0xff; prevclA = clA & 0xff;
          cmpbuf0[inp] = static_cast<uint8_t>((valB - valG) & 0xff);
          cmpbuf1[inp] = static_cast<uint8_t>(valG);
          cmpbuf2[inp] = static_cast<uint8_t>((valR - valG) & 0xff);
          cmpbuf3[inp] = static_cast<uint8_t>(valA);
        }
        inp++;
      }
      // copy current row to upper_line for next iteration
      for (uint32_t x = 0; x < src.width; ++x) {
        if (src.channels == 4) {
          upper_line[x*4 + 0] = s[x*4 + 0];
          upper_line[x*4 + 1] = s[x*4 + 1];
          upper_line[x*4 + 2] = s[x*4 + 2];
          upper_line[x*4 + 3] = s[x*4 + 3];
        } else {
          upper_line[x*4 + 0] = 255;
          upper_line[x*4 + 1] = s[x*3 + 0];
          upper_line[x*4 + 2] = s[x*3 + 1];
          upper_line[x*4 + 3] = s[x*3 + 2];
        }
      }
      // inp is already advanced per pixel
    }

    // write stored blocks
    uint32_t blocksize = 0;
    auto write_plane = [&](const uint8_t* p){ uint8_t meth=1; fwrite(&meth,1,1,fp); write_u32le(fp,(uint32_t)inp); fwrite(p,1,inp,fp); blocksize += 1+4+(uint32_t)inp; };
    write_plane(cmpbuf0.data());
    write_plane(cmpbuf1.data());
    write_plane(cmpbuf2.data());
    if (desired_colors == 4) write_plane(cmpbuf3.data());

    blocksizes[block_index] = blocksize;
  }

  // patch block sizes
  long endpos = ftell(fp);
  fseek(fp, blocksizepos, SEEK_SET);
  for (uint32_t i = 0; i < blockcount; ++i) write_u32le(fp, blocksizes[i]);
  fseek(fp, endpos, SEEK_SET);

  return true;
}
} // namespace tlg::v5
