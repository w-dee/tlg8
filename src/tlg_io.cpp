// Minimal TLG5/6 loader (decoder). For now, implement TLG5; TLG6 to follow.
#include "image_io.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <climits>
#include <limits>

static bool read_exact(FILE* fp, void* buf, size_t n) {
  return fread(buf, 1, n, fp) == n;
}
static uint32_t read_u32le(FILE* fp) {
  uint8_t b[4];
  if (!read_exact(fp, b, 4)) return 0; // caller should check ferror/feof via failure path
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// ====== TLG6 decoding helpers (adapted from existing decoder) ======
namespace tlg6 {
static const int GOLOMB_N_COUNT = 4;
static const int W_BLOCK_SIZE = 8;
static const int H_BLOCK_SIZE = 8;

static short const TLG6GolombCompressed[GOLOMB_N_COUNT][9] = {
  {3,7,15,27,63,108,223,448,130,},
  {3,5,13,24,51,95,192,384,257,},
  {2,5,12,21,39,86,155,320,384,},
  {2,3,9,18,33,61,129,258,511,},
};
static unsigned char TLG6GolombBitLengthTable[GOLOMB_N_COUNT*2*128][GOLOMB_N_COUNT];
static uint32_t TLG6GolombCodeTable[256][8];
static bool tables_inited = false;

static inline void init_tables() {
  if (tables_inited) return;
  tables_inited = true;
  int a;
  for (int n = 0; n < GOLOMB_N_COUNT; n++) {
    a = 0;
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < TLG6GolombCompressed[n][i]; j++)
        TLG6GolombBitLengthTable[a++][n] = (unsigned char)i;
    }
  }
  for (int v = 0; v < 256; v++) {
    int firstbit = 0;
    for (int i = 0; i < 8; i++) if (v & (1<<i)) { firstbit = i + 1; break; }
    for (int k = 0; k < 8; k++) {
      if (firstbit == 0 || firstbit + k > 8) { TLG6GolombCodeTable[v][k] = 0; continue; }
      int n = ((v >> firstbit) & ((1<<k)-1)) + ((firstbit-1)<<k);
      int sign = (n&1) - 1;
      n >>= 1;
      TLG6GolombCodeTable[v][k] = ((firstbit + k) << 8) + (n << 16) + (((n ^ sign) + sign + 1) & 0xff);
    }
  }
}

// bit helpers for golomb decoding
static inline int bsf32(uint32_t x){ return __builtin_ctz(x); }

template<typename ACCESS_T>
static inline void DecodeGolombValues(int8_t* pixelbuf, int pixel_count, uint8_t* bit_pool) {
  int n = GOLOMB_N_COUNT - 1;
  int a = 0;
  int bit_pos = 0;
  uint32_t bits = 0;
  auto FILL_BITS = [&](){ while (bit_pos <= 24) { bits += ((uint32_t)(*(bit_pool++)) << bit_pos); bit_pos += 8; } };
  FILL_BITS();
  bool first_is_nonzero = (bits & 1); bits >>= 1; bit_pos -= 1; FILL_BITS();
  int8_t* limit = pixelbuf + pixel_count*4;
  if (first_is_nonzero) goto nonzero;
  while (true) {
    int count;
    { int b = bsf32(bits); bits >>= (b+1); bit_pos -= (b+1); FILL_BITS(); count = 1 << b; count += (bits) & (count-1); bits >>= b; bit_pos -= b; FILL_BITS(); }
    if (sizeof(ACCESS_T) == sizeof(uint32_t)) { do { *(ACCESS_T*)pixelbuf = 0; pixelbuf += 4; } while(--count); }
    else { pixelbuf += count * (int)sizeof(uint32_t); }
    if (pixelbuf >= limit) break;
nonzero:
    { int b = bsf32(bits); bits >>= (b+1); bit_pos -= (b+1); FILL_BITS(); int count0 = 1 << b; count0 += (bits) & (count0-1); bits >>= b; bit_pos -= b; FILL_BITS(); count = count0; }
    do {
      int k = TLG6GolombBitLengthTable[a][n];
      int v = TLG6GolombCodeTable[bits & 0xff][k];
      if (v) {
        int b = (v >> 8) & 0xff; int add = (v >> 16);
        a += add; bits >>= b; bit_pos -= b; FILL_BITS(); *(ACCESS_T*)pixelbuf = (unsigned char)v; 
      } else {
        int bit_count;
        if (bits) { bit_count = bsf32(bits); bits >>= bit_count; bit_pos -= bit_count; bits >>= 1; bit_pos -= 1; FILL_BITS(); }
        else { bits = 0; bit_pos = 0; bit_count = *(bit_pool++); FILL_BITS(); }
        int vv = (bit_count << k) + (bits & ((1<<k)-1)); bits >>= k; bit_pos -= k; FILL_BITS();
        int sign = (vv & 1) - 1; vv >>= 1; a += vv; *(ACCESS_T*)pixelbuf = (unsigned char)((vv ^ sign) + sign + 1);
      }
      pixelbuf += 4; if(--n < 0) { a >>= 1; n = GOLOMB_N_COUNT - 1; }
    } while(--count);
    if (pixelbuf >= limit) break;
  }
}

static inline void TLG6DecodeGolombValuesForFirst(int8_t* pixelbuf, int pixel_count, uint8_t* bit_pool){
  DecodeGolombValues<uint32_t>(pixelbuf, pixel_count, bit_pool);
}
static inline void TLG6DecodeGolombValuesNext(int8_t* pixelbuf, int pixel_count, uint8_t* bit_pool){
  DecodeGolombValues<uint8_t>(pixelbuf, pixel_count, bit_pool);
}

// predictors and filters (generic)
static inline uint32_t make_gt_mask(uint32_t a, uint32_t b){
  uint32_t tmp2 = ~b;
  uint32_t tmp = ((a & tmp2) + (((a ^ tmp2) >> 1) & 0x7f7f7f7f) ) & 0x80808080;
  tmp = ((tmp >> 7) + 0x7f7f7f7f) ^ 0x7f7f7f7f;
  return tmp;
}
static inline uint32_t packed_bytes_add(uint32_t a, uint32_t b){
  uint32_t tmp = (((a & b)<<1) + ((a ^ b) & 0xfefefefe) ) & 0x01010100;
  return a+b-tmp;
}
static inline uint32_t med2(uint32_t a, uint32_t b, uint32_t c){
  uint32_t aa_gt_bb = make_gt_mask(a, b);
  uint32_t a_xor_b_and_aa_gt_bb = ((a ^ b) & aa_gt_bb);
  uint32_t aa = a_xor_b_and_aa_gt_bb ^ a;
  uint32_t bb = a_xor_b_and_aa_gt_bb ^ b;
  uint32_t n = make_gt_mask(c, bb);
  uint32_t nn = make_gt_mask(aa, c);
  uint32_t m = ~(n | nn);
  return (n & aa) | (nn & bb) | ((bb & m) - (c & m) + (aa & m));
}
static inline uint32_t med(uint32_t a, uint32_t b, uint32_t c, uint32_t v){
  return packed_bytes_add(med2(a, b, c), v);
}
#define AVG_PACKED(x,y) ((((x) & (y)) + ((((x) ^ (y)) & 0xfefefefe) >> 1)) + (((x)^(y))&0x01010101))
static inline uint32_t avg(uint32_t a, uint32_t b, uint32_t /*c*/, uint32_t v){
  return packed_bytes_add(AVG_PACKED(a, b), v);
}

#define DO_CHROMA_DECODE_PROTO(B,G,R,A,POST) do { \
  uint32_t i = *in; uint8_t IB=i; uint8_t IG=i>>8; uint8_t IR=i>>16; uint8_t IA=i>>24; \
  uint32_t u = *prevline; \
  p = med(p, u, up, ((uint32_t)(R)<<16 & 0xff0000) + ((uint32_t)(G)<<8 & 0x00ff00) + ((uint32_t)(B) & 0x0000ff) + ((uint32_t)(A)<<24)); \
  up = u; *curline = p; curline++; prevline++; POST; \
} while(--w);

#define DO_CHROMA_DECODE_PROTO2(B,G,R,A,POST) do { \
  uint32_t i = *in; uint8_t IB=i; uint8_t IG=i>>8; uint8_t IR=i>>16; uint8_t IA=i>>24; \
  uint32_t u = *prevline; \
  p = avg(p, u, up, ((uint32_t)(R)<<16 & 0xff0000) + ((uint32_t)(G)<<8 & 0x00ff00) + ((uint32_t)(B) & 0x0000ff) + ((uint32_t)(A)<<24)); \
  up = u; *curline = p; curline++; prevline++; POST; \
} while(--w);

#define DO_CHROMA_DECODE(N,R,G,B) case ((N)<<1): \
  DO_CHROMA_DECODE_PROTO(R,G,B,IA,{in+=step;}) break; \
  case ((N)<<1)+1: \
  DO_CHROMA_DECODE_PROTO2(R,G,B,IA,{in+=step;}) break;

static void DecodeLineGeneric(uint32_t* prevline, uint32_t* curline, int width, int start_block, int block_limit,
                              uint8_t* filtertypes, int skipblockbytes, uint32_t* in, uint32_t initialp, int oddskip, int dir) {
  uint32_t p, up; int step, i;
  if (start_block) { prevline += start_block * W_BLOCK_SIZE; curline += start_block * W_BLOCK_SIZE; p = curline[-1]; up = prevline[-1]; }
  else { p = up = initialp; }
  in += skipblockbytes * start_block; step = (dir&1)?1:-1;
  for (i = start_block; i < block_limit; i++) {
    int w = width - i*W_BLOCK_SIZE; if (w > W_BLOCK_SIZE) w = W_BLOCK_SIZE; int ww = w; if (step==-1) in += ww-1; if (i&1) in += oddskip * ww;
    switch(filtertypes[i]) {
      DO_CHROMA_DECODE(0, IB, IG, IR);
      DO_CHROMA_DECODE(1, IB+IG, IG, IR+IG);
      DO_CHROMA_DECODE(2, IB, IG+IB, IR+IB+IG);
      DO_CHROMA_DECODE(3, IB+IR+IG, IG+IR, IR);
      DO_CHROMA_DECODE(4, IB+IR, IG+IB+IR, IR+IB+IR+IG);
      DO_CHROMA_DECODE(5, IB+IR, IG+IB+IR, IR);
      DO_CHROMA_DECODE(6, IB+IG, IG, IR);
      DO_CHROMA_DECODE(7, IB, IG+IB, IR);
      DO_CHROMA_DECODE(8, IB, IG, IR+IG);
      DO_CHROMA_DECODE(9, IB+IG+IR+IB, IG+IR+IB, IR+IB);
      DO_CHROMA_DECODE(10, IB+IR, IG+IR, IR);
      DO_CHROMA_DECODE(11, IB, IG+IB, IR+IB);
      DO_CHROMA_DECODE(12, IB, IG+IR+IB, IR+IB);
      DO_CHROMA_DECODE(13, IB+IG, IG+IR+IB+IG, IR+IB+IG);
      DO_CHROMA_DECODE(14, IB+IG+IR, IG+IR, IR+IB+IG+IR);
      DO_CHROMA_DECODE(15, IB, IG+(IB<<1), IR+(IB<<1));
      default: break;
    }
    if (step == 1) in += skipblockbytes - ww; else in += skipblockbytes + 1; if (i&1) in -= oddskip * ww;
  }
}

static void DecodeLine(uint32_t* prevline, uint32_t* curline, int block_count, uint8_t* filtertypes, int skipblockbytes, uint32_t* in, uint32_t initialp, int oddskip, int dir) {
  DecodeLineGeneric(prevline, curline, INT32_MAX, 0, block_count, filtertypes, skipblockbytes, in, initialp, oddskip, dir);
}
} // namespace tlg6

static int tlg5_lzss_decompress(uint8_t* out, const uint8_t* in, int insize, uint8_t* text, int r) {
  const uint8_t* inlim = in + insize;
  if (in >= inlim) return r;
  uint32_t flags;
getmore:
  flags = ((uint32_t)(*in++)) | 0x100u;
loop:
  {
    bool backref = (flags & 1u) != 0;
    flags >>= 1;
    if (!flags) {
      if (in >= inlim) return r;
      goto getmore;
    }
    if (backref) {
      uint16_t word = (uint16_t)(in[0] | ((uint16_t)in[1] << 8));
      in += 2;
      int mpos = word & 0x0fff;
      int mlen = word >> 12;
      if (mlen == 15) mlen += *in++;
      mlen += 3;
      do {
        uint8_t c = text[mpos];
        mpos = (mpos + 1) & (4096 - 1);
        text[r] = c; *out++ = c; r = (r + 1) & (4096 - 1);
      } while (--mlen);
    } else {
      uint8_t c = *in++;
      text[r] = c; *out++ = c; r = (r + 1) & (4096 - 1);
    }
  }
  if (in < inlim) goto loop;
  return r;
}

static bool tlg5_decode_stream(FILE* fp, PixelBuffer &out, std::string &err) {
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

static bool tlg6_decode_stream(FILE* fp, PixelBuffer &out, std::string &err) {
  using namespace tlg6;
  init_tables();
  // Read 4 control bytes; tolerate an extra 0x00 after header (some encoders write 12-byte mark)
  unsigned char hdr4[4];
  if (!read_exact(fp, hdr4, 4)) { err = "tlg6: read header"; return false; }
  int colors; unsigned char f1, f2, f3;
  if ((hdr4[0] == 0) && (hdr4[1] == 1 || hdr4[1] == 3 || hdr4[1] == 4)) {
    // Likely an extra 0x00 after mark; shift
    colors = hdr4[1]; f1 = hdr4[2]; f2 = hdr4[3];
    if (!read_exact(fp, &f3, 1)) { err = "tlg6: read flags"; return false; }
  } else {
    colors = hdr4[0]; f1 = hdr4[1]; f2 = hdr4[2]; f3 = hdr4[3];
  }
  if (!(colors==1 || colors==3 || colors==4)) { err = "tlg6: bad colors"; return false; }
  if (f1 != 0 || f2 != 0 || f3 != 0) { err = "tlg6: unsupported flags"; return false; }
  int width = (int)read_u32le(fp); int height = (int)read_u32le(fp);
  int max_bit_length = (int)read_u32le(fp);
  if (width<=0 || height<=0) { err = "tlg6: invalid dims"; return false; }

  int x_block_count = (width - 1) / W_BLOCK_SIZE + 1;
  int y_block_count = (height - 1) / H_BLOCK_SIZE + 1;
  int main_count = width / W_BLOCK_SIZE;
  int fraction = width - main_count * W_BLOCK_SIZE;

  std::vector<uint8_t> bit_pool((size_t)max_bit_length / 8 + 5);
  std::vector<uint32_t> pixelbuf((size_t)width * H_BLOCK_SIZE + 1);
  std::vector<uint8_t> filter_types((size_t)x_block_count * y_block_count);
  std::vector<uint32_t> zeroline(width);
  std::vector<uint8_t> lzss_text(4096);

  for (int x = 0; x < width; x++) zeroline[x] = (colors==3?0xff000000u:0u);
  // init LZSS_text pattern
  {
    uint32_t* p = reinterpret_cast<uint32_t*>(lzss_text.data());
    for(uint32_t i = 0; i < 32*0x01010101u; i+=0x01010101u) {
      for(uint32_t j = 0; j < 16*0x01010101u; j+=0x01010101u) { p[0] = i; p[1] = j; p += 2; }
    }
  }
  // read filter types (compressed via TLG5 LZSS)
  int inbuf_size = (int)read_u32le(fp); if (inbuf_size <= 0) { err = "tlg6: bad filter size"; return false; }
  std::vector<uint8_t> inbuf(inbuf_size);
  if (!read_exact(fp, inbuf.data(), inbuf_size)) { err = "tlg6: read filter"; return false; }
  int filter_r = tlg5_lzss_decompress(filter_types.data(), inbuf.data(), inbuf_size, lzss_text.data(), 0);
  if (filter_r < 0) {
    std::fill(filter_types.begin(), filter_types.end(), 0);
  }

  out.width = width; out.height = height; out.channels = 4; out.data.resize((size_t)width*height*4);
  uint32_t* prevline = zeroline.data();

  bool needs_bgra_swizzle = false;
  for (int y = 0; y < height; y += H_BLOCK_SIZE) {
    int ylim = y + H_BLOCK_SIZE; if (ylim >= height) ylim = height;
    int pixel_count = (ylim - y) * width;
    bool raw_mode = false;
    std::vector<std::vector<uint8_t>> raw_components(colors);
    for (int c = 0; c < colors; c++) {
      int bit_length = (int)read_u32le(fp);
      int method = (bit_length >> 30) & 3;
      bit_length &= 0x3fffffff;
      int byte_length = bit_length / 8; if (bit_length % 8) byte_length++;
      if (method == 0) {
        if (byte_length < 0 || byte_length > (int)bit_pool.size()) bit_pool.resize(byte_length);
        if (!read_exact(fp, bit_pool.data(), byte_length)) { err = "tlg6: read bitpool"; return false; }
        if (c == 0 && colors != 1) TLG6DecodeGolombValuesForFirst((int8_t*)pixelbuf.data(), pixel_count, bit_pool.data());
        else TLG6DecodeGolombValuesNext((int8_t*)pixelbuf.data() + c, pixel_count, bit_pool.data());
        needs_bgra_swizzle = true;
      } else if (method == 3) {
        raw_mode = true;
        raw_components[c].resize(pixel_count, 0);
        if (byte_length != pixel_count) { err = "tlg6: raw block size mismatch"; return false; }
        if (!read_exact(fp, raw_components[c].data(), byte_length)) { err = "tlg6: read raw"; return false; }
      } else {
        err = "tlg6: unsupported entropy method";
        return false;
      }
    }

    if (raw_mode) {
      // ensure all component arrays are populated
      for (int c = 0; c < colors; ++c) {
        if (raw_components[c].empty()) raw_components[c].assign(pixel_count, (c == 3) ? 255 : 0);
      }
      for (int yy = y; yy < ylim; ++yy) {
        for (int x = 0; x < width; ++x) {
          size_t idx = (size_t)(yy - y) * width + x;
          uint8_t b = colors >= 1 ? raw_components[0][idx] : 0;
          uint8_t g = colors >= 2 ? raw_components[1][idx] : b;
          uint8_t r = colors >= 3 ? raw_components[2][idx] : g;
          uint8_t a = (colors == 4) ? raw_components[3][idx] : 255;
          uint8_t* outp = &out.data[((size_t)yy * width + x) * 4];
          outp[0] = a; outp[1] = r; outp[2] = g; outp[3] = b;
        }
      }
      prevline = reinterpret_cast<uint32_t*>(&out.data[((size_t)ylim - 1) * width * 4]);
      continue;
    }

    uint8_t* ft = filter_types.data() + (y / H_BLOCK_SIZE) * x_block_count;
    int skipbytes = (ylim - y) * W_BLOCK_SIZE;
    for (int yy = y; yy < ylim; yy++) {
      uint32_t* curline = reinterpret_cast<uint32_t*>(&out.data[((size_t)yy * width) * 4]);
      int dir = (yy&1)^1;
      int oddskip = ((ylim - yy - 1) - (yy - y));
      if (main_count) {
        int start = (((width < W_BLOCK_SIZE) ? width : W_BLOCK_SIZE) * (yy - y));
        DecodeLine(prevline, curline, main_count, ft, skipbytes, pixelbuf.data() + start, colors==3?0xff000000u:0u, oddskip, dir);
      }
      if (main_count != x_block_count) {
        int ww = fraction; if (ww > W_BLOCK_SIZE) ww = W_BLOCK_SIZE; int start = ww * (yy - y);
        DecodeLineGeneric(prevline, curline, width, main_count, x_block_count, ft, skipbytes, pixelbuf.data() + start, colors==3?0xff000000u:0u, oddskip, dir);
      }
      prevline = curline;
    }
  }
  if (needs_bgra_swizzle) {
    uint8_t* d = out.data.data();
    size_t total = (size_t)width * height;
    for (size_t i = 0; i < total; ++i) {
      uint8_t b = d[i*4 + 0];
      uint8_t g = d[i*4 + 1];
      uint8_t r = d[i*4 + 2];
      uint8_t a = d[i*4 + 3];
      d[i*4 + 0] = a;
      d[i*4 + 1] = r;
      d[i*4 + 2] = g;
      d[i*4 + 3] = b;
    }
  }
  return true;
}

// ====== TLG6 encoding (minimal, filter type 0 with MED predictor) ======
namespace tlg6enc {
// Reuse predictor helpers from decoder namespace
using namespace tlg6;
constexpr int MAX_COLOR_COMPONENTS = 4;
static inline void write_u32le_fp(FILE* fp, uint32_t v){ uint8_t b[4]={(uint8_t)(v&0xff),(uint8_t)((v>>8)&0xff),(uint8_t)((v>>16)&0xff),(uint8_t)((v>>24)&0xff)}; fwrite(b,1,4,fp);} 

static inline void fetch_argb(const PixelBuffer &src, uint32_t px, uint32_t py,
                              uint8_t &a, uint8_t &r, uint8_t &g, uint8_t &b) {
  size_t idx = static_cast<size_t>(py) * src.width + px;
  if (src.channels == 4) {
    a = src.data[idx*4 + 0];
    r = src.data[idx*4 + 1];
    g = src.data[idx*4 + 2];
    b = src.data[idx*4 + 3];
  } else {
    a = 255;
    r = src.data[idx*3 + 0];
    g = src.data[idx*3 + 1];
    b = src.data[idx*3 + 2];
  }
}

// Filter type compressor using LZSS (copied minimal version from core)
class SlideCompressor {
  struct Chain { int Prev; int Next; };
  static const int SLIDE_N = 4096;
  static const int SLIDE_M = (18+255);
  uint8_t Text[SLIDE_N + SLIDE_M - 1];
  int Map[256*256];
  Chain Chains[SLIDE_N];
  int S;
  void AddMap(int p){ int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8);
    if(Map[place] == -1){ Map[place] = p; } else { int old = Map[place]; Map[place] = p; Chains[old].Prev = p; Chains[p].Next = old; Chains[p].Prev = -1; } }
  void DeleteMap(int p){ int n; if((n = Chains[p].Next) != -1) Chains[n].Prev = Chains[p].Prev; if((n = Chains[p].Prev) != -1){ Chains[n].Next = Chains[p].Next; } else if(Chains[p].Next != -1){ int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8); Map[place] = Chains[p].Next; } else { int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8); Map[place] = -1; } Chains[p].Prev = -1; Chains[p].Next = -1; }
  int GetMatch(const uint8_t*cur, int curlen, int &pos, int s){ if(curlen < 3) return 0; int place = cur[0] + ((int)cur[1] << 8); int maxlen = 0; if((place = Map[place]) != -1){ int place_org; curlen -= 1; do { place_org = place; if(s == place || s == ((place + 1) & (SLIDE_N -1))) continue; place += 2; int lim = (SLIDE_M < curlen ? SLIDE_M : curlen) + place_org; const uint8_t *c = cur + 2; if(lim >= SLIDE_N){ if(place_org <= s && s < SLIDE_N) lim = s; else if(s < (lim&(SLIDE_N-1))) lim = s + SLIDE_N; } else { if(place_org <= s && s < lim) lim = s; } while(Text[place] == *(c++) && place < lim) place++; int matchlen = place - place_org; if(matchlen > maxlen) pos = place_org, maxlen = matchlen; if(matchlen == SLIDE_M) return maxlen; } while((place = Chains[place_org].Next) != -1); } return maxlen; }
public:
  SlideCompressor(){
    S = 0;
    for(int i = 0; i < SLIDE_N + SLIDE_M - 1; ++i) Text[i] = 0;
    for(int i = 0; i < 256*256; ++i) Map[i] = -1;
    for(int i = 0; i < SLIDE_N; ++i) { Chains[i].Prev = -1; Chains[i].Next = -1; }
    for(int i = SLIDE_N - 1; i >= 0; --i) AddMap(i);
  }
  void Encode(const uint8_t *in, size_t inlen, uint8_t *out, size_t &outlen){ uint8_t code[40], codeptr=1, mask=1; code[0]=0; outlen=0; int s=S; while(inlen>0){ int pos=0; int len=GetMatch(in,(int)inlen,pos,s); if(len>=3){ code[0]|=mask; if(len>=18){ code[codeptr++]=pos & 0xff; code[codeptr++]=((pos&0xf00)>>8)|0xf0; code[codeptr++]=len-18; } else { code[codeptr++]=pos & 0xff; code[codeptr++]=((pos&0xf00)>>8)|((len-3)<<4); } while(len--){ uint8_t c = *in++; DeleteMap((s - 1) & (SLIDE_N - 1)); DeleteMap(s); if(s < SLIDE_M - 1) Text[s + SLIDE_N] = c; Text[s]=c; AddMap((s - 1) & (SLIDE_N - 1)); AddMap(s); s++; inlen--; s &= (SLIDE_N - 1); } } else { uint8_t c = *in++; DeleteMap((s - 1) & (SLIDE_N - 1)); DeleteMap(s); if(s < SLIDE_M - 1) Text[s + SLIDE_N] = c; Text[s]=c; AddMap((s - 1) & (SLIDE_N - 1)); AddMap(s); s++; inlen--; s &= (SLIDE_N - 1); code[codeptr++] = c; } mask <<= 1; if(mask==0){ for(int i=0;i<codeptr;i++) out[outlen++]=code[i]; mask=codeptr=1; code[0]=0; } } if(mask!=1){ for(int i=0;i<codeptr;i++) out[outlen++]=code[i]; } S=s; }
};

static void InitializeColorFilterCompressor(SlideCompressor &c){
  uint8_t code[4096]; uint8_t dum[4096]; uint8_t *p = code; for(int i=0;i<32;i++){ for(int j=0;j<16;j++){ p[0]=p[1]=p[2]=p[3]=(uint8_t)i; p+=4; p[0]=p[1]=p[2]=p[3]=(uint8_t)j; p+=4; } } size_t dumlen=0; c.Encode(code, 4096, dum, dumlen);
}

constexpr int FILTER_TRY_COUNT = 16;
constexpr int GOLOMB_GIVE_UP_BYTES = 4;

class TLG6BitStream {
  std::vector<uint8_t>& out_;
  std::vector<uint8_t> buffer_;
  size_t byte_pos_ = 0;
  int bit_pos_ = 0;

  void ensure_capacity() {
    if (buffer_.size() <= byte_pos_) buffer_.resize(byte_pos_ + 1, 0);
  }

public:
  explicit TLG6BitStream(std::vector<uint8_t>& out) : out_(out) {}
  ~TLG6BitStream() { Flush(); }

  int GetBitPos() const { return bit_pos_; }
  size_t GetBytePos() const { return byte_pos_; }
  size_t GetBitLength() const { return byte_pos_ * 8 + bit_pos_; }

  void Put1Bit(bool bit) {
    ensure_capacity();
    if (bit) buffer_[byte_pos_] |= static_cast<uint8_t>(1u << bit_pos_);
    bit_pos_++;
    if (bit_pos_ == 8) {
      bit_pos_ = 0;
      byte_pos_++;
    }
  }

  void PutValue(long value, int len) {
    for (int i = 0; i < len; ++i) Put1Bit((value >> i) & 1);
  }

  void PutGamma(int v) {
    if (v <= 0) return;
    int t = v;
    t >>= 1;
    int cnt = 0;
    while (t) {
      Put1Bit(0);
      t >>= 1;
      cnt++;
    }
    Put1Bit(1);
    while (cnt--) {
      Put1Bit(v & 1);
      v >>= 1;
    }
  }

  void Flush() {
    size_t bytes = byte_pos_ + (bit_pos_ ? 1 : 0);
    if (bytes) {
      ensure_capacity();
      out_.insert(out_.end(), buffer_.begin(), buffer_.begin() + bytes);
    }
    buffer_.clear();
    byte_pos_ = 0;
    bit_pos_ = 0;
  }
};

static inline std::vector<int8_t> to_signed(const std::vector<uint8_t>& src) {
  std::vector<int8_t> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<int8_t>(src[i]);
  return dst;
}

static void compress_values_golomb(TLG6BitStream &bs, const std::vector<int8_t>& buf) {
  if (buf.empty()) return;
  bs.PutValue(buf[0] ? 1 : 0, 1);

  int n = GOLOMB_N_COUNT - 1;
  int a = 0;
  int count = 0;
  const size_t size = buf.size();

  for (size_t i = 0; i < size; ++i) {
    int e = buf[i];
    if (e != 0) {
      if (count) {
        bs.PutGamma(count);
        count = 0;
      }

      size_t ii = i;
      while (ii < size && buf[ii] != 0) ++ii;
      size_t nonzero = ii - i;
      bs.PutGamma(static_cast<int>(nonzero));

      for (; i < ii; ++i) {
        e = buf[i];
        int m = ((e >= 0) ? 2 * e : -2 * e - 1) - 1;
        if (m < 0) m = 0;
        int k = TLG6GolombBitLengthTable[a][n];
        size_t store_limit = bs.GetBytePos() + GOLOMB_GIVE_UP_BYTES;
        bool put_one = true;
        int q = (k > 0) ? (m >> k) : m;
        for (; q > 0; --q) {
          if (bs.GetBytePos() >= store_limit) {
            bs.PutValue(m >> k, 8);
            put_one = false;
            break;
          }
          bs.Put1Bit(0);
        }
        if (put_one && bs.GetBytePos() >= store_limit) {
          bs.PutValue(m >> k, 8);
          put_one = false;
        }
        if (put_one) bs.Put1Bit(1);
        if (k) bs.PutValue(m & ((1 << k) - 1), k);
        a += (m >> 1);
        if (--n < 0) { a >>= 1; n = GOLOMB_N_COUNT - 1; }
      }
      i = ii - 1;
    } else {
      count++;
    }
  }

  if (count) bs.PutGamma(count);
}

static size_t estimate_golomb_bits(const std::vector<uint8_t>& data) {
  std::vector<uint8_t> dummy;
  TLG6BitStream bs(dummy);
  compress_values_golomb(bs, to_signed(data));
  return bs.GetBitLength();
}

static void apply_color_filter(int code,
                               std::vector<uint8_t>& b,
                               std::vector<uint8_t>& g,
                               std::vector<uint8_t>& r) {
  const size_t n = b.size();
  switch (code) {
    case 0:
      return;
    case 1:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 2:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 3:
      for (size_t i = 0; i < n; ++i) {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 4:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 5:
      for (size_t i = 0; i < n; ++i) {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 6:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - g[i]);
      break;
    case 7:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - b[i]);
      break;
    case 8:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - g[i]);
      break;
    case 9:
      for (size_t i = 0; i < n; ++i) {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 10:
      for (size_t i = 0; i < n; ++i) {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 11:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 12:
      for (size_t i = 0; i < n; ++i) {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 13:
      for (size_t i = 0; i < n; ++i) {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 14:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 15:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(b[i] << 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 16:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - r[i]);
      break;
    case 17:
      for (size_t i = 0; i < n; ++i) {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 18:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - b[i]);
      break;
    case 19:
      for (size_t i = 0; i < n; ++i) {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 20:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - r[i]);
      break;
    case 21:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - (g[i] >> 1));
      break;
    case 22:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - (b[i] >> 1));
      break;
    case 23:
      for (size_t i = 0; i < n; ++i) {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 24:
      for (size_t i = 0; i < n; ++i) {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 25:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - (r[i] >> 1));
      break;
    case 26:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - (g[i] >> 1));
      break;
    case 27:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(r[i] >> 1);
        g[i] = static_cast<uint8_t>(g[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 28:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - (b[i] >> 1));
      break;
    case 29:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(g[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 30:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(b[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 31:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - (r[i] >> 1));
      break;
    case 32:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - (b[i] << 1));
      break;
    case 33:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - (g[i] << 1));
      break;
    case 34:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(r[i] << 1);
        g[i] = static_cast<uint8_t>(g[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 35:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - (b[i] << 1));
      break;
    case 36:
      for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint8_t>(r[i] - (g[i] << 1));
      break;
    case 37:
      for (size_t i = 0; i < n; ++i) {
        uint8_t t = static_cast<uint8_t>(g[i] << 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 38:
      for (size_t i = 0; i < n; ++i) g[i] = static_cast<uint8_t>(g[i] - (r[i] << 1));
      break;
    case 39:
      for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(b[i] - (r[i] << 1));
      break;
    default:
      break;
  }
}

static int detect_color_filter(const std::vector<uint8_t>& b,
                               const std::vector<uint8_t>& g,
                               const std::vector<uint8_t>& r,
                               std::vector<uint8_t>& out_b,
                               std::vector<uint8_t>& out_g,
                               std::vector<uint8_t>& out_r)
{
  size_t best_bits = std::numeric_limits<size_t>::max();
  int best_code = 0;
  std::vector<uint8_t> tb, tg, tr;
  tb.reserve(b.size()); tg.reserve(g.size()); tr.reserve(r.size());
  for (int code = 0; code < FILTER_TRY_COUNT; ++code) {
    tb.assign(b.begin(), b.end());
    tg.assign(g.begin(), g.end());
    tr.assign(r.begin(), r.end());
    apply_color_filter(code, tb, tg, tr);
    size_t bits = estimate_golomb_bits(tb) + estimate_golomb_bits(tg) + estimate_golomb_bits(tr);
    if (bits < best_bits) {
      best_bits = bits;
      best_code = code;
      out_b = std::move(tb);
      out_g = std::move(tg);
      out_r = std::move(tr);
    }
  }
  return best_code;
}

static bool write_tlg6_raw(FILE* fp, const PixelBuffer &src, int colors, std::string &err){
  err.clear();
  if (colors < 1 || colors > 4) { err = "unsupported color count"; return false; }

  init_tables();

  unsigned char mark[11] = { 'T','L','G','6','.','0',0,'r','a','w',0x1a };
  fwrite(mark, 1, sizeof(mark), fp);
  uint8_t cbyte = static_cast<uint8_t>(colors);
  fwrite(&cbyte, 1, 1, fp);
  uint8_t zeros[3] = {0,0,0};
  fwrite(zeros, 1, 3, fp);
  write_u32le_fp(fp, src.width);
  write_u32le_fp(fp, src.height);

  const int x_block_count = (int)((src.width - 1) / W_BLOCK_SIZE) + 1;
  const int y_block_count = (int)((src.height - 1) / H_BLOCK_SIZE) + 1;

  const size_t block_capacity = (size_t)W_BLOCK_SIZE * H_BLOCK_SIZE;
  const size_t block_group_capacity = (size_t)H_BLOCK_SIZE * src.width;

  std::vector<uint32_t> argb((size_t)src.width * src.height);
  for (uint32_t y = 0; y < src.height; ++y) {
    for (uint32_t x = 0; x < src.width; ++x) {
      uint8_t a, r, g, b;
      fetch_argb(src, x, y, a, r, g, b);
      argb[(size_t)y * src.width + x] = ((uint32_t)a<<24) | ((uint32_t)r<<16) | ((uint32_t)g<<8) | (uint32_t)b;
    }
  }

  std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> buf;
  std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> block_buf;
  for (int c = 0; c < colors; ++c) {
    buf[c].resize(block_capacity * 3);
    block_buf[c].resize(block_group_capacity);
  }

  std::vector<uint32_t> zeroline(src.width, colors == 3 ? 0xff000000u : 0u);
  std::vector<uint8_t> filtertypes((size_t)x_block_count * y_block_count, 0);
  std::vector<uint8_t> temp_storage;
  temp_storage.reserve((size_t)src.width * src.height);

  int max_bit_length = 0;
  size_t filter_index = 0;

  for (uint32_t y = 0; y < src.height; y += H_BLOCK_SIZE) {
    uint32_t ylim = std::min<uint32_t>(y + H_BLOCK_SIZE, src.height);
    uint32_t block_h = ylim - y;
    uint32_t pixel_count = block_h * src.width;

    for (int c = 0; c < colors; ++c) {
      std::fill(block_buf[c].begin(), block_buf[c].begin() + pixel_count, 0);
    }

    size_t gwp = 0;
    for (uint32_t x = 0, xp = 0; x < src.width; x += W_BLOCK_SIZE, ++xp) {
      uint32_t xlim = std::min<uint32_t>(x + W_BLOCK_SIZE, src.width);
      uint32_t bw = xlim - x;
      size_t block_pixels = static_cast<size_t>(bw) * block_h;

      struct Candidate {
        int predictor = 0;
        int filter = 0;
        size_t total_bits = std::numeric_limits<size_t>::max();
        std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> comps;
      };

      std::array<Candidate, 2> candidates;

      for (int predictor = 0; predictor < 2; ++predictor) {
        for (int c = 0; c < colors; ++c) {
          size_t wp = 0;
          for (uint32_t yy = y; yy < ylim; ++yy) {
            const uint32_t* sl = &argb[yy * src.width];
            const uint32_t* upper_line = (yy == 0) ? zeroline.data() : &argb[(yy - 1) * src.width];
            for (uint32_t xx = x; xx < xlim; ++xx) {
              uint8_t px = static_cast<uint8_t>((sl[xx] >> (c*8)) & 0xff);
              uint8_t pa = (xx > 0) ? static_cast<uint8_t>((sl[xx-1] >> (c*8)) & 0xff) : 0;
              uint8_t pb = static_cast<uint8_t>((upper_line[xx] >> (c*8)) & 0xff);
              uint8_t pc = (xx > 0) ? static_cast<uint8_t>((upper_line[xx-1] >> (c*8)) & 0xff) : 0;
              uint8_t py;
              if (predictor == 0) {
                uint8_t min_ab = pa < pb ? pa : pb;
                uint8_t max_ab = pa > pb ? pa : pb;
                if (pc >= max_ab) py = min_ab;
                else if (pc < min_ab) py = max_ab;
                else py = static_cast<uint8_t>(pa + pb - pc);
              } else {
                py = static_cast<uint8_t>((pa + pb + 1) >> 1);
              }
              buf[c][wp] = static_cast<uint8_t>((static_cast<int>(px) - static_cast<int>(py)) & 0xff);
              wp++;
            }
          }
        }

        int dbofs = (predictor + 1) * (int)block_capacity;
        int wp = 0;
        for (uint32_t yy = y; yy < ylim; ++yy) {
          int ofs;
          if (!(xp & 1)) ofs = (int)((yy - y) * bw);
          else ofs = (int)((ylim - yy - 1) * bw);
          bool dir;
          if (!((block_h) & 1)) dir = ((yy & 1) ^ (xp & 1));
          else {
            if (xp & 1) dir = (yy & 1);
            else dir = ((yy & 1) ^ (xp & 1));
          }
          if (!dir) {
            for (uint32_t xx = 0; xx < bw; ++xx) {
              for (int c = 0; c < colors; ++c) buf[c][dbofs + wp] = buf[c][ofs + xx];
              wp++;
            }
          } else {
            for (int xx = (int)bw - 1; xx >= 0; --xx) {
              for (int c = 0; c < colors; ++c) buf[c][dbofs + wp] = buf[c][ofs + xx];
              wp++;
            }
          }
        }

        Candidate cand;
        cand.predictor = predictor;
        cand.filter = 0;
        for (int c = 0; c < colors; ++c) {
          cand.comps[c].assign(buf[c].begin() + dbofs, buf[c].begin() + dbofs + block_pixels);
        }

        if (colors >= 3) {
          std::vector<uint8_t> filtered_b, filtered_g, filtered_r;
          int ft = detect_color_filter(cand.comps[0], cand.comps[1], cand.comps[2], filtered_b, filtered_g, filtered_r);
          cand.filter = ft;
          cand.comps[0] = std::move(filtered_b);
          cand.comps[1] = std::move(filtered_g);
          cand.comps[2] = std::move(filtered_r);
        }

        size_t bits = 0;
        for (int c = 0; c < colors; ++c) bits += estimate_golomb_bits(cand.comps[c]);
        cand.total_bits = bits;
        candidates[predictor] = std::move(cand);
      }

      const Candidate* best = &candidates[0];
      if (candidates[1].total_bits < best->total_bits) best = &candidates[1];

      filtertypes[filter_index++] = static_cast<uint8_t>((best->filter << 1) | best->predictor);

    for (int c = 0; c < colors; ++c) {
      auto &dst = block_buf[c];
      const auto &src_comp = best->comps[c];
      size_t base = gwp;
      for (size_t i = 0; i < src_comp.size(); ++i) dst[base + i] = src_comp[i];
    }
    gwp += block_pixels;
  }

    if (gwp != pixel_count) {
      err = "tlg6: block size mismatch";
      return false;
    }

    for (int c = 0; c < colors; ++c) {
      std::vector<int8_t> signed_data(pixel_count);
      for (size_t i = 0; i < pixel_count; ++i)
        signed_data[i] = static_cast<int8_t>(block_buf[c][i]);
      std::vector<uint8_t> bit_bytes;
      {
        TLG6BitStream bs(bit_bytes);
        compress_values_golomb(bs, signed_data);
        uint32_t bitlen = static_cast<uint32_t>(bs.GetBitLength());
        if ((int)bitlen > max_bit_length) max_bit_length = (int)bitlen;
        uint8_t header[4] = {
          static_cast<uint8_t>(bitlen & 0xff),
          static_cast<uint8_t>((bitlen >> 8) & 0xff),
          static_cast<uint8_t>((bitlen >> 16) & 0xff),
          static_cast<uint8_t>((bitlen >> 24) & 0xff)
        };
        bs.Flush();
        temp_storage.insert(temp_storage.end(), header, header + 4);
      }
      temp_storage.insert(temp_storage.end(), bit_bytes.begin(), bit_bytes.end());
    }

    const uint32_t* last_line = &argb[(ylim - 1) * src.width];
    std::copy(last_line, last_line + src.width, zeroline.begin());
  }

  if (filter_index != filtertypes.size()) {
    err = "tlg6: filter count mismatch";
    return false;
  }

  write_u32le_fp(fp, (uint32_t)max_bit_length);

  SlideCompressor filter_comp;
  InitializeColorFilterCompressor(filter_comp);
  std::vector<uint8_t> filter_out(filtertypes.empty() ? 0 : filtertypes.size() * 2 + 16);
  size_t outlen = 0;
  if (!filtertypes.empty()) filter_comp.Encode(filtertypes.data(), filtertypes.size(), filter_out.data(), outlen);
  write_u32le_fp(fp, (uint32_t)outlen);
  if (outlen) fwrite(filter_out.data(), 1, outlen, fp);

  if (!temp_storage.empty()) fwrite(temp_storage.data(), 1, temp_storage.size(), fp);
  return true;
}
} // namespace tlg6enc

static bool tlg_decode_raw(FILE* fp, const char* mark11, PixelBuffer &out, std::string &err) {
  if (std::memcmp(mark11, "TLG5.0\x00raw\x1a\x00", 11) == 0) {
    return tlg5_decode_stream(fp, out, err);
  } else if (std::memcmp(mark11, "TLG6.0\x00raw\x1a\x00", 11) == 0) {
    return tlg6_decode_stream(fp, out, err);
  } else {
    err = "invalid tlg raw header";
    return false;
  }
}

bool load_tlg(const std::string &path, PixelBuffer &out, std::string &err) {
  err.clear();
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) { err = "cannot open file"; return false; }
  char mark[11];
  if (!read_exact(fp, mark, 11)) { fclose(fp); err = "read error"; return false; }

  bool ok = false;
  if (std::memcmp(mark, "TLG0.0\x00sds\x1a\x00", 11) == 0) {
    // SDS container
    uint32_t rawlen = read_u32le(fp);
    (void)rawlen;
    char rawmark[11];
    if (!read_exact(fp, rawmark, 11)) { fclose(fp); err = "read error"; return false; }
    ok = tlg_decode_raw(fp, rawmark, out, err);
    if (ok) {
      // Skip to end of SDS if needed (ignore tags)
      // We don't strictly need to seek further for our use-case.
    }
  } else {
    // raw
    ok = tlg_decode_raw(fp, mark, out, err);
  }
  fclose(fp);
  return ok;
}

static inline void write_u32le(FILE* fp, uint32_t v) {
  uint8_t b[4] = { (uint8_t)(v & 0xff), (uint8_t)((v>>8)&0xff), (uint8_t)((v>>16)&0xff), (uint8_t)((v>>24)&0xff) };
  fwrite(b, 1, 4, fp);
}

static bool write_tlg5_raw(FILE* fp, const PixelBuffer &src, int desired_colors, std::string &err) {
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

bool save_tlg(const std::string &path, const PixelBuffer &src, const TlgOptions &opt, std::string &err) {
  err.clear();
  if (!(src.channels == 3 || src.channels == 4)) { err = "unsupported pixel channels"; return false; }

  int desired_colors;
  if (opt.fmt == ImageFormat::A8R8G8B8) desired_colors = 4;
  else if (opt.fmt == ImageFormat::R8G8B8) desired_colors = 3;
  else desired_colors = src.has_alpha() ? 4 : 3;

  if (opt.version == 6) {
    // True TLG6 RAW encoding
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { err = "cannot open file"; return false; }
    bool ok = tlg6enc::write_tlg6_raw(fp, src, desired_colors, err);
    fclose(fp);
    return ok;
  }

  // TLG5 RAW writer (stored mode only)
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) { err = "cannot open file"; return false; }
  bool ok = write_tlg5_raw(fp, src, desired_colors, err);
  fclose(fp);
  return ok;
}
