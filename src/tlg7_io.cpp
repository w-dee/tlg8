#include "tlg7_io.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "tlg_io_common.h"

namespace
{
  using tlg::detail::read_exact;
  using tlg::detail::read_u32le;
  using tlg::detail::tlg5_lzss_decompress;
}

// CAS-8 lightweight classifier + 2-choice predictor selector (C++17)
// - Deterministic on both encoder/decoder sides
// - Uses only causal neighbors: a=left, b=top, c=top-left, d=top-right, f=top-top
// - Two-choice per class; choice driven by per-predictor recent |e| (exponential decay)
// - Optional: Planar-lite (Pp) can be enabled to replace one candidate in DIAG/FLAT

#include <cstdint>
#include <algorithm>
#include <array>

namespace tlg7
{

  // ---------- Tunables ----------
  constexpr int T1 = 2; // flat threshold
  constexpr int T2 = 6; // orientation threshold (Dh-Dv or Dv-Dh)

  // Exponential decay for recent |e| scores (smaller is better)
  constexpr int ERR_DECAY_SHIFT = 2; // ~= 1/4 smoothing

  // ---------- Small helpers ----------
  template <typename T>
  inline T clip_int(int v, int lo, int hi)
  {
    return static_cast<T>(std::min(hi, std::max(lo, v)));
  }

  inline int iabs(int v) { return v < 0 ? -v : v; }

  // ---------- Classes ----------
  enum class Class : uint8_t
  {
    FLAT = 0,
    HORZ = 1,
    VERT = 2,
    DIAG = 3
  };

  // ---------- Predictor indices ----------
  enum PredId : uint8_t
  {
    P0 = 0, // a
    P1,     // b
    P2,     // (a+b+1)>>1
    P3,     // a+b-c        (MED/Paeth core)
    P4,     // a + ((b-c)>>1)
    P5,     // b + ((a-c)>>1)
    P6,     // ((a<<1)+b-c+2)>>2
    P7,     // (a+(b<<1)-c+2)>>2
    Pp,     // Planar-lite (optional)
    PRED_COUNT
  };

  // Keep recent |e| scores per predictor (lower = better).
  struct SelectorState
  {
    std::array<uint16_t, PRED_COUNT> errScore{}; // initialized to 0

    // Update after decoding one pixel with predictor pid and residual e
    inline void update(PredId pid, int e_abs)
    {
      // y[n] = ( (y[n-1] * (2^s - 1)) + e_abs ) / 2^s  ; here implemented branchlessly
      uint16_t prev = errScore[pid];
      uint16_t dec = static_cast<uint16_t>(prev - (prev >> ERR_DECAY_SHIFT));
      errScore[pid] = static_cast<uint16_t>(dec + std::min(e_abs, 0xFFFF));
    }
  };

  // ---------- Classification ----------
  inline Class classify_cas8(int a, int b, int c)
  {
    const int Dh = iabs(a - b);
    const int Dv = iabs(b - c);
    if (Dh <= T1 && Dv <= T1)
      return Class::FLAT;
    if ((Dh - Dv) >= T2)
      return Class::VERT; // vertical edge (favor left predictors)
    if ((Dv - Dh) >= T2)
      return Class::HORZ; // horizontal edge (favor top predictors)
    return Class::DIAG;
  }

  // ---------- Predictors ----------
  // NOTE: pass pixel depth (lo..hi) to enforce identical clipping on encoder/decoder
  template <typename T>
  inline int predict(PredId pid, int a, int b, int c, int d, int f, int lo, int hi)
  {
    int p;
    switch (pid)
    {
    case P0:
      p = a;
      break;
    case P1:
      p = b;
      break;
    case P2:
      p = (a + b + 1) >> 1;
      break;
    case P3:
      p = a + b - c;
      break;
    case P4:
      p = a + ((b - c) >> 1);
      break;
    case P5:
      p = b + ((a - c) >> 1);
      break;
    case P6:
      p = ((a << 1) + b - c + 2) >> 2;
      break;
    case P7:
      p = (a + (b << 1) - c + 2) >> 2;
      break;
    case Pp:
    {
      // Planar-lite (very light plane fit using 5 causal taps)
      const int Hx = (a - c) + (b - d);
      const int Vy = (b - c) + (f - b);
      p = a + ((Hx + Vy + 2) >> 2);
      break;
    }
    default:
      p = b;
      break;
    }
    return clip_int<T>(p, lo, hi);
  }

  // ---------- Two-choice mapping per class ----------
  // FLAT  -> {P2, P3}
  // VERT  -> {P0, P6}
  // HORZ  -> {P1, P7}
  // DIAG  -> {P3, best(P4,P5)} ; if Planar-lite enabled, you may replace P3 with Pp when its score is better.
  struct Cas8Config
  {
    bool enablePlanarLite = false; // if true, allow Pp to replace P3 in FLAT/DIAG when its score is better
  };

  // Decide final predictor id given neighbors and selector state.
  // Does NOT update state; call state.update(pid, abs(e)) after you decode e.
  inline PredId select_pred_cas8(const SelectorState &st, Class klass, const Cas8Config &cfg)
  {
    PredId a = P2, b = P3; // defaults for FLAT
    switch (klass)
    {
    case Class::FLAT:
      a = P2;
      b = P3;
      if (cfg.enablePlanarLite && st.errScore[Pp] + 0u < st.errScore[b] + 0u)
        b = Pp;
      break;
    case Class::VERT:
      a = P0;
      b = P6;
      break;
    case Class::HORZ:
      a = P1;
      b = P7;
      break;
    case Class::DIAG:
    {
      PredId second = (st.errScore[P4] + 0u <= st.errScore[P5] + 0u) ? P4 : P5;
      a = P3;
      b = second;
      if (cfg.enablePlanarLite && st.errScore[Pp] + 0u < st.errScore[a] + 0u)
        a = Pp;
      break;
    }
    }
    // Final 2-choice by smaller recent |e|
    return (st.errScore[a] + 0u <= st.errScore[b] + 0u) ? a : b;
  }

  // ---------- One-stop predict-or-select ----------
  // Returns (pred, chosen predictor id). T = storage type (e.g., uint8_t, uint16_t)
  template <typename T>
  inline std::pair<int, PredId>
  cas8_predict(int a, int b, int c, int d, int f,
               int lo, int hi,
               const Cas8Config &cfg,
               const SelectorState &state)
  {
    const Class cls = classify_cas8(a, b, c);
    const PredId pid = select_pred_cas8(state, cls, cfg);
    const int pred = predict<T>(pid, a, b, c, d, f, lo, hi);
    return {pred, pid};
  }

} // namespace tlg7

namespace tlg7
{
  static const int GOLOMB_N_COUNT = 4;
  static const int W_BLOCK_SIZE = 8;
  static const int H_BLOCK_SIZE = 8;

  static short const TLG7GolombCompressed[GOLOMB_N_COUNT][9] = {
      {
          3,
          7,
          15,
          27,
          63,
          108,
          223,
          448,
          130,
      },
      {
          3,
          5,
          13,
          24,
          51,
          95,
          192,
          384,
          257,
      },
      {
          2,
          5,
          12,
          21,
          39,
          86,
          155,
          320,
          384,
      },
      {
          2,
          3,
          9,
          18,
          33,
          61,
          129,
          258,
          511,
      },
  };
  static unsigned char TLG7GolombBitLengthTable[GOLOMB_N_COUNT * 2 * 128][GOLOMB_N_COUNT];
  static uint32_t TLG7GolombCodeTable[256][8];
  static bool tables_inited = false;

  static inline void init_tables()
  {
    if (tables_inited)
      return;
    tables_inited = true;
    int a;
    for (int n = 0; n < GOLOMB_N_COUNT; n++)
    {
      a = 0;
      for (int i = 0; i < 9; i++)
      {
        for (int j = 0; j < TLG7GolombCompressed[n][i]; j++)
          TLG7GolombBitLengthTable[a++][n] = (unsigned char)i;
      }
    }
    for (int v = 0; v < 256; v++)
    {
      int firstbit = 0;
      for (int i = 0; i < 8; i++)
        if (v & (1 << i))
        {
          firstbit = i + 1;
          break;
        }
      for (int k = 0; k < 8; k++)
      {
        if (firstbit == 0 || firstbit + k > 8)
        {
          TLG7GolombCodeTable[v][k] = 0;
          continue;
        }
        int n = ((v >> firstbit) & ((1 << k) - 1)) + ((firstbit - 1) << k);
        int sign = (n & 1) - 1;
        n >>= 1;
        TLG7GolombCodeTable[v][k] = ((firstbit + k) << 8) + (n << 16) + (((n ^ sign) + sign + 1) & 0xff);
      }
    }
  }

  // bit helpers for golomb decoding
  static inline int bsf32(uint32_t x) { return __builtin_ctz(x); }

  template <typename ACCESS_T>
  static inline void DecodeGolombValues(int8_t *pixelbuf, int pixel_count, uint8_t *bit_pool)
  {
    int n = GOLOMB_N_COUNT - 1;
    int a = 0;
    int bit_pos = 0;
    uint32_t bits = 0;
    auto FILL_BITS = [&]()
    { while (bit_pos <= 24) { bits += ((uint32_t)(*(bit_pool++)) << bit_pos); bit_pos += 8; } };
    FILL_BITS();
    bool first_is_nonzero = (bits & 1);
    bits >>= 1;
    bit_pos -= 1;
    FILL_BITS();
    int8_t *limit = pixelbuf + pixel_count * 4;
    if (first_is_nonzero)
      goto nonzero;
    while (true)
    {
      int count;
      {
        int b = bsf32(bits);
        bits >>= (b + 1);
        bit_pos -= (b + 1);
        FILL_BITS();
        count = 1 << b;
        count += (bits) & (count - 1);
        bits >>= b;
        bit_pos -= b;
        FILL_BITS();
      }
      if (sizeof(ACCESS_T) == sizeof(uint32_t))
      {
        do
        {
          *(ACCESS_T *)pixelbuf = 0;
          pixelbuf += 4;
        } while (--count);
      }
      else
      {
        pixelbuf += count * (int)sizeof(uint32_t);
      }
      if (pixelbuf >= limit)
        break;
    nonzero:
    {
      int b = bsf32(bits);
      bits >>= (b + 1);
      bit_pos -= (b + 1);
      FILL_BITS();
      int count0 = 1 << b;
      count0 += (bits) & (count0 - 1);
      bits >>= b;
      bit_pos -= b;
      FILL_BITS();
      count = count0;
    }
      do
      {
        int k = TLG7GolombBitLengthTable[a][n];
        int v = TLG7GolombCodeTable[bits & 0xff][k];
        if (v)
        {
          int b = (v >> 8) & 0xff;
          int add = (v >> 16);
          a += add;
          bits >>= b;
          bit_pos -= b;
          FILL_BITS();
          *(ACCESS_T *)pixelbuf = (unsigned char)v;
        }
        else
        {
          int bit_count;
          if (bits)
          {
            bit_count = bsf32(bits);
            bits >>= bit_count;
            bit_pos -= bit_count;
            bits >>= 1;
            bit_pos -= 1;
            FILL_BITS();
          }
          else
          {
            bits = 0;
            bit_pos = 0;
            bit_count = *(bit_pool++);
            FILL_BITS();
          }
          int vv = (bit_count << k) + (bits & ((1 << k) - 1));
          bits >>= k;
          bit_pos -= k;
          FILL_BITS();
          int sign = (vv & 1) - 1;
          vv >>= 1;
          a += vv;
          *(ACCESS_T *)pixelbuf = (unsigned char)((vv ^ sign) + sign + 1);
        }
        pixelbuf += 4;
        if (--n < 0)
        {
          a >>= 1;
          n = GOLOMB_N_COUNT - 1;
        }
      } while (--count);
      if (pixelbuf >= limit)
        break;
    }
  }

  static inline void TLG7DecodeGolombValuesForFirst(int8_t *pixelbuf, int pixel_count, uint8_t *bit_pool)
  {
    DecodeGolombValues<uint32_t>(pixelbuf, pixel_count, bit_pool);
  }
  static inline void TLG7DecodeGolombValuesNext(int8_t *pixelbuf, int pixel_count, uint8_t *bit_pool)
  {
    DecodeGolombValues<uint8_t>(pixelbuf, pixel_count, bit_pool);
  }

  // predictors and filters (generic)
  static inline uint32_t make_gt_mask(uint32_t a, uint32_t b)
  {
    uint32_t tmp2 = ~b;
    uint32_t tmp = ((a & tmp2) + (((a ^ tmp2) >> 1) & 0x7f7f7f7f)) & 0x80808080;
    tmp = ((tmp >> 7) + 0x7f7f7f7f) ^ 0x7f7f7f7f;
    return tmp;
  }
  static inline uint32_t packed_bytes_add(uint32_t a, uint32_t b)
  {
    uint32_t tmp = (((a & b) << 1) + ((a ^ b) & 0xfefefefe)) & 0x01010100;
    return a + b - tmp;
  }
  static inline uint32_t med2(uint32_t a, uint32_t b, uint32_t c)
  {
    uint32_t aa_gt_bb = make_gt_mask(a, b);
    uint32_t a_xor_b_and_aa_gt_bb = ((a ^ b) & aa_gt_bb);
    uint32_t aa = a_xor_b_and_aa_gt_bb ^ a;
    uint32_t bb = a_xor_b_and_aa_gt_bb ^ b;
    uint32_t n = make_gt_mask(c, bb);
    uint32_t nn = make_gt_mask(aa, c);
    uint32_t m = ~(n | nn);
    return (n & aa) | (nn & bb) | ((bb & m) - (c & m) + (aa & m));
  }
  static inline uint32_t med(uint32_t a, uint32_t b, uint32_t c, uint32_t v)
  {
    return packed_bytes_add(med2(a, b, c), v);
  }
#define AVG_PACKED(x, y) ((((x) & (y)) + ((((x) ^ (y)) & 0xfefefefe) >> 1)) + (((x) ^ (y)) & 0x01010101))
  static inline uint32_t avg(uint32_t a, uint32_t b, uint32_t /*c*/, uint32_t v)
  {
    return packed_bytes_add(AVG_PACKED(a, b), v);
  }

#define DO_CHROMA_DECODE_PROTO(B, G, R, A, POST)                                                                                                \
  do                                                                                                                                            \
  {                                                                                                                                             \
    uint32_t i = *in;                                                                                                                           \
    uint8_t IB = i;                                                                                                                             \
    uint8_t IG = i >> 8;                                                                                                                        \
    uint8_t IR = i >> 16;                                                                                                                       \
    uint8_t IA = i >> 24;                                                                                                                       \
    uint32_t u = *prevline;                                                                                                                     \
    p = med(p, u, up, ((uint32_t)(R) << 16 & 0xff0000) + ((uint32_t)(G) << 8 & 0x00ff00) + ((uint32_t)(B) & 0x0000ff) + ((uint32_t)(A) << 24)); \
    up = u;                                                                                                                                     \
    *curline = p;                                                                                                                               \
    curline++;                                                                                                                                  \
    prevline++;                                                                                                                                 \
    POST;                                                                                                                                       \
  } while (--w);

#define DO_CHROMA_DECODE_PROTO2(B, G, R, A, POST)                                                                                               \
  do                                                                                                                                            \
  {                                                                                                                                             \
    uint32_t i = *in;                                                                                                                           \
    uint8_t IB = i;                                                                                                                             \
    uint8_t IG = i >> 8;                                                                                                                        \
    uint8_t IR = i >> 16;                                                                                                                       \
    uint8_t IA = i >> 24;                                                                                                                       \
    uint32_t u = *prevline;                                                                                                                     \
    p = avg(p, u, up, ((uint32_t)(R) << 16 & 0xff0000) + ((uint32_t)(G) << 8 & 0x00ff00) + ((uint32_t)(B) & 0x0000ff) + ((uint32_t)(A) << 24)); \
    up = u;                                                                                                                                     \
    *curline = p;                                                                                                                               \
    curline++;                                                                                                                                  \
    prevline++;                                                                                                                                 \
    POST;                                                                                                                                       \
  } while (--w);

#define DO_CHROMA_DECODE(N, R, G, B)                      \
  case ((N) << 1):                                        \
    DO_CHROMA_DECODE_PROTO(R, G, B, IA, { in += step; })  \
    break;                                                \
  case ((N) << 1) + 1:                                    \
    DO_CHROMA_DECODE_PROTO2(R, G, B, IA, { in += step; }) \
    break;

  static void DecodeLineGeneric(uint32_t *prevline, uint32_t *curline, int width, int start_block, int block_limit,
                                uint8_t *filtertypes, int skipblockbytes, uint32_t *in, uint32_t initialp, int oddskip, int dir)
  {
    uint32_t p, up;
    int step, i;
    if (start_block)
    {
      prevline += start_block * W_BLOCK_SIZE;
      curline += start_block * W_BLOCK_SIZE;
      p = curline[-1];
      up = prevline[-1];
    }
    else
    {
      p = up = initialp;
    }
    in += skipblockbytes * start_block;
    step = (dir & 1) ? 1 : -1;
    for (i = start_block; i < block_limit; i++)
    {
      int w = width - i * W_BLOCK_SIZE;
      if (w > W_BLOCK_SIZE)
        w = W_BLOCK_SIZE;
      int ww = w;
      if (step == -1)
        in += ww - 1;
      if (i & 1)
        in += oddskip * ww;
      switch (filtertypes[i])
      {
        DO_CHROMA_DECODE(0, IB, IG, IR);
        DO_CHROMA_DECODE(1, IB + IG, IG, IR + IG);
        DO_CHROMA_DECODE(2, IB, IG + IB, IR + IB + IG);
        DO_CHROMA_DECODE(3, IB + IR + IG, IG + IR, IR);
        DO_CHROMA_DECODE(4, IB + IR, IG + IB + IR, IR + IB + IR + IG);
        DO_CHROMA_DECODE(5, IB + IR, IG + IB + IR, IR);
        DO_CHROMA_DECODE(6, IB + IG, IG, IR);
        DO_CHROMA_DECODE(7, IB, IG + IB, IR);
        DO_CHROMA_DECODE(8, IB, IG, IR + IG);
        DO_CHROMA_DECODE(9, IB + IG + IR + IB, IG + IR + IB, IR + IB);
        DO_CHROMA_DECODE(10, IB + IR, IG + IR, IR);
        DO_CHROMA_DECODE(11, IB, IG + IB, IR + IB);
        DO_CHROMA_DECODE(12, IB, IG + IR + IB, IR + IB);
        DO_CHROMA_DECODE(13, IB + IG, IG + IR + IB + IG, IR + IB + IG);
        DO_CHROMA_DECODE(14, IB + IG + IR, IG + IR, IR + IB + IG + IR);
        DO_CHROMA_DECODE(15, IB, IG + (IB << 1), IR + (IB << 1));
      default:
        break;
      }
      if (step == 1)
        in += skipblockbytes - ww;
      else
        in += skipblockbytes + 1;
      if (i & 1)
        in -= oddskip * ww;
    }
  }

  static void DecodeLine(uint32_t *prevline, uint32_t *curline, int block_count, uint8_t *filtertypes, int skipblockbytes, uint32_t *in, uint32_t initialp, int oddskip, int dir)
  {
    DecodeLineGeneric(prevline, curline, INT32_MAX, 0, block_count, filtertypes, skipblockbytes, in, initialp, oddskip, dir);
  }
}

namespace tlg::v7
{

  bool decode_stream(FILE *fp, PixelBuffer &out, std::string &err)
  {
    using namespace tlg7;
    init_tables();
    // Read 4 control bytes; tolerate an extra 0x00 after header (some encoders write 12-byte mark)
    unsigned char hdr4[4];
    if (!read_exact(fp, hdr4, 4))
    {
      err = "tlg7: read header";
      return false;
    }
    int colors;
    unsigned char f1, f2, f3;
    if ((hdr4[0] == 0) && (hdr4[1] == 1 || hdr4[1] == 3 || hdr4[1] == 4))
    {
      // Likely an extra 0x00 after mark; shift
      colors = hdr4[1];
      f1 = hdr4[2];
      f2 = hdr4[3];
      if (!read_exact(fp, &f3, 1))
      {
        err = "tlg7: read flags";
        return false;
      }
    }
    else
    {
      colors = hdr4[0];
      f1 = hdr4[1];
      f2 = hdr4[2];
      f3 = hdr4[3];
    }
    if (!(colors == 1 || colors == 3 || colors == 4))
    {
      err = "tlg7: bad colors";
      return false;
    }
    if (f1 != 0 || f2 != 0 || f3 != 0)
    {
      err = "tlg7: unsupported flags";
      return false;
    }
    int width = (int)read_u32le(fp);
    int height = (int)read_u32le(fp);
    int max_bit_length = (int)read_u32le(fp);
    if (width <= 0 || height <= 0)
    {
      err = "tlg7: invalid dims";
      return false;
    }

    int x_block_count = (width - 1) / W_BLOCK_SIZE + 1;
    int y_block_count = (height - 1) / H_BLOCK_SIZE + 1;
    int main_count = width / W_BLOCK_SIZE;
    int fraction = width - main_count * W_BLOCK_SIZE;

    std::vector<uint8_t> bit_pool((size_t)max_bit_length / 8 + 5);
    std::vector<uint32_t> pixelbuf((size_t)width * H_BLOCK_SIZE + 1);
    std::vector<uint8_t> filter_types((size_t)x_block_count * y_block_count);
    std::vector<uint32_t> zeroline(width);
    std::vector<uint8_t> lzss_text(4096);

    for (int x = 0; x < width; x++)
      zeroline[x] = (colors == 3 ? 0xff000000u : 0u);
    // init LZSS_text pattern
    {
      uint32_t *p = reinterpret_cast<uint32_t *>(lzss_text.data());
      for (uint32_t i = 0; i < 32 * 0x01010101u; i += 0x01010101u)
      {
        for (uint32_t j = 0; j < 16 * 0x01010101u; j += 0x01010101u)
        {
          p[0] = i;
          p[1] = j;
          p += 2;
        }
      }
    }
    // read filter types (compressed via TLG5 LZSS)
    int inbuf_size = (int)read_u32le(fp);
    if (inbuf_size <= 0)
    {
      err = "tlg7: bad filter size";
      return false;
    }
    std::vector<uint8_t> inbuf(inbuf_size);
    if (!read_exact(fp, inbuf.data(), inbuf_size))
    {
      err = "tlg7: read filter";
      return false;
    }
    int filter_r = tlg5_lzss_decompress(filter_types.data(), inbuf.data(), inbuf_size, lzss_text.data(), 0);
    if (filter_r < 0)
    {
      std::fill(filter_types.begin(), filter_types.end(), 0);
    }

    out.width = width;
    out.height = height;
    out.channels = 4;
    out.data.resize((size_t)width * height * 4);
    uint32_t *prevline = zeroline.data();

    bool needs_bgra_swizzle = false;
    for (int y = 0; y < height; y += H_BLOCK_SIZE)
    {
      int ylim = y + H_BLOCK_SIZE;
      if (ylim >= height)
        ylim = height;
      int pixel_count = (ylim - y) * width;
      bool raw_mode = false;
      std::vector<std::vector<uint8_t>> raw_components(colors);
      for (int c = 0; c < colors; c++)
      {
        int bit_length = (int)read_u32le(fp);
        int method = (bit_length >> 30) & 3;
        bit_length &= 0x3fffffff;
        int byte_length = bit_length / 8;
        if (bit_length % 8)
          byte_length++;
        if (method == 0)
        {
          if (byte_length < 0 || byte_length > (int)bit_pool.size())
            bit_pool.resize(byte_length);
          if (!read_exact(fp, bit_pool.data(), byte_length))
          {
            err = "tlg7: read bitpool";
            return false;
          }
          if (c == 0 && colors != 1)
            TLG7DecodeGolombValuesForFirst((int8_t *)pixelbuf.data(), pixel_count, bit_pool.data());
          else
            TLG7DecodeGolombValuesNext((int8_t *)pixelbuf.data() + c, pixel_count, bit_pool.data());
          needs_bgra_swizzle = true;
        }
        else if (method == 3)
        {
          raw_mode = true;
          raw_components[c].resize(pixel_count, 0);
          if (byte_length != pixel_count)
          {
            err = "tlg7: raw block size mismatch";
            return false;
          }
          if (!read_exact(fp, raw_components[c].data(), byte_length))
          {
            err = "tlg7: read raw";
            return false;
          }
        }
        else
        {
          err = "tlg7: unsupported entropy method";
          return false;
        }
      }

      if (raw_mode)
      {
        // ensure all component arrays are populated
        for (int c = 0; c < colors; ++c)
        {
          if (raw_components[c].empty())
            raw_components[c].assign(pixel_count, (c == 3) ? 255 : 0);
        }
        for (int yy = y; yy < ylim; ++yy)
        {
          for (int x = 0; x < width; ++x)
          {
            size_t idx = (size_t)(yy - y) * width + x;
            uint8_t b = colors >= 1 ? raw_components[0][idx] : 0;
            uint8_t g = colors >= 2 ? raw_components[1][idx] : b;
            uint8_t r = colors >= 3 ? raw_components[2][idx] : g;
            uint8_t a = (colors == 4) ? raw_components[3][idx] : 255;
            uint8_t *outp = &out.data[((size_t)yy * width + x) * 4];
            outp[0] = a;
            outp[1] = r;
            outp[2] = g;
            outp[3] = b;
          }
        }
        prevline = reinterpret_cast<uint32_t *>(&out.data[((size_t)ylim - 1) * width * 4]);
        continue;
      }

      uint8_t *ft = filter_types.data() + (y / H_BLOCK_SIZE) * x_block_count;
      int skipbytes = (ylim - y) * W_BLOCK_SIZE;
      for (int yy = y; yy < ylim; yy++)
      {
        uint32_t *curline = reinterpret_cast<uint32_t *>(&out.data[((size_t)yy * width) * 4]);
        int dir = (yy & 1) ^ 1;
        int oddskip = ((ylim - yy - 1) - (yy - y));
        if (main_count)
        {
          int start = (((width < W_BLOCK_SIZE) ? width : W_BLOCK_SIZE) * (yy - y));
          DecodeLine(prevline, curline, main_count, ft, skipbytes, pixelbuf.data() + start, colors == 3 ? 0xff000000u : 0u, oddskip, dir);
        }
        if (main_count != x_block_count)
        {
          int ww = fraction;
          if (ww > W_BLOCK_SIZE)
            ww = W_BLOCK_SIZE;
          int start = ww * (yy - y);
          DecodeLineGeneric(prevline, curline, width, main_count, x_block_count, ft, skipbytes, pixelbuf.data() + start, colors == 3 ? 0xff000000u : 0u, oddskip, dir);
        }
        prevline = curline;
      }
    }
    if (needs_bgra_swizzle)
    {
      uint8_t *d = out.data.data();
      size_t total = (size_t)width * height;
      for (size_t i = 0; i < total; ++i)
      {
        uint8_t b = d[i * 4 + 0];
        uint8_t g = d[i * 4 + 1];
        uint8_t r = d[i * 4 + 2];
        uint8_t a = d[i * 4 + 3];
        d[i * 4 + 0] = a;
        d[i * 4 + 1] = r;
        d[i * 4 + 2] = g;
        d[i * 4 + 3] = b;
      }
    }
    return true;
  }
} // namespace tlg::v7

namespace tlg::v7::enc
{
  // Reuse predictor helpers from decoder namespace
  using namespace tlg7;
  constexpr int MAX_COLOR_COMPONENTS = 4;
  static inline void write_u32le_fp(FILE *fp, uint32_t v)
  {
    uint8_t b[4] = {(uint8_t)(v & 0xff), (uint8_t)((v >> 8) & 0xff), (uint8_t)((v >> 16) & 0xff), (uint8_t)((v >> 24) & 0xff)};
    fwrite(b, 1, 4, fp);
  }

  static inline void fetch_argb(const PixelBuffer &src, uint32_t px, uint32_t py,
                                uint8_t &a, uint8_t &r, uint8_t &g, uint8_t &b)
  {
    size_t idx = static_cast<size_t>(py) * src.width + px;
    if (src.channels == 4)
    {
      a = src.data[idx * 4 + 0];
      r = src.data[idx * 4 + 1];
      g = src.data[idx * 4 + 2];
      b = src.data[idx * 4 + 3];
    }
    else
    {
      a = 255;
      r = src.data[idx * 3 + 0];
      g = src.data[idx * 3 + 1];
      b = src.data[idx * 3 + 2];
    }
  }

  // Filter type compressor using LZSS (copied minimal version from core)
  class SlideCompressor
  {
    struct Chain
    {
      int Prev;
      int Next;
    };
    static const int SLIDE_N = 4096;
    static const int SLIDE_M = (18 + 255);
    uint8_t Text[SLIDE_N + SLIDE_M - 1];
    int Map[256 * 256];
    Chain Chains[SLIDE_N];
    int S;
    void AddMap(int p)
    {
      int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8);
      if (Map[place] == -1)
      {
        Map[place] = p;
      }
      else
      {
        int old = Map[place];
        Map[place] = p;
        Chains[old].Prev = p;
        Chains[p].Next = old;
        Chains[p].Prev = -1;
      }
    }
    void DeleteMap(int p)
    {
      int n;
      if ((n = Chains[p].Next) != -1)
        Chains[n].Prev = Chains[p].Prev;
      if ((n = Chains[p].Prev) != -1)
      {
        Chains[n].Next = Chains[p].Next;
      }
      else if (Chains[p].Next != -1)
      {
        int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8);
        Map[place] = Chains[p].Next;
      }
      else
      {
        int place = Text[p] + ((int)Text[(p + 1) & (SLIDE_N - 1)] << 8);
        Map[place] = -1;
      }
      Chains[p].Prev = -1;
      Chains[p].Next = -1;
    }
    int GetMatch(const uint8_t *cur, int curlen, int &pos, int s)
    {
      if (curlen < 3)
        return 0;
      int place = cur[0] + ((int)cur[1] << 8);
      int maxlen = 0;
      if ((place = Map[place]) != -1)
      {
        int place_org;
        curlen -= 1;
        do
        {
          place_org = place;
          if (s == place || s == ((place + 1) & (SLIDE_N - 1)))
            continue;
          place += 2;
          int lim = (SLIDE_M < curlen ? SLIDE_M : curlen) + place_org;
          const uint8_t *c = cur + 2;
          if (lim >= SLIDE_N)
          {
            if (place_org <= s && s < SLIDE_N)
              lim = s;
            else if (s < (lim & (SLIDE_N - 1)))
              lim = s + SLIDE_N;
          }
          else
          {
            if (place_org <= s && s < lim)
              lim = s;
          }
          while (Text[place] == *(c++) && place < lim)
            place++;
          int matchlen = place - place_org;
          if (matchlen > maxlen)
            pos = place_org, maxlen = matchlen;
          if (matchlen == SLIDE_M)
            return maxlen;
        } while ((place = Chains[place_org].Next) != -1);
      }
      return maxlen;
    }

  public:
    SlideCompressor()
    {
      S = 0;
      for (int i = 0; i < SLIDE_N + SLIDE_M - 1; ++i)
        Text[i] = 0;
      for (int i = 0; i < 256 * 256; ++i)
        Map[i] = -1;
      for (int i = 0; i < SLIDE_N; ++i)
      {
        Chains[i].Prev = -1;
        Chains[i].Next = -1;
      }
      for (int i = SLIDE_N - 1; i >= 0; --i)
        AddMap(i);
    }
    void Encode(const uint8_t *in, size_t inlen, uint8_t *out, size_t &outlen)
    {
      uint8_t code[40], codeptr = 1, mask = 1;
      code[0] = 0;
      outlen = 0;
      int s = S;
      while (inlen > 0)
      {
        int pos = 0;
        int len = GetMatch(in, (int)inlen, pos, s);
        if (len >= 3)
        {
          code[0] |= mask;
          if (len >= 18)
          {
            code[codeptr++] = pos & 0xff;
            code[codeptr++] = ((pos & 0xf00) >> 8) | 0xf0;
            code[codeptr++] = len - 18;
          }
          else
          {
            code[codeptr++] = pos & 0xff;
            code[codeptr++] = ((pos & 0xf00) >> 8) | ((len - 3) << 4);
          }
          while (len--)
          {
            uint8_t c = *in++;
            DeleteMap((s - 1) & (SLIDE_N - 1));
            DeleteMap(s);
            if (s < SLIDE_M - 1)
              Text[s + SLIDE_N] = c;
            Text[s] = c;
            AddMap((s - 1) & (SLIDE_N - 1));
            AddMap(s);
            s++;
            inlen--;
            s &= (SLIDE_N - 1);
          }
        }
        else
        {
          uint8_t c = *in++;
          DeleteMap((s - 1) & (SLIDE_N - 1));
          DeleteMap(s);
          if (s < SLIDE_M - 1)
            Text[s + SLIDE_N] = c;
          Text[s] = c;
          AddMap((s - 1) & (SLIDE_N - 1));
          AddMap(s);
          s++;
          inlen--;
          s &= (SLIDE_N - 1);
          code[codeptr++] = c;
        }
        mask <<= 1;
        if (mask == 0)
        {
          for (int i = 0; i < codeptr; i++)
            out[outlen++] = code[i];
          mask = codeptr = 1;
          code[0] = 0;
        }
      }
      if (mask != 1)
      {
        for (int i = 0; i < codeptr; i++)
          out[outlen++] = code[i];
      }
      S = s;
    }
  };

  static void InitializeColorFilterCompressor(SlideCompressor &c)
  {
    uint8_t code[4096];
    uint8_t dum[4096];
    uint8_t *p = code;
    for (int i = 0; i < 32; i++)
    {
      for (int j = 0; j < 16; j++)
      {
        p[0] = p[1] = p[2] = p[3] = (uint8_t)i;
        p += 4;
        p[0] = p[1] = p[2] = p[3] = (uint8_t)j;
        p += 4;
      }
    }
    size_t dumlen = 0;
    c.Encode(code, 4096, dum, dumlen);
  }

  constexpr int FILTER_TRY_COUNT = 32;
  constexpr int GOLOMB_GIVE_UP_BYTES = 4;

  class TLG7BitStream
  {
    std::vector<uint8_t> &out_;
    std::vector<uint8_t> buffer_;
    size_t byte_pos_ = 0;
    int bit_pos_ = 0;

    void ensure_capacity()
    {
      if (buffer_.size() <= byte_pos_)
        buffer_.resize(byte_pos_ + 1, 0);
    }

  public:
    explicit TLG7BitStream(std::vector<uint8_t> &out) : out_(out) {}
    ~TLG7BitStream() { Flush(); }

    int GetBitPos() const { return bit_pos_; }
    size_t GetBytePos() const { return byte_pos_; }
    size_t GetBitLength() const { return byte_pos_ * 8 + bit_pos_; }

    void Put1Bit(bool bit)
    {
      ensure_capacity();
      if (bit)
        buffer_[byte_pos_] |= static_cast<uint8_t>(1u << bit_pos_);
      bit_pos_++;
      if (bit_pos_ == 8)
      {
        bit_pos_ = 0;
        byte_pos_++;
      }
    }

    void PutValue(long value, int len)
    {
      for (int i = 0; i < len; ++i)
        Put1Bit((value >> i) & 1);
    }

    void PutGamma(int v)
    {
      if (v <= 0)
        return;
      int t = v;
      t >>= 1;
      int cnt = 0;
      while (t)
      {
        Put1Bit(0);
        t >>= 1;
        cnt++;
      }
      Put1Bit(1);
      while (cnt--)
      {
        Put1Bit(v & 1);
        v >>= 1;
      }
    }

    void Flush()
    {
      size_t bytes = byte_pos_ + (bit_pos_ ? 1 : 0);
      if (bytes)
      {
        ensure_capacity();
        out_.insert(out_.end(), buffer_.begin(), buffer_.begin() + bytes);
      }
      buffer_.clear();
      byte_pos_ = 0;
      bit_pos_ = 0;
    }
  };

  static inline std::vector<int8_t> to_signed(const std::vector<uint8_t> &src)
  {
    std::vector<int8_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dst[i] = static_cast<int8_t>(src[i]);
    return dst;
  }

  static void compress_values_golomb(TLG7BitStream &bs, const std::vector<int8_t> &buf)
  {
    if (buf.empty())
      return;
    bs.PutValue(buf[0] ? 1 : 0, 1);

    int n = GOLOMB_N_COUNT - 1;
    int a = 0;
    int count = 0;
    const size_t size = buf.size();

    for (size_t i = 0; i < size; ++i)
    {
      int e = buf[i];
      if (e != 0)
      {
        if (count)
        {
          bs.PutGamma(count);
          count = 0;
        }

        size_t ii = i;
        while (ii < size && buf[ii] != 0)
          ++ii;
        size_t nonzero = ii - i;
        bs.PutGamma(static_cast<int>(nonzero));

        for (; i < ii; ++i)
        {
          e = buf[i];
          int m = ((e >= 0) ? 2 * e : -2 * e - 1) - 1;
          if (m < 0)
            m = 0;
          int k = TLG7GolombBitLengthTable[a][n];
          size_t store_limit = bs.GetBytePos() + GOLOMB_GIVE_UP_BYTES;
          bool put_one = true;
          int q = (k > 0) ? (m >> k) : m;
          for (; q > 0; --q)
          {
            if (bs.GetBytePos() >= store_limit)
            {
              bs.PutValue(m >> k, 8);
              put_one = false;
              break;
            }
            bs.Put1Bit(0);
          }
          if (put_one && bs.GetBytePos() >= store_limit)
          {
            bs.PutValue(m >> k, 8);
            put_one = false;
          }
          if (put_one)
            bs.Put1Bit(1);
          if (k)
            bs.PutValue(m & ((1 << k) - 1), k);
          a += (m >> 1);
          if (--n < 0)
          {
            a >>= 1;
            n = GOLOMB_N_COUNT - 1;
          }
        }
        i = ii - 1;
      }
      else
      {
        count++;
      }
    }

    if (count)
      bs.PutGamma(count);
  }

  static size_t estimate_golomb_bits(const std::vector<uint8_t> &data)
  {
    std::vector<uint8_t> dummy;
    TLG7BitStream bs(dummy);
    compress_values_golomb(bs, to_signed(data));
    return bs.GetBitLength();
  }

  static void apply_color_filter(int code,
                                 std::vector<uint8_t> &b,
                                 std::vector<uint8_t> &g,
                                 std::vector<uint8_t> &r)
  {
    const size_t n = b.size();
    switch (code)
    {
    case 0:
      return;
    case 1:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 2:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 3:
      for (size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 4:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 5:
      for (size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 6:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      break;
    case 7:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      break;
    case 8:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      break;
    case 9:
      for (size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 10:
      for (size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      }
      break;
    case 11:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 12:
      for (size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      }
      break;
    case 13:
      for (size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 14:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      }
      break;
    case 15:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(b[i] << 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 16:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - r[i]);
      break;
    case 17:
      for (size_t i = 0; i < n; ++i)
      {
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - g[i]);
      }
      break;
    case 18:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - b[i]);
      break;
    case 19:
      for (size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 20:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
      break;
    case 21:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (g[i] >> 1));
      break;
    case 22:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (b[i] >> 1));
      break;
    case 23:
      for (size_t i = 0; i < n; ++i)
      {
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
      }
      break;
    case 24:
      for (size_t i = 0; i < n; ++i)
      {
        b[i] = static_cast<uint8_t>(b[i] - r[i]);
        r[i] = static_cast<uint8_t>(r[i] - g[i]);
        g[i] = static_cast<uint8_t>(g[i] - b[i]);
      }
      break;
    case 25:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (r[i] >> 1));
      break;
    case 26:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (g[i] >> 1));
      break;
    case 27:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(r[i] >> 1);
        g[i] = static_cast<uint8_t>(g[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 28:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (b[i] >> 1));
      break;
    case 29:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(g[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 30:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(b[i] >> 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        g[i] = static_cast<uint8_t>(g[i] - t);
      }
      break;
    case 31:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (r[i] >> 1));
      break;
    case 32:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (b[i] << 1));
      break;
    case 33:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (g[i] << 1));
      break;
    case 34:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(r[i] << 1);
        g[i] = static_cast<uint8_t>(g[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 35:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (b[i] << 1));
      break;
    case 36:
      for (size_t i = 0; i < n; ++i)
        r[i] = static_cast<uint8_t>(r[i] - (g[i] << 1));
      break;
    case 37:
      for (size_t i = 0; i < n; ++i)
      {
        uint8_t t = static_cast<uint8_t>(g[i] << 1);
        r[i] = static_cast<uint8_t>(r[i] - t);
        b[i] = static_cast<uint8_t>(b[i] - t);
      }
      break;
    case 38:
      for (size_t i = 0; i < n; ++i)
        g[i] = static_cast<uint8_t>(g[i] - (r[i] << 1));
      break;
    case 39:
      for (size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint8_t>(b[i] - (r[i] << 1));
      break;
    default:
      break;
    }
  }

  static int detect_color_filter(const std::vector<uint8_t> &b,
                                 const std::vector<uint8_t> &g,
                                 const std::vector<uint8_t> &r,
                                 std::vector<uint8_t> &out_b,
                                 std::vector<uint8_t> &out_g,
                                 std::vector<uint8_t> &out_r)
  {
    size_t best_bits = std::numeric_limits<size_t>::max();
    int best_code = 0;
    std::vector<uint8_t> tb, tg, tr;
    tb.reserve(b.size());
    tg.reserve(g.size());
    tr.reserve(r.size());
    for (int code = 0; code < FILTER_TRY_COUNT; ++code)
    {
      tb.assign(b.begin(), b.end());
      tg.assign(g.begin(), g.end());
      tr.assign(r.begin(), r.end());
      apply_color_filter(code, tb, tg, tr);
      size_t bits = estimate_golomb_bits(tb) + estimate_golomb_bits(tg) + estimate_golomb_bits(tr);
      if (bits < best_bits)
      {
        best_bits = bits;
        best_code = code;
        out_b = std::move(tb);
        out_g = std::move(tg);
        out_r = std::move(tr);
      }
    }
    return best_code;
  }

  struct Candidate
  {
    int predictor = 0;
    int filter = 0;
    size_t total_bits = std::numeric_limits<size_t>::max();
    std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> comps;
  };

  struct EncodingContext
  {
    const PixelBuffer &src;
    int colors;
    int x_block_count;
    int y_block_count;
    size_t block_capacity;
    size_t block_group_capacity;
    std::vector<uint32_t> argb;
    std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> buf;
    std::array<std::vector<uint8_t>, MAX_COLOR_COMPONENTS> block_buf;
    std::vector<uint32_t> zeroline;
    std::vector<uint8_t> filtertypes;
    std::vector<uint8_t> temp_storage;
    int max_bit_length = 0;
    size_t filter_index = 0;

    EncodingContext(const PixelBuffer &src_in, int color_count, int xb, int yb)
        : src(src_in),
          colors(color_count),
          x_block_count(xb),
          y_block_count(yb),
          block_capacity((size_t)W_BLOCK_SIZE * H_BLOCK_SIZE),
          block_group_capacity((size_t)H_BLOCK_SIZE * src_in.width),
          argb((size_t)src_in.width * src_in.height),
          zeroline(src_in.width, color_count == 3 ? 0xff000000u : 0u),
          filtertypes((size_t)xb * yb, 0)
    {
      for (int c = 0; c < colors; ++c)
      {
        buf[c].resize(block_capacity * 3);
        block_buf[c].resize(block_group_capacity);
      }
      temp_storage.reserve((size_t)src_in.width * src_in.height);
    }
  };

  static void write_stream_header(FILE *fp, const PixelBuffer &src, int colors)
  {
    unsigned char mark[11] = {'T', 'L', 'G', '7', '.', '0', 0, 'r', 'a', 'w', 0x1a};
    fwrite(mark, 1, sizeof(mark), fp);
    uint8_t cbyte = static_cast<uint8_t>(colors);
    fwrite(&cbyte, 1, 1, fp);
    uint8_t zeros[3] = {0, 0, 0};
    fwrite(zeros, 1, 3, fp);
    write_u32le_fp(fp, src.width);
    write_u32le_fp(fp, src.height);
  }

  static void populate_argb_buffer(const PixelBuffer &src, std::vector<uint32_t> &argb)
  {
    for (uint32_t y = 0; y < src.height; ++y)
    {
      for (uint32_t x = 0; x < src.width; ++x)
      {
        uint8_t a, r, g, b;
        fetch_argb(src, x, y, a, r, g, b);
        argb[(size_t)y * src.width + x] = ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
      }
    }
  }

  static Candidate build_candidate(EncodingContext &ctx,
                                   int predictor,
                                   std::array<tlg7::SelectorState, MAX_COLOR_COMPONENTS> &state,
                                   uint32_t x,
                                   uint32_t xlim,
                                   uint32_t y,
                                   uint32_t ylim,
                                   uint32_t xp,
                                   size_t block_pixels)
  {
    Candidate cand;
    cand.predictor = predictor;
    const uint32_t width = ctx.src.width;
    const uint32_t bw = xlim - x;
    const uint32_t block_h = ylim - y;

    for (int c = 0; c < ctx.colors; ++c)
    {
      size_t wp = 0;
      for (uint32_t yy = y; yy < ylim; ++yy)
      {
        const uint32_t *sl = &ctx.argb[yy * width];
        const uint32_t *upper_line = (yy < 1) ? ctx.zeroline.data() : &ctx.argb[(yy - 1) * width];
        const uint32_t *upper_upper_line = (yy < 2) ? ctx.zeroline.data() : &ctx.argb[(yy - 2) * width];
        for (uint32_t xx = x; xx < xlim; ++xx)
        {
          uint8_t px = static_cast<uint8_t>((sl[xx] >> (c * 8)) & 0xff);
          uint8_t pa = (xx > 0) ? static_cast<uint8_t>((sl[xx - 1] >> (c * 8)) & 0xff) : 0;
          uint8_t pb = static_cast<uint8_t>((upper_line[xx] >> (c * 8)) & 0xff);
          uint8_t pc = (xx > 0) ? static_cast<uint8_t>((upper_line[xx - 1] >> (c * 8)) & 0xff) : 0;
          uint8_t pd = (xx < width - 1) ? static_cast<uint8_t>((upper_line[xx + 1] >> (c * 8)) & 0xff) : 0;
          uint8_t pf = static_cast<uint8_t>((upper_upper_line[xx] >> (c * 8)) & 0xff);
          uint8_t py;
          if (predictor == 0)
          {
            auto [pred, pid] = tlg7::cas8_predict<uint8_t>(
                pa, pb, pc, pd, pf, 0, 255, {true /*enablePlanarLite*/}, state[c]);

            py = pred;
            state[c].update(pid, std::abs((int)px - (int)py));
          }
          else
          {
            py = static_cast<uint8_t>((pa + pb + 1) >> 1);
          }
          int pix = (static_cast<int>(px) - static_cast<int>(py)) & 0xff;
          ctx.buf[c][wp] = static_cast<uint8_t>(pix);
          ++wp;
        }
      }
    }

    size_t dbofs = (size_t)(predictor + 1) * ctx.block_capacity;
    size_t write_pos = 0;
    for (uint32_t yy = y; yy < ylim; ++yy)
    {
      size_t ofs;
      if (!(xp & 1))
        ofs = (size_t)(yy - y) * bw;
      else
        ofs = (size_t)(ylim - yy - 1) * bw;
      bool dir;
      if (!(block_h & 1))
        dir = ((yy & 1) ^ (xp & 1)) != 0;
      else
      {
        if (xp & 1)
          dir = (yy & 1) != 0;
        else
          dir = ((yy & 1) ^ (xp & 1)) != 0;
      }
      if (!dir)
      {
        for (uint32_t xx = 0; xx < bw; ++xx)
        {
          for (int c = 0; c < ctx.colors; ++c)
            ctx.buf[c][dbofs + write_pos] = ctx.buf[c][ofs + xx];
          ++write_pos;
        }
      }
      else
      {
        for (int xx = (int)bw - 1; xx >= 0; --xx)
        {
          for (int c = 0; c < ctx.colors; ++c)
            ctx.buf[c][dbofs + write_pos] = ctx.buf[c][ofs + (size_t)xx];
          ++write_pos;
        }
      }
    }

    for (int c = 0; c < ctx.colors; ++c)
    {
      const uint8_t *begin = ctx.buf[c].data() + dbofs;
      cand.comps[c].assign(begin, begin + block_pixels);
    }

    if (ctx.colors >= 3)
    {
      std::vector<uint8_t> filtered_b, filtered_g, filtered_r;
      int ft = detect_color_filter(cand.comps[0], cand.comps[1], cand.comps[2], filtered_b, filtered_g, filtered_r);
      cand.filter = ft;
      cand.comps[0] = std::move(filtered_b);
      cand.comps[1] = std::move(filtered_g);
      cand.comps[2] = std::move(filtered_r);
    }

    size_t bits = 0;
    for (int c = 0; c < ctx.colors; ++c)
      bits += estimate_golomb_bits(cand.comps[c]);
    cand.total_bits = bits;

    return cand;
  }

  static void append_encoded_components(EncodingContext &ctx, uint32_t pixel_count)
  {
    for (int c = 0; c < ctx.colors; ++c)
    {
      std::vector<int8_t> signed_data(pixel_count);
      for (size_t i = 0; i < pixel_count; ++i)
        signed_data[i] = static_cast<int8_t>(ctx.block_buf[c][i]);
      std::vector<uint8_t> bit_bytes;
      TLG7BitStream bs(bit_bytes);
      compress_values_golomb(bs, signed_data);
      uint32_t bitlen = static_cast<uint32_t>(bs.GetBitLength());
      if ((int)bitlen > ctx.max_bit_length)
        ctx.max_bit_length = (int)bitlen;
      uint8_t header[4] = {
          static_cast<uint8_t>(bitlen & 0xff),
          static_cast<uint8_t>((bitlen >> 8) & 0xff),
          static_cast<uint8_t>((bitlen >> 16) & 0xff),
          static_cast<uint8_t>((bitlen >> 24) & 0xff)};
      bs.Flush();
      ctx.temp_storage.insert(ctx.temp_storage.end(), header, header + 4);
      ctx.temp_storage.insert(ctx.temp_storage.end(), bit_bytes.begin(), bit_bytes.end());
    }
  }

  static bool encode_block_row(EncodingContext &ctx, uint32_t y, std::string &err)
  {
    const uint32_t width = ctx.src.width;
    uint32_t ylim = std::min<uint32_t>(y + H_BLOCK_SIZE, ctx.src.height);
    uint32_t block_h = ylim - y;
    uint32_t pixel_count = block_h * width;
    std::array<tlg7::SelectorState, MAX_COLOR_COMPONENTS> states;

    for (int c = 0; c < ctx.colors; ++c)
      std::fill(ctx.block_buf[c].begin(), ctx.block_buf[c].begin() + pixel_count, 0);

    size_t gwp = 0;
    for (uint32_t x = 0, xp = 0; x < width; x += W_BLOCK_SIZE, ++xp)
    {
      uint32_t xlim = std::min<uint32_t>(x + W_BLOCK_SIZE, width);
      uint32_t bw = xlim - x;
      size_t block_pixels = (size_t)bw * block_h;

      std::array<Candidate, 2> candidates = {
          build_candidate(ctx, 0, states, x, xlim, y, ylim, xp, block_pixels),
          build_candidate(ctx, 1, states, x, xlim, y, ylim, xp, block_pixels)};

      const Candidate *best = &candidates[0];
      //      if (candidates[1].total_bits < best->total_bits)
      //        best = &candidates[1];

      ctx.filtertypes[ctx.filter_index++] = static_cast<uint8_t>((best->filter << 1) | best->predictor);

      for (int c = 0; c < ctx.colors; ++c)
      {
        uint8_t *dst = ctx.block_buf[c].data() + gwp;
        std::copy(best->comps[c].begin(), best->comps[c].end(), dst);
      }

      gwp += block_pixels;
    }

    if (gwp != pixel_count)
    {
      err = "tlg7: block size mismatch";
      return false;
    }

    append_encoded_components(ctx, pixel_count);

    const uint32_t *last_line = &ctx.argb[(ylim - 1) * width];
    std::copy(last_line, last_line + width, ctx.zeroline.begin());

    return true;
  }

  static void write_encoded_payload(FILE *fp, const EncodingContext &ctx)
  {
    write_u32le_fp(fp, (uint32_t)ctx.max_bit_length);

    SlideCompressor filter_comp;
    InitializeColorFilterCompressor(filter_comp);
    std::vector<uint8_t> filter_out(ctx.filtertypes.empty() ? 0 : ctx.filtertypes.size() * 2 + 16);
    size_t outlen = 0;
    if (!ctx.filtertypes.empty())
      filter_comp.Encode(ctx.filtertypes.data(), ctx.filtertypes.size(), filter_out.data(), outlen);
    write_u32le_fp(fp, (uint32_t)outlen);
    if (outlen)
      fwrite(filter_out.data(), 1, outlen, fp);

    if (!ctx.temp_storage.empty())
      fwrite(ctx.temp_storage.data(), 1, ctx.temp_storage.size(), fp);
  }

  static bool write_tlg7_raw(FILE *fp, const PixelBuffer &src, int colors, std::string &err)
  {
    err.clear();
    if (colors < 1 || colors > 4)
    {
      err = "unsupported color count";
      return false;
    }

    init_tables();

    const int x_block_count = (int)((src.width - 1) / W_BLOCK_SIZE) + 1;
    const int y_block_count = (int)((src.height - 1) / H_BLOCK_SIZE) + 1;

    write_stream_header(fp, src, colors);

    EncodingContext ctx(src, colors, x_block_count, y_block_count);
    populate_argb_buffer(src, ctx.argb);

    for (uint32_t y = 0; y < src.height; y += H_BLOCK_SIZE)
    {
      if (!encode_block_row(ctx, y, err))
        return false;
    }

    if (ctx.filter_index != ctx.filtertypes.size())
    {
      err = "tlg7: filter count mismatch";
      return false;
    }

    write_encoded_payload(fp, ctx);
    return true;
  }
}
bool tlg::v7::enc::write_raw(FILE *fp, const PixelBuffer &src, int colors, std::string &err)
{
  return write_tlg7_raw(fp, src, colors, err);
}
