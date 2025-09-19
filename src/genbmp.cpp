#include <cstdio>
#include <cstdint>
#include <vector>

#pragma pack(push, 1)
struct FH
{
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t r1;
  uint16_t r2;
  uint32_t bfOffBits;
};
struct IH
{
  uint32_t biSize;
  int32_t w;
  int32_t h;
  uint16_t planes;
  uint16_t bpp;
  uint32_t comp;
  uint32_t img;
  int32_t xppm;
  int32_t yppm;
  uint32_t clrUsed;
  uint32_t clrImp;
};
#pragma pack(pop)
int main()
{
  const int W = 8, H = 8;
  const int BPP = 24;
  const int stride = ((W * 3 + 3) & ~3);
  std::vector<uint8_t> row(stride);
  FH fh{};
  IH ih{};
  fh.bfType = 0x4D42;
  fh.bfOffBits = sizeof(FH) + sizeof(IH);
  fh.bfSize = fh.bfOffBits + stride * H;
  ih.biSize = sizeof(IH);
  ih.w = W;
  ih.h = H;
  ih.planes = 1;
  ih.bpp = BPP;
  ih.comp = 0;
  ih.img = stride * H;
  FILE *fp = fopen("test_8x8.bmp", "wb");
  if (!fp)
    return 1;
  fwrite(&fh, sizeof(fh), 1, fp);
  fwrite(&ih, sizeof(ih), 1, fp);
  for (int y = H - 1; y >= 0; --y)
  {
    for (int x = 0; x < W; ++x)
    {
      // simple pattern with soft gradient (no alpha)
      uint8_t r = (uint8_t)(x * 32);
      uint8_t g = (uint8_t)(y * 32);
      uint8_t b = (uint8_t)((x + y) * 16);
      row[x * 3 + 0] = b;
      row[x * 3 + 1] = g;
      row[x * 3 + 2] = r;
    }
    for (int p = W * 3; p < stride; ++p)
      row[p] = 0;
    fwrite(row.data(), 1, stride, fp);
  }
  fclose(fp);
  return 0;
}
