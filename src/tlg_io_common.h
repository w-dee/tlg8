#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace tlg::detail
{

    bool read_exact(FILE *fp, void *buf, size_t n);
    uint32_t read_u32le(FILE *fp);
    void write_u32le(FILE *fp, uint32_t v);
    int tlg5_lzss_decompress(uint8_t *out, const uint8_t *in, int insize, uint8_t *text, int r);

} // namespace tlg::detail
