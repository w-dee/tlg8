#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace tlg::detail
{

    bool read_exact(FILE *fp, void *buf, size_t n);
    uint32_t read_u32le(FILE *fp);
    void write_u32le(FILE *fp, uint32_t v);
    int tlg5_lzss_decompress(uint8_t *out, const uint8_t *in, int insize, uint8_t *text, int r);

    // 値が0のときは0ビットと見なして、必要なビット数を返す
    template <typename T>
    inline constexpr unsigned bit_width(T value) noexcept
    {
        static_assert(std::is_integral_v<T>, "整数型のみサポートします");
        using U = std::make_unsigned_t<T>;

        U v = static_cast<U>(value);
        unsigned bits = 0;

        while (v != 0)
        {
            ++bits;
            v >>= 1;
        }
        return bits;
    }

} // namespace tlg::detail
