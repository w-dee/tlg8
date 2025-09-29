// bitio.hpp - C++17, speed-first bit I/O with 128-bit accumulator (fallback to 64-bit).
// Notes:
// - Internal bit order: LSB-first in the accumulator (we pop from LSB).
// - Safe unaligned loads/stores via memcpy (avoids UB).
// - Designed so the raw buffers are contiguous -> easy to add SIMD kernels later.
// - Build flags: define BITIO_NO_UINT128 to force 64-bit accumulator.

#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <cstddef> // ptrdiff_tのために追加

#ifndef BITIO_LIKELY
#if defined(__GNUC__) || defined(__clang__)
#define BITIO_LIKELY(x) __builtin_expect(!!(x), 1)
#define BITIO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BITIO_LIKELY(x) (x)
#define BITIO_UNLIKELY(x) (x)
#endif
#endif

#if !defined(BITIO_NO_UINT128) && (defined(__GNUC__) || defined(__clang__))
#define BITIO_HAVE_UINT128 1
#else
#define BITIO_HAVE_UINT128 0
#endif

namespace tlg::v8::detail::bitio
{

#if BITIO_HAVE_UINT128
    using acc_t = unsigned __int128; // 128-bit accumulator
    static inline constexpr unsigned ACC_BITS = 128;
#else
    using acc_t = unsigned long long; // 64-bit accumulator fallback
    static inline constexpr unsigned ACC_BITS = 64;
#endif

    // -------- Endian helpers (branchless, portable) --------
    static inline uint16_t bswap16(uint16_t x) { return (uint16_t)((x << 8) | (x >> 8)); }
    static inline uint32_t bswap32(uint32_t x)
    {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_bswap32(x);
#else
        return (x << 24) | ((x << 8) & 0x00FF0000u) | ((x >> 8) & 0x0000FF00u) | (x >> 24);
#endif
    }

    // -------- BitReader --------
    class BitReader
    {
    public:
        BitReader(const uint8_t *data, size_t size)
            : p_(data), end_(data + size), acc_(0), bits_(0) {}

        // Ensure at least k bits are available (up to ACC_BITS).
        inline void fill(unsigned k = 32)
        {
            if (bits_ >= k)
                return;
            // Load 8 bytes at a time (amortized good), append at current 'bits_'.
            while (bits_ + 64 <= ACC_BITS && p_ + 8 <= end_ && bits_ < k)
            {
                uint64_t chunk;
                std::memcpy(&chunk, p_, 8);
                p_ += 8;
                acc_ |= (acc_t)chunk << bits_;
                bits_ += 64;
            }
            // Tail: load remaining bytes (0..7)
            if (BITIO_UNLIKELY(bits_ < k) && p_ < end_)
            {
                uint64_t chunk = 0;
                size_t remain = (size_t)(end_ - p_);
                size_t take = remain > 7 ? 7 : remain; // keep <=7 here; 8B handled above
                std::memcpy(&chunk, p_, take);
                p_ += take;
                acc_ |= (acc_t)chunk << bits_;
                bits_ += (unsigned)(take * 8);
            }
        }

        // Peek n bits (n <= 32 is typical; supports up to 32 fast-path).
        inline uint32_t peek(unsigned n)
        {
            // For 1..8 bits common case we still use same logic; compiler will fold masks.
            if (BITIO_UNLIKELY(n == 0))
                return 0;
            if (bits_ < n)
                fill(n);
            if (BITIO_UNLIKELY(bits_ < n))
                return 0; // underflow policy: return 0
            const uint64_t mask = (n == 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
            return (uint32_t)((uint64_t)(acc_ & (acc_t)mask));
        }

        // Get n bits and consume them. (n <= 32 recommended)
        inline uint32_t get(unsigned n)
        {
            uint32_t v = peek(n);
            consume(n);
            return v;
        }

        // Optimized path for 1..8 bits (very common). Branch-light.
        inline uint32_t get_upto8(unsigned n)
        {
            // precondition: 1 <= n <= 8
            if (bits_ < n)
                fill(8);
            const uint32_t v = (uint32_t)((uint64_t)acc_ & ((1u << n) - 1u));
            acc_ >>= n;
            bits_ -= (bits_ >= n ? n : bits_);
            return v;
        }

        inline void consume(unsigned n)
        {
            if (BITIO_UNLIKELY(n == 0))
                return;
            if (bits_ < n)
                fill(n);
            const unsigned c = (bits_ >= n) ? n : bits_;
            acc_ >>= c;
            bits_ -= c;
            const unsigned rem = n - c;
            if (rem)
            {
                fill(rem);
                acc_ >>= rem;
                bits_ -= rem;
            }
        }

        inline void align_to_byte()
        {
            unsigned m = bits_ & 7u;
            if (m)
            {
                acc_ >>= m;
                bits_ -= m;
            }
        }

        inline bool byte_aligned() const { return (bits_ & 7u) == 0; }
        inline bool eof() const { return (bits_ == 0) && (p_ >= end_); }
        inline size_t remaining_bytes_approx() const
        {
            // approx: unread bytes in source + bytes represented in acc_
            return (size_t)(end_ - p_) + (bits_ >> 3);
        }

        // ---- Byte & Integer reads (little/big endian). These bypass bit reservoir by aligning first. ----
        inline bool read_u8(uint8_t &out)
        {
            if (!byte_aligned())
                align_to_byte();
            if (p_ >= end_)
                return false;
            out = *p_++;
            return true;
        }

        inline bool read_u16_le(uint16_t &out)
        {
            if (!byte_aligned())
                align_to_byte();
            if (end_ - p_ < 2)
                return false;
            uint16_t v;
            std::memcpy(&v, p_, 2);
            p_ += 2;
            out = v;
            return true;
        }

        inline bool read_u16_be(uint16_t &out)
        {
            uint16_t v;
            if (!read_u16_le(v))
                return false;
            out = bswap16(v);
            return true;
        }

        inline bool read_u32_le(uint32_t &out)
        {
            if (!byte_aligned())
                align_to_byte();
            if (end_ - p_ < 4)
                return false;
            uint32_t v;
            std::memcpy(&v, p_, 4);
            p_ += 4;
            out = v;
            return true;
        }

        inline bool read_u32_be(uint32_t &out)
        {
            uint32_t v;
            if (!read_u32_le(v))
                return false;
            out = bswap32(v);
            return true;
        }

        // Access to raw pointer (for future SIMD scans, etc.)
        inline const uint8_t *raw_ptr() const { return p_; }
        inline const uint8_t *raw_end() const { return end_; }

    private:
        const uint8_t *p_;
        const uint8_t *end_;
        acc_t acc_;
        unsigned bits_;
    };

    // -------- BitWriter --------
    class BitWriter
    {
    public:
        BitWriter(uint8_t *dst, size_t capacity)
            : base_(dst), p_(dst), end_(dst + capacity), acc_(0), bits_(0) {}

        // Put n bits (lowest n bits of v), 1..32 typical
        inline void put(uint32_t v, unsigned n)
        {
            // Append into accumulator at current 'bits_'.
            acc_ |= (acc_t)(v & ((n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u))) << bits_;
            bits_ += n;
            flush_if_full();
        }

        // Optimized 1..8 bits
        inline void put_upto8(uint32_t v, unsigned n)
        {
            // precondition: 1 <= n <= 8
            acc_ |= (acc_t)(v & ((1u << n) - 1u)) << bits_;
            bits_ += n;
            flush_if_full();
        }

        inline void align_to_byte_zero()
        {
            unsigned m = bits_ & 7u;
            if (m)
            {
                // zero-pad (already zero since we ORed exact bits)
                bits_ += (8 - m);
                flush_if_full();
            }
            flush_bytes(); // write any full bytes out
        }

        inline bool write_u8(uint8_t v)
        {
            align_to_byte_zero();
            return store_bytes(&v, 1);
        }

        inline bool write_u16_le(uint16_t v)
        {
            align_to_byte_zero();
            return store_bytes(&v, 2);
        }

        inline bool write_u16_be(uint16_t v)
        {
            uint16_t be = bswap16(v);
            return write_u16_le(be);
        }

        inline bool write_u32_le(uint32_t v)
        {
            align_to_byte_zero();
            return store_bytes(&v, 4);
        }

        inline bool write_u32_be(uint32_t v)
        {
            uint32_t be = bswap32(v);
            return write_u32_le(be);
        }

        inline bool align_to_u32_zero()
        {
            align_to_byte_zero();
            const size_t padding = (4 - (bytes_written() & 3u)) & 3u;
            if (padding == 0)
                return true;
            static const uint8_t kZeroPad[4] = {0, 0, 0, 0};
            return store_bytes(kZeroPad, padding);
        }

        // Finish writing: flush remaining full bytes; if leftover bits exist, pad with zeros to next byte.
        inline bool finish()
        {
            if (bits_ & 7u)
            {
                unsigned pad = 8 - (bits_ & 7u);
                bits_ += pad; // zero-pad
            }
            return flush_all();
        }

        inline size_t bytes_written() const { return (size_t)(p_ - base_); }
        inline uint8_t *raw_ptr() const { return p_; }
        inline uint8_t *raw_base() const { return const_cast<uint8_t *>(base_); }
        inline uint8_t *raw_end() const { return const_cast<uint8_t *>(end_); }

    private:
        // Write out as many full bytes as possible from acc_.
        inline void flush_bytes()
        {
            // Write in 8-byte chunks when possible.
            while (bits_ >= 64 && p_ + 8 <= end_)
            {
                uint64_t chunk = (uint64_t)acc_;
                std::memcpy(p_, &chunk, 8);
                p_ += 8;
                acc_ >>= 64;
                bits_ -= 64;
            }
            // Then write remaining full bytes (0..7)
            while (bits_ >= 8 && p_ < end_)
            {
                uint8_t b = (uint8_t)acc_;
                *p_++ = b;
                acc_ >>= 8;
                bits_ -= 8;
            }
        }

        inline void flush_if_full()
        {
            // If accumulator is getting full, flush out full bytes.
            if (BITIO_LIKELY(bits_ < ACC_BITS - 64))
                return;
            flush_bytes();
        }

        inline bool flush_all()
        {
            // Flush all full bytes and check capacity for trailing byte if any.
            flush_bytes();
            // No partial bytes should remain after finish() pad; if still <8 bits remain, we need one more.
            if (bits_ > 0)
            {
                if (p_ >= end_)
                    return false;
                uint8_t b = (uint8_t)acc_;
                *p_++ = b;
                acc_ >>= 8;
                bits_ = (bits_ >= 8) ? (bits_ - 8) : 0;
            }
            return true;
        }

        inline bool store_bytes(const void *src, size_t n)
        {
            if (end_ - p_ < (ptrdiff_t)n)
                return false;
            std::memcpy(p_, src, n);
            p_ += n;
            return true;
        }

    private:
        uint8_t *base_;
        uint8_t *p_;
        uint8_t *end_;
        acc_t acc_;
        unsigned bits_;
    };

} // namespace bitio
