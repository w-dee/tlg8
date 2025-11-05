#pragma once

// 共通ユーティリティ関数群。

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tlg8::fastpath {

// 欠損値として用いるラベル値。
constexpr std::uint16_t kInvalidLabel = 0xFFFFu;

// FNV-1a 64bit ハッシュを計算する。
inline std::uint64_t fnv1a64(const void* data, std::size_t length, std::uint64_t seed = 1469598103934665603ull) {
    const std::uint8_t* ptr = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = seed;
    for (std::size_t i = 0; i < length; ++i) {
        hash ^= static_cast<std::uint64_t>(ptr[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

// 文字列に対して FNV-1a 64bit ハッシュを適用する。
inline std::uint64_t fnv1a64_str(const std::string& text, std::uint64_t seed = 1469598103934665603ull) {
    return fnv1a64(text.data(), text.size(), seed);
}

// 現在時刻を ISO-8601 形式 (UTC) の文字列として返す。
inline std::string now_iso8601() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto time = clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// カンマ区切り文字列を分割する (空要素は除去する)。
inline std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> result;
    std::string current;
    for (char ch : value) {
        if (ch == ',') {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

}  // namespace tlg8::fastpath
