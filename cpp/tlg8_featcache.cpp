#include "common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;
using tlg8::fastpath::fnv1a64;
using tlg8::fastpath::now_iso8601;

namespace {

constexpr int kMaxComponents = 4;
constexpr int kBlockEdge = 8;
constexpr int kOrientationBins = 8;
constexpr int kOrientationStats = 5;
constexpr int kOrientationDim = kOrientationBins + kOrientationStats;
constexpr int kFeatureDim = kMaxComponents * kBlockEdge * kBlockEdge + 3 + kOrientationDim;
constexpr float kPi = 3.14159265358979323846f;

struct Options {
    std::vector<fs::path> jsonl_paths;
    fs::path out_npy;
    fs::path out_scaler;
    fs::path out_idx;
    fs::path out_meta;
};

struct FileMeta {
    fs::path path;
    std::uint64_t size = 0;
    double mtime = 0.0;
};

double to_unix_seconds(const fs::file_time_type& tp) {
    using namespace std::chrono;
    const auto sctp = time_point_cast<system_clock::duration>(tp - fs::file_time_type::clock::now() + system_clock::now());
    const auto seconds = duration_cast<duration<double>>(sctp.time_since_epoch());
    return seconds.count();
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--jsonl") {
            if (i + 1 >= argc) {
                std::cerr << "--jsonl の引数が不足しています\n";
                return false;
            }
            opts.jsonl_paths.emplace_back(argv[++i]);
        } else if (arg == "--out-npy") {
            if (i + 1 >= argc) {
                std::cerr << "--out-npy の引数が不足しています\n";
                return false;
            }
            opts.out_npy = argv[++i];
        } else if (arg == "--out-scaler") {
            if (i + 1 >= argc) {
                std::cerr << "--out-scaler の引数が不足しています\n";
                return false;
            }
            opts.out_scaler = argv[++i];
        } else if (arg == "--out-idx") {
            if (i + 1 >= argc) {
                std::cerr << "--out-idx の引数が不足しています\n";
                return false;
            }
            opts.out_idx = argv[++i];
        } else if (arg == "--out-meta") {
            if (i + 1 >= argc) {
                std::cerr << "--out-meta の引数が不足しています\n";
                return false;
            }
            opts.out_meta = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "tlg8_featcache --jsonl data.jsonl --out-npy features.npy --out-scaler scaler.npz [--out-idx idx.bin] [--out-meta meta.json]\n";
            return false;
        } else {
            std::cerr << "未知のオプションです: " << arg << "\n";
            return false;
        }
    }
    if (opts.jsonl_paths.empty()) {
        std::cerr << "JSONL ファイルが指定されていません\n";
        return false;
    }
    if (opts.out_npy.empty() || opts.out_scaler.empty()) {
        std::cerr << "--out-npy と --out-scaler は必須です\n";
        return false;
    }
    if (opts.out_meta.empty()) {
        opts.out_meta = opts.out_npy;
        opts.out_meta += ".meta.json";
    }
    return true;
}

void write_npy_header(std::ostream& out, const std::vector<std::size_t>& shape, const std::string& descr) {
    std::ostringstream header_ss;
    header_ss << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        header_ss << shape[i];
        if (shape.size() == 1) {
            header_ss << ",";
        } else if (i + 1 < shape.size()) {
            header_ss << ", ";
        }
    }
    header_ss << "), }";
    std::string header = header_ss.str();
    std::size_t header_len = header.size() + 1;
    std::size_t padding = 16 - ((10 + header_len) % 16);
    if (padding == 16) {
        padding = 0;
    }
    header.append(padding, ' ');
    header.push_back('\n');

    const std::uint16_t header_size = static_cast<std::uint16_t>(header.size());
    const char magic[] = "\x93NUMPY";
    unsigned char version[2] = {1, 0};
    out.write(magic, 6);
    out.write(reinterpret_cast<const char*>(version), 2);
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), header.size());
}

std::vector<std::uint8_t> make_npy_buffer(const std::vector<float>& data, const std::vector<std::size_t>& shape) {
    std::ostringstream buffer;
    buffer.exceptions(std::ios::failbit | std::ios::badbit);
    write_npy_header(buffer, shape, "<f4");
    buffer.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    const std::string& str = buffer.str();
    return std::vector<std::uint8_t>(str.begin(), str.end());
}

uint32_t* crc32_table() {
    static bool initialized = false;
    static uint32_t table[256];
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                if (crc & 1U) {
                    crc = (crc >> 1) ^ 0xEDB88320u;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        initialized = true;
    }
    return table;
}

uint32_t compute_crc32(const std::uint8_t* data, std::size_t length) {
    uint32_t crc = 0xFFFFFFFFu;
    const auto* table = crc32_table();
    for (std::size_t i = 0; i < length; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFFu];
    }
    return crc ^ 0xFFFFFFFFu;
}

void write_le16(std::ostream& out, uint16_t value) {
    char buf[2];
    buf[0] = static_cast<char>(value & 0xFFu);
    buf[1] = static_cast<char>((value >> 8) & 0xFFu);
    out.write(buf, 2);
}

void write_le32(std::ostream& out, uint32_t value) {
    char buf[4];
    buf[0] = static_cast<char>(value & 0xFFu);
    buf[1] = static_cast<char>((value >> 8) & 0xFFu);
    buf[2] = static_cast<char>((value >> 16) & 0xFFu);
    buf[3] = static_cast<char>((value >> 24) & 0xFFu);
    out.write(buf, 4);
}

bool write_npz(const fs::path& path, const std::vector<std::uint8_t>& mean_buf, const std::vector<std::uint8_t>& std_buf) {
    struct Entry {
        const char* name;
        const std::vector<std::uint8_t>* data;
    } entries[] = {{"mean.npy", &mean_buf}, {"std.npy", &std_buf}};

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    std::vector<uint32_t> local_offsets;
    std::vector<uint32_t> crcs;
    local_offsets.reserve(2);
    crcs.reserve(2);

    for (const auto& entry : entries) {
        const auto& data = *entry.data;
        const uint32_t crc = compute_crc32(data.data(), data.size());
        crcs.push_back(crc);
        const uint32_t size = static_cast<uint32_t>(data.size());
        const uint16_t name_len = static_cast<uint16_t>(std::strlen(entry.name));
        const uint32_t local_offset = static_cast<uint32_t>(out.tellp());
        local_offsets.push_back(local_offset);

        write_le32(out, 0x04034B50u);
        write_le16(out, 20u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le32(out, crc);
        write_le32(out, size);
        write_le32(out, size);
        write_le16(out, name_len);
        write_le16(out, 0u);
        out.write(entry.name, name_len);
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }

    const uint32_t central_dir_offset = static_cast<uint32_t>(out.tellp());
    for (std::size_t idx = 0; idx < std::size(entries); ++idx) {
        const auto& entry = entries[idx];
        const auto& data = *entry.data;
        const uint32_t size = static_cast<uint32_t>(data.size());
        const uint16_t name_len = static_cast<uint16_t>(std::strlen(entry.name));
        write_le32(out, 0x02014B50u);
        write_le16(out, 20u);
        write_le16(out, 20u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le32(out, crcs[idx]);
        write_le32(out, size);
        write_le32(out, size);
        write_le16(out, name_len);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le16(out, 0u);
        write_le32(out, 0u);
        write_le32(out, local_offsets[idx]);
        out.write(entry.name, name_len);
    }
    const uint32_t central_dir_size = static_cast<uint32_t>(out.tellp()) - central_dir_offset;

    write_le32(out, 0x06054B50u);
    write_le16(out, 0u);
    write_le16(out, 0u);
    write_le16(out, static_cast<uint16_t>(std::size(entries)));
    write_le16(out, static_cast<uint16_t>(std::size(entries)));
    write_le32(out, central_dir_size);
    write_le32(out, central_dir_offset);
    write_le16(out, 0u);

    out.flush();
    return static_cast<bool>(out);
}

void compute_orientation_features(
    const std::vector<float>& mean_plane,
    int width,
    int height,
    std::array<float, kOrientationDim>& out
) {
    static constexpr float kSobelX[3][3] = {
        {1.0f, 0.0f, -1.0f},
        {2.0f, 0.0f, -2.0f},
        {1.0f, 0.0f, -1.0f},
    };
    static constexpr float kSobelY[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {-1.0f, -2.0f, -1.0f},
    };
    std::array<float, kOrientationBins> hist{};
    std::vector<float> magnitudes;
    magnitudes.reserve(static_cast<std::size_t>(width * height));
    float sum_mag = 0.0f;
    float sum_abs_gx = 0.0f;
    float sum_abs_gy = 0.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                const int yy = std::clamp(y + ky, 0, height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int xx = std::clamp(x + kx, 0, width - 1);
                    const float val = mean_plane[static_cast<std::size_t>(yy) * width + xx];
                    gx += kSobelX[ky + 1][kx + 1] * val;
                    gy += kSobelY[ky + 1][kx + 1] * val;
                }
            }
            const float mag = std::hypot(gx, gy);
            const float theta = std::atan2(gy, gx);
            int bin = static_cast<int>(std::floor((theta + kPi) * (kOrientationBins / (2.0f * kPi))));
            if (bin < 0) {
                bin = 0;
            } else if (bin >= kOrientationBins) {
                bin = kOrientationBins - 1;
            }
            hist[static_cast<std::size_t>(bin)] += mag;
            magnitudes.push_back(mag);
            sum_mag += mag;
            sum_abs_gx += std::fabs(gx);
            sum_abs_gy += std::fabs(gy);
        }
    }
    const float hist_total = std::accumulate(hist.begin(), hist.end(), 0.0f);
    if (hist_total > 0.0f) {
        for (float& value : hist) {
            value /= hist_total;
        }
    }
    float mean_mag = 0.0f;
    float p95 = 0.0f;
    float edge_ratio = 0.0f;
    float low_high_ratio = 0.0f;
    if (!magnitudes.empty()) {
        mean_mag = sum_mag / static_cast<float>(magnitudes.size());
        std::vector<float> sorted = magnitudes;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t max_index = sorted.size() - 1;
        const std::size_t idx = static_cast<std::size_t>(std::floor(0.95f * static_cast<float>(max_index)));
        p95 = sorted[std::min(idx, max_index)];
        if (mean_mag > 0.0f) {
            const float threshold = mean_mag * 1.5f;
            std::size_t count = 0;
            for (float mag : magnitudes) {
                if (mag > threshold) {
                    ++count;
                }
            }
            edge_ratio = static_cast<float>(count) / static_cast<float>(magnitudes.size());
        }
        const std::size_t quart = std::max<std::size_t>(1, magnitudes.size() / 4);
        if (quart > 0) {
            float low_sum = 0.0f;
            float high_sum = 0.0f;
            for (std::size_t i = 0; i < quart; ++i) {
                low_sum += sorted[i];
                high_sum += sorted[sorted.size() - 1 - i];
            }
            const float low_mean = low_sum / static_cast<float>(quart);
            const float high_mean = high_sum / static_cast<float>(quart);
            low_high_ratio = low_mean / (high_mean + 1e-6f);
        }
    }
    const float hv_balance = (sum_abs_gx - sum_abs_gy) / (sum_abs_gx + sum_abs_gy + 1e-6f);
    for (std::size_t i = 0; i < kOrientationBins; ++i) {
        out[i] = hist[i];
    }
    out[kOrientationBins + 0] = mean_mag;
    out[kOrientationBins + 1] = p95;
    out[kOrientationBins + 2] = edge_ratio;
    out[kOrientationBins + 3] = hv_balance;
    out[kOrientationBins + 4] = low_high_ratio;
}

bool build_feature(const json& record, std::vector<float>& out) {
    out.assign(kFeatureDim, 0.0f);
    if (!record.contains("pixels") || !record.contains("block_size") || !record.contains("components")) {
        return false;
    }
    const auto& pixels_json = record["pixels"];
    const auto& block_size_json = record["block_size"];
    const auto& components_json = record["components"];
    if (!pixels_json.is_array() || !block_size_json.is_array() || block_size_json.size() < 2) {
        return false;
    }
    const int components = components_json.get<int>();
    if (components <= 0 || components > kMaxComponents) {
        return false;
    }
    const int block_w = block_size_json.at(0).get<int>();
    const int block_h = block_size_json.at(1).get<int>();
    if (block_w <= 0 || block_w > kBlockEdge || block_h <= 0 || block_h > kBlockEdge) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(block_w) * static_cast<std::size_t>(block_h) * static_cast<std::size_t>(components);
    if (pixels_json.size() < expected) {
        return false;
    }
    const std::size_t plane_stride = static_cast<std::size_t>(kBlockEdge * kBlockEdge);
    const std::size_t base_offset = plane_stride * kMaxComponents;
    std::vector<float> mean_plane(static_cast<std::size_t>(block_w * block_h), 0.0f);
    for (int c = 0; c < components; ++c) {
        for (int y = 0; y < block_h; ++y) {
            for (int x = 0; x < block_w; ++x) {
                const std::size_t src_index = (static_cast<std::size_t>(y) * block_w + x) * components + c;
                const int raw = pixels_json.at(src_index).get<int>();
                const float normalized = static_cast<float>(std::clamp(raw, 0, 255)) / 255.0f;
                const std::size_t dst_index = static_cast<std::size_t>(c) * plane_stride + static_cast<std::size_t>(y) * kBlockEdge + x;
                out[dst_index] = normalized;
                const std::size_t pix_index = static_cast<std::size_t>(y) * block_w + x;
                mean_plane[pix_index] += normalized;
            }
        }
    }
    if (components > 0) {
        const float inv_comp = 1.0f / static_cast<float>(components);
        for (float& value : mean_plane) {
            value *= inv_comp;
        }
    }
    std::array<float, kOrientationDim> orientation{};
    if (block_w > 0 && block_h > 0) {
        compute_orientation_features(mean_plane, block_w, block_h, orientation);
    }
    out[base_offset + 0] = static_cast<float>(block_w) / 8.0f;
    out[base_offset + 1] = static_cast<float>(block_h) / 8.0f;
    out[base_offset + 2] = static_cast<float>(components) / 4.0f;
    for (int i = 0; i < kOrientationDim; ++i) {
        out[base_offset + 3 + i] = orientation[static_cast<std::size_t>(i)];
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }

    std::vector<FileMeta> files;
    files.reserve(opts.jsonl_paths.size());

    std::uint64_t dataset_hash = 1469598103934665603ull;
    std::uint64_t record_count = 0;

    for (const auto& path : opts.jsonl_paths) {
        if (!fs::exists(path)) {
            std::cerr << "入力ファイルが存在しません: " << path << "\n";
            return 1;
        }
        FileMeta meta;
        meta.path = fs::absolute(path);
        meta.size = static_cast<std::uint64_t>(fs::file_size(path));
        meta.mtime = to_unix_seconds(fs::last_write_time(path));
        files.push_back(meta);

        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "ファイルを開けませんでした: " << path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            dataset_hash = fnv1a64(line.data(), line.size(), dataset_hash);
            dataset_hash = fnv1a64("\n", 1, dataset_hash);
            ++record_count;
        }
    }

    if (record_count == 0) {
        std::cerr << "入力 JSONL に有効な行がありません\n";
        return 1;
    }

    fs::create_directories(opts.out_npy.parent_path());
    fs::create_directories(opts.out_scaler.parent_path());
    if (!opts.out_idx.empty()) {
        fs::create_directories(opts.out_idx.parent_path());
    }
    fs::create_directories(opts.out_meta.parent_path());

    std::ofstream feature_out(opts.out_npy, std::ios::binary);
    if (!feature_out) {
        std::cerr << "特徴量ファイルを開けませんでした: " << opts.out_npy << "\n";
        return 1;
    }
    write_npy_header(feature_out, {static_cast<std::size_t>(record_count), static_cast<std::size_t>(kFeatureDim)}, "<f4");

    std::ofstream idx_out;
    if (!opts.out_idx.empty()) {
        idx_out.open(opts.out_idx, std::ios::binary);
        if (!idx_out) {
            std::cerr << "インデックスファイルを開けませんでした: " << opts.out_idx << "\n";
            return 1;
        }
    }

    std::vector<double> sum(kFeatureDim, 0.0);
    std::vector<double> sumsq(kFeatureDim, 0.0);
    std::vector<float> feature(kFeatureDim, 0.0f);

    std::uint64_t processed = 0;
    for (std::size_t file_id = 0; file_id < opts.jsonl_paths.size(); ++file_id) {
        const auto& path = opts.jsonl_paths[file_id];
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "ファイルを開けませんでした: " << path << "\n";
            return 1;
        }
        std::string line;
        while (true) {
            std::streampos start_pos = in.tellg();
            if (!std::getline(in, line)) {
                break;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            json record;
            bool ok = false;
            try {
                record = json::parse(line);
                ok = build_feature(record, feature);
            } catch (const std::exception&) {
                ok = false;
            }
            if (!ok) {
                feature.assign(kFeatureDim, 0.0f);
            }
            feature_out.write(reinterpret_cast<const char*>(feature.data()), static_cast<std::streamsize>(feature.size() * sizeof(float)));
            if (!feature_out) {
                std::cerr << "特徴量書き出し中にエラーが発生しました\n";
                return 1;
            }
            for (int i = 0; i < kFeatureDim; ++i) {
                const double val = static_cast<double>(feature[i]);
                sum[i] += val;
                sumsq[i] += val * val;
            }
            if (idx_out) {
                const std::uint32_t file_id_u32 = static_cast<std::uint32_t>(file_id);
                const std::uint64_t offset = static_cast<std::uint64_t>(start_pos);
                idx_out.write(reinterpret_cast<const char*>(&file_id_u32), sizeof(file_id_u32));
                idx_out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
                if (!idx_out) {
                    std::cerr << "インデックス書き出し中にエラーが発生しました\n";
                    return 1;
                }
            }
            ++processed;
        }
    }

    if (processed != record_count) {
        std::cerr << "行数が一致しません: " << processed << " vs " << record_count << "\n";
        return 1;
    }

    feature_out.flush();
    if (!feature_out) {
        std::cerr << "特徴量ファイル書き出し時にエラーが発生しました\n";
        return 1;
    }
    if (idx_out) {
        idx_out.flush();
        if (!idx_out) {
            std::cerr << "インデックスファイル書き出し時にエラーが発生しました\n";
            return 1;
        }
    }

    std::vector<float> mean(kFeatureDim, 0.0f);
    std::vector<float> stddev(kFeatureDim, 1.0f);
    const double count = static_cast<double>(record_count);
    for (int i = 0; i < kFeatureDim; ++i) {
        const double mean_val = (count > 0.0) ? sum[i] / count : 0.0;
        double variance = 1.0;
        if (count > 1.0) {
            variance = (sumsq[i] - (sum[i] * sum[i]) / count) / (count - 1.0);
        }
        variance = std::max(variance, 1e-12);
        double std_val = std::sqrt(variance);
        if (std_val < 1e-6) {
            std_val = 1.0;
        }
        mean[i] = static_cast<float>(mean_val);
        stddev[i] = static_cast<float>(std_val);
    }

    const auto mean_buf = make_npy_buffer(mean, {static_cast<std::size_t>(kFeatureDim)});
    const auto std_buf = make_npy_buffer(stddev, {static_cast<std::size_t>(kFeatureDim)});
    if (!write_npz(opts.out_scaler, mean_buf, std_buf)) {
        std::cerr << "スケーラー NPZ の書き出しに失敗しました\n";
        return 1;
    }

    json meta;
    meta["format"] = "tlg8-ranker-features";
    meta["version"] = 1;
    meta["created_at"] = now_iso8601();
    meta["n_samples"] = record_count;
    meta["feature_dim"] = kFeatureDim;
    meta["jsonl_hash"] = dataset_hash;
    meta["source_files"] = json::array();
    for (const auto& fm : files) {
        json f;
        f["path"] = fm.path.string();
        f["size"] = fm.size;
        f["mtime"] = fm.mtime;
        meta["source_files"].push_back(f);
    }
    meta["npy_path"] = opts.out_npy.string();
    meta["scaler_path"] = opts.out_scaler.string();
    if (!opts.out_idx.empty()) {
        meta["idx_path"] = opts.out_idx.string();
    }

    std::ofstream meta_out(opts.out_meta);
    if (!meta_out) {
        std::cerr << "メタデータを書き出せませんでした: " << opts.out_meta << "\n";
        return 1;
    }
    meta_out << meta.dump(2) << '\n';
    meta_out.flush();
    if (!meta_out) {
        std::cerr << "メタデータ書き出し中にエラーが発生しました\n";
        return 1;
    }

    std::cout << "特徴量キャッシュを生成しました (サンプル数=" << record_count << ", 次元=" << kFeatureDim << ")\n";
    return 0;
}
