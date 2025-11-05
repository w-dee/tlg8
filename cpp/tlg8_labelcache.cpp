#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;
using tlg8::fastpath::fnv1a64;
using tlg8::fastpath::kInvalidLabel;
using tlg8::fastpath::now_iso8601;
using tlg8::fastpath::split_csv;

namespace {

// C++ ツール側で固定するヘッド名と ID。
struct HeadSpec {
    const char* name;
    std::uint32_t id;
};

constexpr HeadSpec kKnownHeads[] = {
    {"predictor", 1},
    {"reorder", 2},
    {"interleave", 3},
    {"filter_primary", 4},
    {"filter_secondary", 5},
};

struct Options {
    std::vector<fs::path> jsonl_paths;
    fs::path out_bin;
    fs::path out_meta;
    fs::path out_topk;
    std::vector<std::string> heads;
    int topk = 0;
};

struct FilterParts {
    int perm = -1;
    int primary = -1;
    int secondary = -1;
};

struct HeadDesc {
    std::uint32_t name_id;
    std::uint32_t n_classes;
    std::uint32_t stride;
    std::uint32_t offset;
};

#pragma pack(push, 1)
struct LabelsHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t n_heads;
    std::uint64_t n_samples;
    std::uint64_t head_desc_off;
    std::uint64_t data_off;
    std::uint64_t jsonl_hash;
};
#pragma pack(pop)

static_assert(sizeof(LabelsHeader) == 48, "LabelsHeader サイズが想定と異なります");
static_assert(sizeof(HeadDesc) == 16, "HeadDesc サイズが想定と異なります");

struct FileMeta {
    fs::path path;
    std::uint64_t size = 0;
    double mtime = 0.0;
};

std::optional<HeadSpec> find_head(const std::string& name) {
    for (const auto& head : kKnownHeads) {
        if (name == head.name) {
            return head;
        }
    }
    return std::nullopt;
}

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
        } else if (arg == "--out-bin") {
            if (i + 1 >= argc) {
                std::cerr << "--out-bin の引数が不足しています\n";
                return false;
            }
            opts.out_bin = argv[++i];
        } else if (arg == "--out-meta") {
            if (i + 1 >= argc) {
                std::cerr << "--out-meta の引数が不足しています\n";
                return false;
            }
            opts.out_meta = argv[++i];
        } else if (arg == "--out-topk") {
            if (i + 1 >= argc) {
                std::cerr << "--out-topk の引数が不足しています\n";
                return false;
            }
            opts.out_topk = argv[++i];
        } else if (arg == "--topk") {
            if (i + 1 >= argc) {
                std::cerr << "--topk の引数が不足しています\n";
                return false;
            }
            opts.topk = std::stoi(argv[++i]);
            if (opts.topk < 0) {
                std::cerr << "--topk には 0 以上を指定してください\n";
                return false;
            }
        } else if (arg == "--heads") {
            if (i + 1 >= argc) {
                std::cerr << "--heads の引数が不足しています\n";
                return false;
            }
            opts.heads = split_csv(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "tlg8_labelcache --jsonl file1.jsonl [--jsonl file2.jsonl ...] \\\n  --out-bin output.bin --out-meta output.json [--out-topk topk.bin] [--topk K] [--heads list]\n";
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
    if (opts.out_bin.empty() || opts.out_meta.empty()) {
        std::cerr << "--out-bin と --out-meta は必須です\n";
        return false;
    }
    if (opts.heads.empty()) {
        opts.heads = {"predictor", "reorder", "interleave", "filter_primary", "filter_secondary"};
    }
    for (const auto& head : opts.heads) {
        if (!find_head(head)) {
            std::cerr << "未対応のヘッド名が指定されました: " << head << "\n";
            return false;
        }
    }
    if (!opts.out_topk.empty() && opts.topk <= 0) {
        opts.topk = 2;
    }
    if (opts.topk > 0 && opts.out_topk.empty()) {
        std::cerr << "警告: --topk が指定されましたが --out-topk が未指定のため topK 出力は無効化されます\n";
    }
    return true;
}

FilterParts split_filter(int code) {
    if (code < 0) {
        return {-1, -1, -1};
    }
    const int perm = ((code >> 4) & 0x7) % 6;
    const int primary = ((code >> 2) & 0x3) % 4;
    const int secondary = (code & 0x3) % 4;
    return {perm, primary, secondary};
}

int decode_int(const json& value, int fallback) {
    if (value.is_number_integer()) {
        return static_cast<int>(value.get<std::int64_t>());
    }
    if (value.is_number_unsigned()) {
        return static_cast<int>(value.get<std::uint64_t>());
    }
    if (value.is_number_float()) {
        return static_cast<int>(value.get<double>());
    }
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        try {
            size_t idx = 0;
            const int v = std::stoi(text, &idx, 0);
            if (idx == text.size()) {
                return v;
            }
        } catch (...) {
        }
    }
    return fallback;
}

int extract_scalar(const json& entry, const std::string& key, int fallback) {
    const auto it = entry.find(key);
    if (it == entry.end()) {
        return fallback;
    }
    return decode_int(*it, fallback);
}

json load_json(const std::string& line) {
    try {
        return json::parse(line);
    } catch (const std::exception& exc) {
        std::cerr << "JSON パースに失敗しました: " << exc.what() << "\n";
        throw;
    }
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
            ++record_count;
            dataset_hash = fnv1a64(line.data(), line.size(), dataset_hash);
            dataset_hash = fnv1a64("\n", 1, dataset_hash);
        }
    }

    if (record_count == 0) {
        std::cerr << "入力 JSONL に有効な行がありません\n";
        return 1;
    }

    std::vector<HeadDesc> head_descs;
    head_descs.reserve(opts.heads.size());
    std::unordered_map<std::string, std::size_t> head_index;
    head_index.reserve(opts.heads.size());
    std::vector<int> max_label(opts.heads.size(), -1);

    std::uint32_t offset = 0;
    for (std::size_t idx = 0; idx < opts.heads.size(); ++idx) {
        const auto spec = find_head(opts.heads[idx]).value();
        HeadDesc desc{};
        desc.name_id = spec.id;
        desc.n_classes = 0;  // 後で更新
        desc.stride = sizeof(std::uint16_t);
        desc.offset = offset;
        offset += desc.stride;
        head_descs.push_back(desc);
        head_index.emplace(opts.heads[idx], idx);
    }
    const std::uint32_t sample_stride = offset;

    const bool write_topk = (!opts.out_topk.empty() && opts.topk > 0);

    if (!opts.out_bin.parent_path().empty()) {
        fs::create_directories(opts.out_bin.parent_path());
    }
    if (!opts.out_meta.parent_path().empty()) {
        fs::create_directories(opts.out_meta.parent_path());
    }
    if (write_topk && !opts.out_topk.parent_path().empty()) {
        fs::create_directories(opts.out_topk.parent_path());
    }

    std::ofstream bin_out(opts.out_bin, std::ios::binary);
    if (!bin_out) {
        std::cerr << "出力ファイルを開けませんでした: " << opts.out_bin << "\n";
        return 1;
    }

    LabelsHeader header{};
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, "TLG8LBL", 7);
    header.version = 1;
    header.n_heads = static_cast<std::uint32_t>(head_descs.size());
    header.n_samples = record_count;
    header.head_desc_off = sizeof(LabelsHeader);
    header.data_off = header.head_desc_off + head_descs.size() * sizeof(HeadDesc);
    header.jsonl_hash = dataset_hash;

    bin_out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    bin_out.write(reinterpret_cast<const char*>(head_descs.data()), head_descs.size() * sizeof(HeadDesc));

    std::ofstream topk_out;
    LabelsHeader topk_header{};
    std::vector<HeadDesc> topk_descs;
    std::uint32_t topk_stride = 0;
    if (write_topk) {
        topk_out.open(opts.out_topk, std::ios::binary);
        if (!topk_out) {
            std::cerr << "topK 出力ファイルを開けませんでした: " << opts.out_topk << "\n";
            return 1;
        }
        std::memset(&topk_header, 0, sizeof(topk_header));
        std::memcpy(topk_header.magic, "TLG8TK\0", 7);
        topk_header.version = 1;
        topk_header.n_heads = header.n_heads;
        topk_header.n_samples = header.n_samples;
        topk_header.head_desc_off = sizeof(LabelsHeader);
        topk_header.data_off = topk_header.head_desc_off + header.n_heads * sizeof(HeadDesc);
        topk_header.jsonl_hash = dataset_hash;

        topk_descs.reserve(head_descs.size());
        std::uint32_t tk_offset = 0;
        for (const auto& desc : head_descs) {
            HeadDesc tk = desc;
            tk.n_classes = desc.n_classes;
            tk.stride = static_cast<std::uint32_t>(sizeof(std::uint16_t) * static_cast<std::uint32_t>(opts.topk));
            tk.offset = tk_offset;
            tk_offset += tk.stride;
            topk_descs.push_back(tk);
        }
        topk_stride = tk_offset;
        topk_out.write(reinterpret_cast<const char*>(&topk_header), sizeof(topk_header));
        topk_out.write(reinterpret_cast<const char*>(topk_descs.data()), topk_descs.size() * sizeof(HeadDesc));
    }

    std::vector<std::uint16_t> best_buffer(head_descs.size());
    std::vector<std::uint16_t> topk_buffer(write_topk ? (head_descs.size() * opts.topk) : 0);

    std::uint64_t processed = 0;
    for (const auto& path : opts.jsonl_paths) {
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
            if (line.empty()) {
                std::fill(best_buffer.begin(), best_buffer.end(), kInvalidLabel);
                if (write_topk) {
                    std::fill(topk_buffer.begin(), topk_buffer.end(), kInvalidLabel);
                }
                bin_out.write(reinterpret_cast<const char*>(best_buffer.data()), sample_stride);
                if (write_topk) {
                    topk_out.write(reinterpret_cast<const char*>(topk_buffer.data()), topk_stride);
                }
                ++processed;
                continue;
            }
            json record;
            try {
                record = json::parse(line);
            } catch (const std::exception&) {
                std::cerr << "JSON 行の解析に失敗したため欠損扱いにします (行=" << processed << ")\n";
                std::fill(best_buffer.begin(), best_buffer.end(), kInvalidLabel);
                if (write_topk) {
                    std::fill(topk_buffer.begin(), topk_buffer.end(), kInvalidLabel);
                }
                bin_out.write(reinterpret_cast<const char*>(best_buffer.data()), sample_stride);
                if (write_topk) {
                    topk_out.write(reinterpret_cast<const char*>(topk_buffer.data()), topk_stride);
                }
                ++processed;
                continue;
            }

            json best_entry = record.contains("best") ? record["best"] : json::object();
            if (!best_entry.is_object()) {
                best_entry = json::object();
            }
            json second_entry = record.contains("second") ? record["second"] : json::object();
            if (!second_entry.is_object()) {
                second_entry = json::object();
            }

            const int best_filter_code = extract_scalar(best_entry, "filter", -1);
            const int second_filter_code = extract_scalar(second_entry, "filter", -1);
            const FilterParts best_filter = split_filter(best_filter_code);
            const FilterParts second_filter = split_filter(second_filter_code);

            for (std::size_t head_idx = 0; head_idx < opts.heads.size(); ++head_idx) {
                const auto& name = opts.heads[head_idx];
                int best_value = -1;
                int second_value = -1;
                if (name == "predictor") {
                    best_value = extract_scalar(best_entry, "predictor", -1);
                    second_value = extract_scalar(second_entry, "predictor", -1);
                } else if (name == "reorder") {
                    best_value = extract_scalar(best_entry, "reorder", -1);
                    second_value = extract_scalar(second_entry, "reorder", -1);
                } else if (name == "interleave") {
                    best_value = extract_scalar(best_entry, "interleave", -1);
                    second_value = extract_scalar(second_entry, "interleave", -1);
                } else if (name == "filter_primary") {
                    best_value = best_filter.primary;
                    second_value = second_filter.primary;
                } else if (name == "filter_secondary") {
                    best_value = best_filter.secondary;
                    second_value = second_filter.secondary;
                } else {
                    best_value = -1;
                    second_value = -1;
                }

                if (best_value >= 0 && best_value <= std::numeric_limits<std::uint16_t>::max()) {
                    best_buffer[head_idx] = static_cast<std::uint16_t>(best_value);
                    max_label[head_idx] = std::max(max_label[head_idx], best_value);
                } else {
                    best_buffer[head_idx] = kInvalidLabel;
                }

                if (write_topk) {
                    const std::size_t base = head_idx * opts.topk;
                    if (opts.topk >= 1) {
                        topk_buffer[base] = best_buffer[head_idx];
                    }
                    for (int k = 1; k < opts.topk; ++k) {
                        topk_buffer[base + k] = kInvalidLabel;
                    }
                    if (opts.topk >= 2) {
                        if (second_value >= 0 && second_value <= std::numeric_limits<std::uint16_t>::max()) {
                            topk_buffer[base + 1] = static_cast<std::uint16_t>(second_value);
                            max_label[head_idx] = std::max(max_label[head_idx], second_value);
                        } else {
                            topk_buffer[base + 1] = kInvalidLabel;
                        }
                    }
                } else {
                    if (second_value >= 0) {
                        max_label[head_idx] = std::max(max_label[head_idx], second_value);
                    }
                }
            }

            bin_out.write(reinterpret_cast<const char*>(best_buffer.data()), sample_stride);
            if (write_topk) {
                topk_out.write(reinterpret_cast<const char*>(topk_buffer.data()), topk_stride);
            }
            ++processed;
        }
    }

    if (processed != record_count) {
        std::cerr << "行数カウントと書き込み件数が一致しません: " << processed << " vs " << record_count << "\n";
        return 1;
    }

    bin_out.flush();
    if (!bin_out) {
        std::cerr << "バイナリ書き出し中にエラーが発生しました\n";
        return 1;
    }
    if (write_topk) {
        topk_out.flush();
        if (!topk_out) {
            std::cerr << "topK バイナリ書き出し中にエラーが発生しました\n";
            return 1;
        }
    }

    for (std::size_t idx = 0; idx < head_descs.size(); ++idx) {
        const int max_val = max_label[idx];
        if (max_val >= 0) {
            head_descs[idx].n_classes = static_cast<std::uint32_t>(max_val + 1);
        } else {
            head_descs[idx].n_classes = 0;
        }
    }

    bin_out.seekp(sizeof(LabelsHeader), std::ios::beg);
    bin_out.write(reinterpret_cast<const char*>(head_descs.data()), head_descs.size() * sizeof(HeadDesc));
    bin_out.seekp(0, std::ios::end);
    bin_out.flush();

    if (write_topk) {
        for (std::size_t idx = 0; idx < topk_descs.size(); ++idx) {
            topk_descs[idx].n_classes = head_descs[idx].n_classes;
        }
        topk_out.seekp(sizeof(LabelsHeader), std::ios::beg);
        topk_out.write(reinterpret_cast<const char*>(topk_descs.data()), topk_descs.size() * sizeof(HeadDesc));
        topk_out.seekp(0, std::ios::end);
        topk_out.flush();
    }

    json meta;
    meta["format"] = "tlg8-ranker-labels";
    meta["version"] = 1;
    meta["created_at"] = now_iso8601();
    meta["n_samples"] = record_count;
    meta["jsonl_hash"] = dataset_hash;
    meta["invalid_value"] = kInvalidLabel;
    meta["heads"] = json::array();
    for (std::size_t idx = 0; idx < head_descs.size(); ++idx) {
        json head_json;
        head_json["name"] = opts.heads[idx];
        head_json["name_id"] = head_descs[idx].name_id;
        head_json["n_classes"] = head_descs[idx].n_classes;
        head_json["stride"] = head_descs[idx].stride;
        head_json["offset"] = head_descs[idx].offset;
        meta["heads"].push_back(head_json);
    }
    meta["source_files"] = json::array();
    for (const auto& fm : files) {
        json file_json;
        file_json["path"] = fm.path.string();
        file_json["size"] = fm.size;
        file_json["mtime"] = fm.mtime;
        meta["source_files"].push_back(file_json);
    }
    if (write_topk) {
        json topk_meta;
        topk_meta["path"] = opts.out_topk.string();
        topk_meta["k"] = opts.topk;
        topk_meta["stride"] = topk_stride;
        meta["topk"] = topk_meta;
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

    std::cout << "ラベルキャッシュを生成しました (サンプル数=" << record_count << ", ヘッド=" << head_descs.size() << ")\n";
    return 0;
}
