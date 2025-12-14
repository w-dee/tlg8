#!/usr/bin/env python3
"""
pack_artifacts.py が生成した packed ディレクトリから、学習用の NumPy 配列（features/labels）を生成します。

入力（--in-dir）:
  - training.all.jsonl : 1行=1ブロックの JSONL（pixels + 座標など）
  - labels.all.bin     : LabelRecord(128B) の連結（training と同一順序）
  - index.jsonl        : 1行=1画像（training/label のオフセットと件数）

出力（--out-dir）:
  - features.npy         : float32 [N, D]
  - labels_reorder.npy   : int64   [N]
  - feature_mean.npy     : float32 [D]
  - feature_std.npy      : float32 [D]
  - index.jsonl          : 1行=1ブロック（追跡用）
  - meta.json            : メタ情報（特徴量仕様、次元、正規化の説明など）

注意:
  - JSONL は巨大になりやすいので、全件を list にせず行単位で処理します。
  - 特徴量は「FEATURE の試行回数削減」が目的のため、読みやすさ・改造容易性を優先します。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm


LABEL_RECORD_SIZE = 128
LABEL_RECORD_STRUCT = struct.Struct("<IHH12hI92s")  # 128 bytes
LABEL_MAGIC = 0x4C424C38  # "LBL8"
LABEL_VERSION = 1

FEATURE_SET_RAW_PLUS_STATS_V1 = "raw_plus_stats_v1"

BLOCK_W = 8
BLOCK_H = 8
RAW_RGB_DIMS = BLOCK_W * BLOCK_H * 3  # 192

EDGE_THRESHOLD_Y = 20.0 / 255.0
HF_NORM_EPS = 1e-6


@dataclass(frozen=True)
class PackedImageEntry:
    relpath: str
    training_line_offset: int
    training_line_count: int
    label_record_offset: int
    label_record_count: int
    label_record_size: int


@dataclass
class _WorkBuffers:
    y: np.ndarray  # float32 [8,8]
    cb: np.ndarray  # float32 [8,8]
    cr: np.ndarray  # float32 [8,8]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"真偽値として解釈できません: {value!r}（例: true/false）")


def _load_packed_index(path: Path, limit_images: Optional[int]) -> list[PackedImageEntry]:
    entries: list[PackedImageEntry] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="packed index (images)", unit="img"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entry = PackedImageEntry(
                relpath=str(obj["relpath"]),
                training_line_offset=int(obj["training_line_offset"]),
                training_line_count=int(obj["training_line_count"]),
                label_record_offset=int(obj["label_record_offset"]),
                label_record_count=int(obj["label_record_count"]),
                label_record_size=int(obj.get("label_record_size", LABEL_RECORD_SIZE)),
            )
            if entry.label_record_size != LABEL_RECORD_SIZE:
                raise ValueError(
                    f"label_record_size が想定と不一致です: {entry.label_record_size} expected={LABEL_RECORD_SIZE}"
                )
            if entry.training_line_count != entry.label_record_count:
                raise ValueError(
                    "training_line_count と label_record_count が不一致です: "
                    f"relpath={entry.relpath} lines={entry.training_line_count} records={entry.label_record_count}"
                )
            entries.append(entry)
            if limit_images is not None and len(entries) >= limit_images:
                break
    if not entries:
        raise ValueError(f"packed index が空です: {path}")
    return entries


def _feature_names_raw_plus_stats_v1() -> list[str]:
    names: list[str] = []
    for y in range(BLOCK_H):
        for x in range(BLOCK_W):
            names.append(f"rgb_y{y}_x{x}_r")
            names.append(f"rgb_y{y}_x{x}_g")
            names.append(f"rgb_y{y}_x{x}_b")
    names.extend(
        [
            "block_w",
            "block_h",
            "mean_Y",
            "var_Y",
            "var_Cb",
            "var_Cr",
            "Gx_Y",
            "Gy_Y",
            "edge_density_Y",
            "hf_energy_Y",
            "hf_norm_Y",
        ]
    )
    return names


def _feature_dim(feature_set: str) -> int:
    if feature_set != FEATURE_SET_RAW_PLUS_STATS_V1:
        raise ValueError(f"未対応の --feature-set: {feature_set}")
    return len(_feature_names_raw_plus_stats_v1())


def _check_overwrite(out_dir: Path, force: bool, dry_run: bool) -> None:
    targets = [
        out_dir / "features.npy",
        out_dir / "labels_reorder.npy",
        out_dir / "feature_mean.npy",
        out_dir / "feature_std.npy",
        out_dir / "index.jsonl",
        out_dir / "meta.json",
    ]
    existing = [p for p in targets if p.exists()]
    if not existing:
        return
    msg = "出力ファイルが既に存在します: " + ", ".join(str(p) for p in existing)
    if dry_run:
        print("WARN:", msg, "（dry-run のため書き込みは行いません）")
        return
    if not force:
        raise FileExistsError(msg + "（上書きするには --force）")
    print("WARN:", msg, "（--force により上書きします）")


def _validate_inputs(in_dir: Path) -> dict[str, Path]:
    paths = {
        "training": in_dir / "training.all.jsonl",
        "labels": in_dir / "labels.all.bin",
        "index": in_dir / "index.jsonl",
    }
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {k}={p}")
    return paths


def _read_label_best_reorder(fp) -> int:
    chunk = fp.read(LABEL_RECORD_SIZE)
    if len(chunk) != LABEL_RECORD_SIZE:
        raise ValueError("labels.all.bin の途中でファイルが終わりました")
    magic, version, reserved, *rest = LABEL_RECORD_STRUCT.unpack(chunk)
    labels = rest[:12]
    if magic != LABEL_MAGIC:
        raise ValueError(f"LabelRecord magic 不一致: 0x{magic:08x}")
    if version != LABEL_VERSION:
        raise ValueError(f"LabelRecord version 不明: {version}")
    if reserved != 0:
        raise ValueError(f"LabelRecord reserved が 0 ではありません: {reserved}")
    best_reorder = int(labels[4])
    if not (0 <= best_reorder <= 7):
        raise ValueError(f"best_reorder が範囲外です: {best_reorder}")
    return best_reorder


def _fill_features_raw_plus_stats_v1(dst_row: np.ndarray, row: dict[str, Any], work: _WorkBuffers) -> dict[str, Any]:
    block_w = int(row["block_size"][0])
    block_h = int(row["block_size"][1])
    components = int(row["components"])
    pixels = row["pixels"]

    if components not in (3, 4):
        raise ValueError(f"components が不正です: {components}")
    if not (1 <= block_w <= BLOCK_W and 1 <= block_h <= BLOCK_H):
        raise ValueError(f"block_size が不正です: {(block_w, block_h)}")
    expected_len = block_w * block_h * components
    if len(pixels) != expected_len:
        raise ValueError(f"pixels の長さが不一致です: len={len(pixels)} expected={expected_len}")

    # raw RGB (8x8) はゼロパディング前提で、最初に 0 で埋める
    dst_row[:RAW_RGB_DIMS] = 0.0

    # Y/Cb/Cr は 8x8 のワークバッファを使い回す（実ブロック範囲だけ参照する）
    y_buf = work.y
    cb_buf = work.cb
    cr_buf = work.cr

    inv255 = 1.0 / 255.0
    raw_out = dst_row[:RAW_RGB_DIMS]

    # pixels は y -> x -> comp（components==4 の場合は ARGB で A を捨てる）
    for yy in range(block_h):
        for xx in range(block_w):
            if components == 3:
                base = (yy * block_w + xx) * 3
                r = float(pixels[base + 0]) * inv255
                g = float(pixels[base + 1]) * inv255
                b = float(pixels[base + 2]) * inv255
            else:
                base = (yy * block_w + xx) * 4
                r = float(pixels[base + 1]) * inv255
                g = float(pixels[base + 2]) * inv255
                b = float(pixels[base + 3]) * inv255

            out_base = (yy * BLOCK_W + xx) * 3
            raw_out[out_base + 0] = r
            raw_out[out_base + 1] = g
            raw_out[out_base + 2] = b

            # 軽量な Y/Cb/Cr ライク変換（R/G/B は [0,1]）
            yv = 0.299 * r + 0.587 * g + 0.114 * b
            y_buf[yy, xx] = yv
            cb_buf[yy, xx] = b - yv
            cr_buf[yy, xx] = r - yv

    # 追加: block_w/block_h（実サイズ）
    offset = RAW_RGB_DIMS
    dst_row[offset + 0] = float(block_w)
    dst_row[offset + 1] = float(block_h)
    offset += 2

    # 統計量（実ブロック範囲のみ）
    n = float(block_w * block_h)
    sum_y = 0.0
    sum_cb = 0.0
    sum_cr = 0.0
    for yy in range(block_h):
        for xx in range(block_w):
            sum_y += float(y_buf[yy, xx])
            sum_cb += float(cb_buf[yy, xx])
            sum_cr += float(cr_buf[yy, xx])

    mean_y = float(sum_y / n)
    mean_cb = float(sum_cb / n)
    mean_cr = float(sum_cr / n)

    var_y = 0.0
    var_cb = 0.0
    var_cr = 0.0
    for yy in range(block_h):
        for xx in range(block_w):
            dy = float(y_buf[yy, xx]) - mean_y
            dcb = float(cb_buf[yy, xx]) - mean_cb
            dcr = float(cr_buf[yy, xx]) - mean_cr
            var_y += dy * dy
            var_cb += dcb * dcb
            var_cr += dcr * dcr
    var_y = float(var_y / n)
    var_cb = float(var_cb / n)
    var_cr = float(var_cr / n)

    # 勾配エネルギー（有効な近傍差分のみ）
    gx = 0.0
    if block_w >= 2:
        for yy in range(block_h):
            for xx in range(block_w - 1):
                d = float(y_buf[yy, xx + 1]) - float(y_buf[yy, xx])
                gx += d * d

    gy = 0.0
    if block_h >= 2:
        for yy in range(block_h - 1):
            for xx in range(block_w):
                d = float(y_buf[yy + 1, xx]) - float(y_buf[yy, xx])
                gy += d * d

    # エッジ密度（右・下の両方が存在する位置のみ）
    edge_count = 0
    if block_w >= 2 and block_h >= 2:
        # 小さいので Python ループで十分（可読性優先）
        for yy in range(block_h - 1):
            for xx in range(block_w - 1):
                c = float(y_buf[yy, xx])
                dx = float(y_buf[yy, xx + 1]) - c
                dy = float(y_buf[yy + 1, xx]) - c
                if abs(dx) + abs(dy) > EDGE_THRESHOLD_Y:
                    edge_count += 1
    edge_density = float(edge_count / n)

    # 高周波（Laplacian 風）エネルギー（内点のみ）
    hf_energy = 0.0
    if block_w >= 3 and block_h >= 3:
        for yy in range(1, block_h - 1):
            for xx in range(1, block_w - 1):
                c = float(y_buf[yy, xx])
                lap = 4.0 * c
                lap -= float(y_buf[yy, xx - 1])
                lap -= float(y_buf[yy, xx + 1])
                lap -= float(y_buf[yy - 1, xx])
                lap -= float(y_buf[yy + 1, xx])
                hf_energy += lap * lap
    hf_norm = float(hf_energy / (var_y + HF_NORM_EPS))

    # stats の格納（名前・順序は meta.json と一致させる）
    dst_row[offset + 0] = mean_y
    dst_row[offset + 1] = var_y
    dst_row[offset + 2] = var_cb
    dst_row[offset + 3] = var_cr
    dst_row[offset + 4] = gx
    dst_row[offset + 5] = gy
    dst_row[offset + 6] = edge_density
    dst_row[offset + 7] = hf_energy
    dst_row[offset + 8] = hf_norm

    # index.jsonl 用に必要なものを返す（最小限）
    return {
        "tile_origin": row["tile_origin"],
        "block_origin": row["block_origin"],
        "block_size": row["block_size"],
        "components": components,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_np_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    np.save(tmp, arr)
    # np.save は拡張子 .npy を自動付与するので、実ファイル名を合わせる
    saved = tmp
    if not saved.name.endswith(".npy"):
        saved = saved.with_suffix(saved.suffix + ".npy")
    os.replace(saved, path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="packed アーティファクトを ML 用 NumPy 配列に変換します。")
    parser.add_argument("--in-dir", required=True, type=Path, help="packed ディレクトリ（training.all.jsonl 等）")
    parser.add_argument("--out-dir", required=True, type=Path, help="出力先ディレクトリ")
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_RAW_PLUS_STATS_V1,
        help=f"特徴量セット（現在は {FEATURE_SET_RAW_PLUS_STATS_V1} のみ）",
    )
    parser.add_argument("--force", action="store_true", help="出力ファイルを上書きします。")
    parser.add_argument("--dry-run", action="store_true", help="書き込みを行わず、検証と予定出力のみ表示します。")
    parser.add_argument("--limit-images", type=int, default=None, help="先頭 N 画像のみ処理します。")
    parser.add_argument("--limit-blocks", type=int, default=None, help="先頭 N ブロックのみ処理します。")
    parser.add_argument(
        "--strict",
        type=_parse_bool,
        default=True,
        help="true: 不整合があれば即失敗 / false: 軽微な欠損は許容（default: true）",
    )
    args = parser.parse_args(argv)

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    feature_set: str = args.feature_set
    force: bool = args.force
    dry_run: bool = args.dry_run
    limit_images: Optional[int] = args.limit_images
    limit_blocks: Optional[int] = args.limit_blocks
    strict: bool = args.strict

    if limit_images is not None and limit_images < 0:
        raise ValueError("--limit-images は 0 以上で指定してください。")
    if limit_blocks is not None and limit_blocks < 0:
        raise ValueError("--limit-blocks は 0 以上で指定してください。")

    paths = _validate_inputs(in_dir)
    _check_overwrite(out_dir, force=force, dry_run=dry_run)

    entries = _load_packed_index(paths["index"], limit_images=limit_images)
    total_lines = sum(e.training_line_count for e in entries)
    total_records = sum(e.label_record_count for e in entries)
    if total_lines != total_records:
        raise ValueError(f"合計件数が不一致です: lines={total_lines} records={total_records}")

    if limit_blocks is None:
        n_blocks = total_lines
    else:
        n_blocks = min(total_lines, limit_blocks)

    d = _feature_dim(feature_set)
    feature_names = _feature_names_raw_plus_stats_v1()
    if len(feature_names) != d:
        raise RuntimeError("内部エラー: feature_names と feature_dim が一致しません")

    # labels.all.bin のサイズ検証（少なくとも total_records 分は必要）
    label_size = paths["labels"].stat().st_size
    expected_min_size = (entries[0].label_record_offset + n_blocks) * LABEL_RECORD_SIZE
    if label_size < expected_min_size:
        raise ValueError(
            "labels.all.bin のサイズが不足しています: "
            f"size={label_size} expected_at_least={expected_min_size}"
        )
    if strict and label_size % LABEL_RECORD_SIZE != 0:
        raise ValueError(f"labels.all.bin のサイズが {LABEL_RECORD_SIZE} の倍数ではありません: {label_size}")

    print(f"feature_set={feature_set} N={n_blocks} D={d} images={len(entries)}")
    if dry_run:
        print("dry-run: 書き込みは行いません。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # 配列は全体を保持して良い前提（後で mean/std を計算する）
    x = np.empty((n_blocks, d), dtype=np.float32)
    y = np.empty((n_blocks,), dtype=np.int64)

    # index.jsonl は逐次出力する（ブロック単位の追跡情報）
    index_tmp = out_dir / f"index.jsonl.tmp.{os.getpid()}"
    work = _WorkBuffers(
        y=np.empty((BLOCK_H, BLOCK_W), dtype=np.float32),
        cb=np.empty((BLOCK_H, BLOCK_W), dtype=np.float32),
        cr=np.empty((BLOCK_H, BLOCK_W), dtype=np.float32),
    )
    with (
        paths["training"].open("r", encoding="utf-8") as training_fp,
        paths["labels"].open("rb") as label_fp,
        index_tmp.open("w", encoding="utf-8", newline="\n") as out_index_fp,
    ):
        # 先頭オフセット（通常 0 だが念のため）
        start_line = entries[0].training_line_offset
        start_rec = entries[0].label_record_offset

        if start_rec != 0:
            label_fp.seek(start_rec * LABEL_RECORD_SIZE, os.SEEK_SET)
        if start_line != 0:
            for _ in tqdm(range(start_line), desc="skip training lines", unit="line"):
                if not training_fp.readline():
                    raise ValueError("training.all.jsonl の先頭スキップ中に EOF になりました")

        # relpath の割り当て（global line index を見て画像エントリを進める）
        img_i = 0
        img_start = entries[img_i].training_line_offset
        img_end = img_start + entries[img_i].training_line_count
        relpath = entries[img_i].relpath

        global_line_index = start_line
        for i in tqdm(range(n_blocks), desc="blocks", unit="blk"):
            # 画像境界を跨いだら進める
            while global_line_index >= img_end:
                img_i += 1
                if img_i >= len(entries):
                    raise ValueError("packed index の範囲を超えて読み込もうとしました")
                img_start = entries[img_i].training_line_offset
                img_end = img_start + entries[img_i].training_line_count
                relpath = entries[img_i].relpath

            line = training_fp.readline()
            if not line:
                raise ValueError("training.all.jsonl の途中で EOF になりました")
            row = json.loads(line)

            # label
            y[i] = _read_label_best_reorder(label_fp)

            # feature
            if feature_set == FEATURE_SET_RAW_PLUS_STATS_V1:
                idx_fields = _fill_features_raw_plus_stats_v1(x[i], row, work=work)
            else:
                raise ValueError(f"未対応の --feature-set: {feature_set}")

            # ブロック単位 index（追跡用、最小限）
            out_index_fp.write(
                json.dumps(
                    {
                        "relpath": relpath,
                        "global_block_index": i,
                        "tile_origin": idx_fields["tile_origin"],
                        "block_origin": idx_fields["block_origin"],
                        "block_size": idx_fields["block_size"],
                        "components": idx_fields["components"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            global_line_index += 1

        # 余計なラベルを読んでいないことは i で担保（label_fp はちょうど n_blocks ぶん進む）

    os.replace(index_tmp, out_dir / "index.jsonl")

    # 正規化統計量（全ブロックの per-dim mean/std）
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)

    # 保存（np.save は自動で .npy を付けるケースがあるため原子的に置換）
    _atomic_np_save(out_dir / "features.npy", x)
    _atomic_np_save(out_dir / "labels_reorder.npy", y)
    _atomic_np_save(out_dir / "feature_mean.npy", mean)
    _atomic_np_save(out_dir / "feature_std.npy", std)

    meta = {
        "schema": 1,
        "created_at": _now_iso(),
        "in_dir": str(in_dir),
        "feature_set": feature_set,
        "total_blocks": int(n_blocks),
        "feature_dim": int(d),
        "x_dtype": "float32",
        "y_dtype": "int64",
        "label_name": "reorder",
        "reorder_classes": 8,
        "feature_names": feature_names,
        "feature_layout": {
            "raw_rgb": {"dims": RAW_RGB_DIMS, "order": "y->x->rgb", "range": "[0,1]"},
            "extra": {"names": ["block_w", "block_h"]},
            "stats": {
                "names": [
                    "mean_Y",
                    "var_Y",
                    "var_Cb",
                    "var_Cr",
                    "Gx_Y",
                    "Gy_Y",
                    "edge_density_Y",
                    "hf_energy_Y",
                    "hf_norm_Y",
                ]
            },
        },
        "color_transform": {
            "note": "R/G/B は raw RGB を [0,1] に正規化した値",
            "Y": "0.299*R + 0.587*G + 0.114*B",
            "Cb": "B - Y",
            "Cr": "R - Y",
        },
        "edge_density": {
            "threshold_Y": EDGE_THRESHOLD_Y,
            "criterion": "|dx| + |dy| > threshold_Y",
            "neighbors": "right and bottom (both exist)",
            "denominator": "w*h (actual block area)",
        },
        "hf_energy": {
            "laplacian": "4*Y[x,y] - Y[x-1,y] - Y[x+1,y] - Y[x,y-1] - Y[x,y+1] (interior only)",
            "hf_norm": f"hf_energy / (var_Y + {HF_NORM_EPS})",
        },
        "normalization": {
            "kind": "zscore",
            "mean_file": "feature_mean.npy",
            "std_file": "feature_std.npy",
            "computed_over": "full dataset (no split)",
        },
        "inputs": {
            "training_all_jsonl": "training.all.jsonl",
            "labels_all_bin": "labels.all.bin",
            "packed_index_jsonl": "index.jsonl",
        },
    }
    _atomic_write_text(out_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
