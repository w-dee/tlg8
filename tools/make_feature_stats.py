#!/usr/bin/env python3
"""JSONL から特徴量統計を構築するユーティリティ。"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from multitask_model import HEAD_SPECS, conditioned_extra_dim

MAGIC = b"FSC8"
VERSION = 1
HEADER = struct.Struct("<4sIIQ")


def parse_head_list(raw: str) -> List[str]:
    """カンマ区切りのヘッド名リストを正規化する。"""

    if not raw:
        return []
    heads = [item.strip() for item in raw.split(",") if item.strip()]
    return heads


def build_onehot(index: int, size: int) -> List[float]:
    """整数 ID からワンホットベクトルを生成する。"""

    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def iter_feature_vectors(
    path: Path,
    base_key: str,
    cond_heads: Sequence[str],
    encoding: str,
) -> Iterable[np.ndarray]:
    """JSONL から条件付き特徴量ベクトルを逐次生成する。"""

    encoding_norm = (encoding or "onehot").lower()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            base = record.get(base_key)
            if base is None:
                raise KeyError(f"レコードに {base_key} が存在しません")
            feature = list(map(float, base))
            labels = record.get("labels", {})
            for head in cond_heads:
                spec = HEAD_SPECS.get(head)
                if spec is None:
                    raise KeyError(f"未知のヘッド名です: {head}")
                best = labels.get(head, {}).get("best", -1)
                idx = int(best)
                if encoding_norm == "onehot":
                    feature.extend(build_onehot(idx, spec))
                elif encoding_norm == "id":
                    feature.append(float(idx))
                else:
                    raise ValueError(f"未対応のエンコーディングです: {encoding}")
            yield np.asarray(feature, dtype=np.float64)


def accumulate_stats(vectors: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """ベクトル列から総和と自乗和、件数を算出する。"""

    total = 0
    sum_vec: np.ndarray | None = None
    sumsq_vec: np.ndarray | None = None
    for vec in vectors:
        if sum_vec is None:
            sum_vec = np.zeros_like(vec)
            sumsq_vec = np.zeros_like(vec)
        elif vec.shape != sum_vec.shape:
            raise ValueError("特徴量次元が行ごとに一致しません")
        sum_vec += vec
        sumsq_vec += vec * vec
        total += 1
    if sum_vec is None or sumsq_vec is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0
    return sum_vec, sumsq_vec, total


def write_stats(path: Path, sums: np.ndarray, sumsq: np.ndarray, count: int) -> None:
    """FSC8 形式で統計量を出力する。"""

    dimension = int(sums.shape[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as fp:
        fp.write(HEADER.pack(MAGIC, VERSION, dimension, int(count)))
        if dimension:
            fp.write(sums.astype("<f8", copy=False).tobytes())
            fp.write(sumsq.astype("<f8", copy=False).tobytes())
    tmp.replace(path)


def main() -> int:
    """コマンドライン引数を処理して統計量を生成する。"""

    parser = argparse.ArgumentParser(description="特徴量統計を生成する補助スクリプト")
    parser.add_argument("input", type=Path, help="入力 JSONL ファイル")
    parser.add_argument("output", type=Path, help="出力先 stats.bin パス")
    parser.add_argument(
        "--condition-heads",
        type=str,
        default="reorder,interleave",
        help="条件付き特徴量に利用するヘッド名 (カンマ区切り)",
    )
    parser.add_argument(
        "--condition-encoding",
        type=str,
        default="onehot",
        choices=["onehot", "id"],
        help="条件付きヘッドのエンコーディング方式",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default="features",
        help="JSON レコード内で特徴量が格納されているキー名",
    )
    args = parser.parse_args()

    cond_heads = parse_head_list(args.condition_heads)
    encoding_norm = args.condition_encoding.lower()
    vectors = iter_feature_vectors(args.input, args.feature_key, cond_heads, encoding_norm)
    sums, sumsq, count = accumulate_stats(vectors)
    write_stats(args.output, sums, sumsq, count)

    cond_extra = conditioned_extra_dim(cond_heads, encoding_norm)
    final_dim = int(sums.shape[0])
    print(
        f"wrote {args.output} (dim={final_dim}, count={count}, condition_heads={cond_heads}, encoding={encoding_norm}, cond_extra={cond_extra})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
