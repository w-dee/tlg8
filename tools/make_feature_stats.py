#!/usr/bin/env python3
"""JSONL/NPY から特徴量統計を構築するユーティリティ。"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from io_utils import find_features_path, sha256_file
from multitask_model import HEAD_SPECS, conditioned_extra_dim

MAGIC = b"FSC8"
VERSION = 1
HEADER = struct.Struct("<4sIIQ")


class FeatureValidationError(ValueError):
    """特徴量検証失敗時に詳細情報を保持する例外。"""

    def __init__(self, rows: List[int], samples: List[np.ndarray]) -> None:
        super().__init__("特徴量行列に NaN/Inf が含まれています")
        self.rows = rows
        self.samples = samples


def _parse_bool(raw: str) -> bool:
    """文字列を真偽値に変換する。"""

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"真偽値に変換できない文字列です: {raw}")


def load_feature_matrix(path: Path, *, mmap: bool = True) -> np.ndarray:
    """NPY 形式の特徴量行列を読み込む。"""

    mode = "r" if mmap else None
    try:
        array = np.load(path, mmap_mode=mode, allow_pickle=False)
    except OSError as exc:
        raise ValueError(f"{path} の読み込みに失敗しました: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"特徴量行列は 2 次元である必要があります (shape={array.shape})")
    if array.shape[1] <= 0:
        raise ValueError("特徴量次元が 0 以下です")
    return array


def compute_feature_mean_std(
    array: np.ndarray,
    *,
    batch_size: int = 65536,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """特徴量行列から平均と標準偏差を逐次計算する。"""

    if array.ndim != 2:
        raise ValueError("特徴量行列は 2 次元である必要があります")
    total = int(array.shape[0])
    dim = int(array.shape[1])
    if total == 0:
        return np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32), 0
    sum_vec = np.zeros((dim,), dtype=np.float64)
    sumsq_vec = np.zeros((dim,), dtype=np.float64)
    invalid_rows: List[int] = []
    invalid_samples: List[np.ndarray] = []
    count = 0
    step = max(1, int(batch_size))
    for start in range(0, total, step):
        end = min(total, start + step)
        batch = np.asarray(array[start:end], dtype=np.float64)
        if batch.size == 0:
            continue
        row_invalid = ~np.isfinite(batch).all(axis=1)
        if np.any(row_invalid) and len(invalid_rows) < 10:
            for offset in np.nonzero(row_invalid)[0]:
                if len(invalid_rows) >= 10:
                    break
                absolute = start + int(offset)
                invalid_rows.append(absolute)
                invalid_samples.append(batch[int(offset)].astype(np.float32, copy=False))
        sum_vec += batch.sum(axis=0)
        sumsq_vec += np.square(batch).sum(axis=0)
        count += batch.shape[0]
    if invalid_rows:
        raise FeatureValidationError(invalid_rows, invalid_samples)
    if count == 0:
        return np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32), 0
    mean = sum_vec / float(count)
    var = sumsq_vec / float(count) - mean * mean
    var[var < 0.0] = 0.0
    std = np.sqrt(var)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), count


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
    parser.add_argument("input", type=Path, nargs="?", help="入力 JSONL ファイル")
    parser.add_argument("output", type=Path, nargs="?", help="出力先 stats.bin パス")
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
    parser.add_argument(
        "--features-npy",
        type=str,
        default=None,
        help="事前計算済み特徴量行列 (.npy) のパス",
    )
    parser.add_argument(
        "--features-version",
        type=int,
        default=0,
        help="優先的に選択する特徴量バージョン (0 で自動)",
    )
    parser.add_argument(
        "--features-autodetect",
        type=_parse_bool,
        default=True,
        help="--features-npy 未指定時に既定パスから自動検出するか",
    )
    parser.add_argument(
        "--scaler-out",
        type=Path,
        default=Path(".cache/ranker.scaler.npz"),
        help="再計算したスケーラー (.npz) の出力先",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="特徴量 NPY から平均・標準偏差を再計算して --scaler-out に保存",
    )
    args = parser.parse_args()

    if not args.recompute and (args.input is None or args.output is None):
        parser.error("--recompute 未指定の場合は input と output を与えてください")

    if args.input is not None and args.output is not None:
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

    if args.recompute:
        try:
            selected = find_features_path(args.features_npy, args.features_version, bool(args.features_autodetect))
        except FileNotFoundError as exc:
            print(f"エラー: 特徴量行列が見つかりません: {exc}", file=sys.stderr)
            return 1
        features_path = Path(selected)
        try:
            matrix = load_feature_matrix(features_path)
            mean, std, count = compute_feature_mean_std(matrix)
        except FeatureValidationError as exc:
            print("エラー: 特徴量に無効値が含まれています。先頭 10 行を表示します:", file=sys.stderr)
            for row, sample in zip(exc.rows, exc.samples):
                preview = np.asarray(sample, dtype=np.float32)
                print(f"  row={row}: {preview.tolist()}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"エラー: 特徴量行列の検証に失敗しました: {exc}", file=sys.stderr)
            return 1
        checksum = sha256_file(features_path)
        scaler_out = Path(args.scaler_out)
        scaler_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scaler_out,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            dim=int(mean.shape[0]),
            count=int(count),
            path=str(features_path),
            checksum=checksum,
        )
        print(
            f"recomputed scaler -> {scaler_out} (path={features_path}, dim={mean.shape[0]}, count={count}, checksum={checksum[:8]}...)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
