#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tlg_ml_utils import (
    RAW_RGB_DIMS,
    build_raw_feature_names,
    collect_numeric_schema,
    ensure_dir,
    extract_numeric_row,
    extract_pixels_rgb,
    join_filter,
    load_json,
    now_iso,
    read_label_record,
    save_json,
)

IGNORE_KEYS = {
    "pixels",
    "best",
    "second",
    "entropy",
    "image",
    "image_size",
    "tile_origin",
    "block_origin",
    "block_size",
    "components",
}


@dataclass
class DatasetCache:
    pixels: np.ndarray
    raw_numeric: np.ndarray
    raw_names: list[str]
    labels_best: np.ndarray
    labels_second: np.ndarray
    bits: np.ndarray
    indices: np.ndarray
    train_indices: np.ndarray
    valid_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset cache for TLG8 ML pipeline")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory with training.all.jsonl, labels.all.bin")
    parser.add_argument("--run-id", required=True, type=str, help="Run ID under ml/runs")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cache exists")
    return parser.parse_args()


def _iter_rows(training_path: Path, label_path: Path) -> Iterable[tuple[int, dict[str, Any], dict[str, int]]]:
    with training_path.open("r", encoding="utf-8") as tfp, label_path.open("rb") as lfp:
        for idx, line in enumerate(tfp):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = read_label_record(lfp)
            yield idx, row, labels


def _ensure_label_eof(label_path: Path, line_count: int) -> None:
    with label_path.open("rb") as fp:
        fp.seek(0, 2)
        size = fp.tell()
    if size % 128 != 0:
        raise ValueError(f"labels.all.bin size is not multiple of 128: {size}")
    if size // 128 < line_count:
        raise ValueError("labels.all.bin has fewer records than training.all.jsonl")


def _is_eligible(row: dict[str, Any], labels: dict[str, int]) -> tuple[bool, str]:
    block_size = row.get("block_size")
    if block_size != [8, 8]:
        return False, "block_size"
    components = int(row.get("components", 0))
    if components not in (3, 4):
        return False, "components"
    if row.get("best") is None or row.get("second") is None:
        return False, "best_or_second_null"
    if labels["second_predictor"] == -1:
        return False, "second_missing"
    best_tuple = (
        labels["best_predictor"],
        labels["best_filter_perm"],
        labels["best_filter_primary"],
        labels["best_filter_secondary"],
        labels["best_reorder"],
        labels["best_interleave"],
    )
    second_tuple = (
        labels["second_predictor"],
        labels["second_filter_perm"],
        labels["second_filter_primary"],
        labels["second_filter_secondary"],
        labels["second_reorder"],
        labels["second_interleave"],
    )
    if best_tuple == second_tuple:
        return False, "best_equals_second"
    return True, ""


def _build_split(
    run_dir: Path,
    seed: int,
    count: int,
    *,
    train_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    split_path = run_dir / "splits" / "split.json"
    if split_path.exists():
        payload = load_json(split_path)
        return (
            np.asarray(payload["train_indices"], dtype=np.int64),
            np.asarray(payload["valid_indices"], dtype=np.int64),
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    train_count = int(count * train_ratio)
    train_indices = indices[:train_count]
    valid_indices = indices[train_count:]
    ensure_dir(split_path.parent)
    save_json(
        split_path,
        {
            "seed": seed,
            "train_ratio": train_ratio,
            "total": count,
            "train_indices": train_indices.tolist(),
            "valid_indices": valid_indices.tolist(),
        },
    )
    return train_indices, valid_indices


def load_dataset_cache(run_dir: Path) -> DatasetCache | None:
    dataset_dir = run_dir / "dataset"
    pixels_path = dataset_dir / "pixels.npy"
    raw_numeric_path = dataset_dir / "raw_numeric.npy"
    raw_names_path = dataset_dir / "raw_numeric_names.json"
    labels_best_path = dataset_dir / "labels_best.npy"
    labels_second_path = dataset_dir / "labels_second.npy"
    bits_path = dataset_dir / "bits.filtered.npy"
    index_path = dataset_dir / "indices.npy"
    split_path = run_dir / "splits" / "split.json"
    if not (
        pixels_path.exists()
        and raw_numeric_path.exists()
        and raw_names_path.exists()
        and labels_best_path.exists()
        and labels_second_path.exists()
        and bits_path.exists()
        and index_path.exists()
        and split_path.exists()
    ):
        return None
    pixels = np.load(pixels_path)
    raw_numeric = np.load(raw_numeric_path)
    labels_best = np.load(labels_best_path)
    labels_second = np.load(labels_second_path)
    bits = np.load(bits_path)
    indices = np.load(index_path)
    raw_names = load_json(raw_names_path)["raw_names"]
    split = load_json(split_path)
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    valid_indices = np.asarray(split["valid_indices"], dtype=np.int64)
    return DatasetCache(
        pixels=pixels,
        raw_numeric=raw_numeric,
        raw_names=raw_names,
        labels_best=labels_best,
        labels_second=labels_second,
        bits=bits,
        indices=indices,
        train_indices=train_indices,
        valid_indices=valid_indices,
    )


def build_dataset(run_dir: Path, in_dir: Path, seed: int, *, rebuild: bool) -> DatasetCache:
    cached = load_dataset_cache(run_dir)
    if cached and not rebuild:
        return cached

    training_path = in_dir / "training.all.jsonl"
    label_path = in_dir / "labels.all.bin"
    bits_path = run_dir / "dataset" / "bits.all.npy"
    if not training_path.is_file():
        raise FileNotFoundError(f"training.all.jsonl not found: {training_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"labels.all.bin not found: {label_path}")
    if not bits_path.is_file():
        raise FileNotFoundError(f"bits.all.npy not found: {bits_path}")

    bits_all = np.load(bits_path)
    schema: dict[str, int] = {}
    invalid_keys: set[str] = set()
    total_rows = 0
    kept_rows = 0
    drop_reasons: dict[str, int] = {}

    for idx, row, labels in _iter_rows(training_path, label_path):
        total_rows += 1
        if idx >= bits_all.shape[0]:
            raise ValueError("bits.all.npy is shorter than training.all.jsonl")
        eligible, reason = _is_eligible(row, labels)
        if not eligible:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        kept_rows += 1
        collect_numeric_schema(row, schema, invalid_keys, IGNORE_KEYS)

    _ensure_label_eof(label_path, total_rows)
    if kept_rows == 0:
        raise ValueError("no eligible rows found after filtering")

    raw_names = build_raw_feature_names(schema)
    raw_dim = len(raw_names)

    pixels = np.empty((kept_rows, RAW_RGB_DIMS), dtype=np.float32)
    raw_numeric = np.empty((kept_rows, raw_dim), dtype=np.float32)
    labels_best = np.empty((kept_rows, 6), dtype=np.int16)
    labels_second = np.empty((kept_rows, 6), dtype=np.int16)
    bits = np.empty((kept_rows, 2), dtype=np.uint64)
    indices = np.empty((kept_rows,), dtype=np.int64)

    write_idx = 0
    for idx, row, labels in _iter_rows(training_path, label_path):
        if idx >= bits_all.shape[0]:
            raise ValueError("bits.all.npy is shorter than training.all.jsonl")
        eligible, _ = _is_eligible(row, labels)
        if not eligible:
            continue
        pixels[write_idx] = extract_pixels_rgb(row)
        raw_numeric[write_idx] = extract_numeric_row(row, schema)
        labels_best[write_idx] = np.asarray(
            [
                labels["best_predictor"],
                labels["best_filter_perm"],
                labels["best_filter_primary"],
                labels["best_filter_secondary"],
                labels["best_reorder"],
                labels["best_interleave"],
            ],
            dtype=np.int16,
        )
        labels_second[write_idx] = np.asarray(
            [
                labels["second_predictor"],
                labels["second_filter_perm"],
                labels["second_filter_primary"],
                labels["second_filter_secondary"],
                labels["second_reorder"],
                labels["second_interleave"],
            ],
            dtype=np.int16,
        )
        bits[write_idx] = bits_all[idx]
        indices[write_idx] = idx
        write_idx += 1

    train_indices, valid_indices = _build_split(run_dir, seed, kept_rows)

    dataset_dir = run_dir / "dataset"
    ensure_dir(dataset_dir)
    np.save(dataset_dir / "pixels.npy", pixels)
    np.save(dataset_dir / "raw_numeric.npy", raw_numeric)
    save_json(dataset_dir / "raw_numeric_names.json", {"raw_names": raw_names})
    np.save(dataset_dir / "labels_best.npy", labels_best)
    np.save(dataset_dir / "labels_second.npy", labels_second)
    np.save(dataset_dir / "bits.filtered.npy", bits)
    np.save(dataset_dir / "indices.npy", indices)

    meta = {
        "created_at": now_iso(),
        "training_all_jsonl": str(training_path),
        "labels_all_bin": str(label_path),
        "bits_all_npy": str(bits_path),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "drop_reasons": drop_reasons,
        "raw_numeric_dim": raw_dim,
    }
    save_json(dataset_dir / "dataset_meta.json", meta)

    return DatasetCache(
        pixels=pixels,
        raw_numeric=raw_numeric,
        raw_names=raw_names,
        labels_best=labels_best,
        labels_second=labels_second,
        bits=bits,
        indices=indices,
        train_indices=train_indices,
        valid_indices=valid_indices,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_root / args.run_id
    ensure_dir(run_dir)
    build_dataset(run_dir, args.in_dir, args.seed, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
