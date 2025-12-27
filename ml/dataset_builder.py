#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
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

try:
    import orjson

    _json_loads = orjson.loads
except Exception:
    def _json_loads(data: bytes) -> Any:  # type: ignore[misc]
        return json.loads(data.decode("utf-8"))

IN_DIR_CACHE_DIRNAME = ".tlg_ml_cache"
IN_DIR_CACHE_VERSION = 1

LABEL_RECORD_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u2"),
        ("reserved", "<u2"),
        ("labels", "<i2", (12,)),
        ("checksum", "<u4"),
        ("padding", "S92"),
    ],
    align=False,
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
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Cap eligible blocks via reservoir sampling (deterministic by --seed); useful for huge datasets",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cache exists")
    return parser.parse_args()


def _loads_line(line: bytes) -> dict[str, Any]:
    if line.endswith(b"\n"):
        line = line[:-1]
    obj = _json_loads(line)
    if isinstance(obj, dict):
        return obj
    raise ValueError("training.all.jsonl line did not decode into an object")


def _iter_rows(training_path: Path, label_path: Path) -> Iterable[tuple[int, dict[str, Any], dict[str, int]]]:
    # Binary mode enables accurate byte offsets and lets orjson parse bytes directly.
    with training_path.open("rb", buffering=8 * 1024 * 1024) as tfp, label_path.open("rb") as lfp:
        for idx, line in enumerate(tfp):
            if not line or line == b"\n":
                raise ValueError(f"blank line in training.all.jsonl at line {idx + 1} (alignment would break)")
            row = _loads_line(line)
            labels = read_label_record(lfp)
            yield idx, row, labels


def _log_progress(phase: str, idx: int, total: int, start_time: float, every: int) -> None:
    if idx == 0 or (idx + 1) % every != 0:
        return
    elapsed = time.time() - start_time
    ratio = (idx + 1) / total if total > 0 else 1.0
    print(
        f"[{phase}] {idx + 1:,}/{total:,} ({ratio * 100:.1f}%) elapsed {elapsed:.1f}s",
        flush=True,
    )


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


def _labels_record_count(label_path: Path) -> int:
    size = label_path.stat().st_size
    if size % 128 != 0:
        raise ValueError(f"labels.all.bin size is not multiple of 128: {size}")
    return size // 128


def _bits_sidecar_path(run_dir: Path) -> Path:
    dataset_dir = run_dir / "dataset"
    ensure_dir(dataset_dir)
    return dataset_dir / "bits.all.npy"


def _stat_fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _in_dir_cache_dir(in_dir: Path) -> Path:
    return in_dir / IN_DIR_CACHE_DIRNAME / f"v{IN_DIR_CACHE_VERSION}"


def _in_dir_cache_paths(in_dir: Path) -> dict[str, Path]:
    cache_dir = _in_dir_cache_dir(in_dir)
    return {
        "dir": cache_dir,
        "meta": cache_dir / "meta.json",
        "line_offsets": cache_dir / "training.line_offsets.u64.npy",
        "eligible_mask": cache_dir / "eligible_mask.u8.npy",
        "schema": cache_dir / "schema.json",
        "raw_names": cache_dir / "raw_numeric_names.json",
        "bits_all": cache_dir / "bits.all.npy",
    }


def _load_in_dir_cache(
    in_dir: Path,
    training_path: Path,
    label_path: Path,
    total_rows_expected: int,
) -> tuple[dict[str, int] | None, list[str] | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    paths = _in_dir_cache_paths(in_dir)
    if not (
        paths["meta"].exists()
        and paths["line_offsets"].exists()
        and paths["eligible_mask"].exists()
        and paths["schema"].exists()
        and paths["raw_names"].exists()
        and paths["bits_all"].exists()
    ):
        return None, None, None, None, None

    meta = load_json(paths["meta"])
    if meta.get("version") != IN_DIR_CACHE_VERSION:
        return None, None, None, None, None
    if meta.get("rows_expected") != int(total_rows_expected):
        return None, None, None, None, None
    if meta.get("training_all_jsonl") != _stat_fingerprint(training_path):
        return None, None, None, None, None
    if meta.get("labels_all_bin") != _stat_fingerprint(label_path):
        return None, None, None, None, None

    schema = load_json(paths["schema"])["schema"]
    raw_names = load_json(paths["raw_names"])["raw_names"]
    line_offsets = np.load(paths["line_offsets"], mmap_mode="r")
    eligible_mask = np.load(paths["eligible_mask"], mmap_mode="r")
    bits_all = np.load(paths["bits_all"], mmap_mode="r")
    return schema, raw_names, line_offsets, eligible_mask, bits_all


def _build_in_dir_cache(
    in_dir: Path,
    training_path: Path,
    label_path: Path,
    total_rows_expected: int,
) -> tuple[dict[str, int], list[str], np.ndarray, np.ndarray, np.ndarray]:
    paths = _in_dir_cache_paths(in_dir)
    ensure_dir(paths["dir"])

    tmp_dir = paths["dir"] / f"tmp_{os.getpid()}_{int(time.time())}"
    ensure_dir(tmp_dir)
    tmp_paths = {
        "meta": tmp_dir / "meta.json",
        "line_offsets": tmp_dir / "training.line_offsets.u64.npy",
        "eligible_mask": tmp_dir / "eligible_mask.u8.npy",
        "schema": tmp_dir / "schema.json",
        "raw_names": tmp_dir / "raw_numeric_names.json",
        "bits_all": tmp_dir / "bits.all.npy",
    }

    line_offsets = np.lib.format.open_memmap(
        tmp_paths["line_offsets"],
        mode="w+",
        dtype=np.uint64,
        shape=(total_rows_expected + 1,),
    )
    eligible_mask = np.lib.format.open_memmap(
        tmp_paths["eligible_mask"],
        mode="w+",
        dtype=np.uint8,
        shape=(total_rows_expected,),
    )
    bits_all = np.lib.format.open_memmap(
        tmp_paths["bits_all"],
        mode="w+",
        dtype=np.uint64,
        shape=(total_rows_expected, 2),
    )

    schema: dict[str, int] = {}
    invalid_keys: set[str] = set()
    drop_reasons: dict[str, int] = {}
    eligible_rows = 0

    scan_start = time.time()
    with training_path.open("rb", buffering=8 * 1024 * 1024) as tfp, label_path.open("rb") as lfp:
        for idx in range(total_rows_expected):
            offset = tfp.tell()
            line = tfp.readline()
            if not line or line == b"\n":
                raise ValueError(f"training.all.jsonl ended early or has blank line at {idx + 1}")
            line_offsets[idx] = np.uint64(offset)
            row = _loads_line(line)
            labels = read_label_record(lfp)

            best = row.get("best")
            second = row.get("second")
            max_u64 = np.iinfo(np.uint64).max
            bits_best = int(best.get("bits", max_u64)) if isinstance(best, dict) else max_u64
            bits_second = int(second.get("bits", max_u64)) if isinstance(second, dict) else max_u64
            bits_all[idx, 0] = np.uint64(bits_best)
            bits_all[idx, 1] = np.uint64(bits_second)

            eligible, reason = _is_eligible(row, labels)
            if not eligible:
                eligible_mask[idx] = np.uint8(0)
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                eligible_mask[idx] = np.uint8(1)
                eligible_rows += 1
                collect_numeric_schema(row, schema, invalid_keys, IGNORE_KEYS)

            _log_progress("scan", idx, total_rows_expected, scan_start, every=200_000)

        line_offsets[total_rows_expected] = np.uint64(tfp.tell())

    line_offsets.flush()
    eligible_mask.flush()
    bits_all.flush()

    raw_names = build_raw_feature_names(schema)
    save_json(tmp_paths["schema"], {"schema": schema})
    save_json(tmp_paths["raw_names"], {"raw_names": raw_names})
    save_json(
        tmp_paths["meta"],
        {
            "version": IN_DIR_CACHE_VERSION,
            "created_at": now_iso(),
            "rows_expected": int(total_rows_expected),
            "training_all_jsonl": _stat_fingerprint(training_path),
            "labels_all_bin": _stat_fingerprint(label_path),
            "eligible_rows": int(eligible_rows),
            "drop_reasons": drop_reasons,
        },
    )

    # Atomic-ish publish: rename tmp files into the cache directory.
    for key, dest in paths.items():
        if key in ("dir",):
            continue
        src = tmp_paths[key]
        os.replace(src, dest)
    # Cleanup tmp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # Reload as mmap for consumers
    schema_loaded = load_json(paths["schema"])["schema"]
    raw_names_loaded = load_json(paths["raw_names"])["raw_names"]
    line_offsets_loaded = np.load(paths["line_offsets"], mmap_mode="r")
    eligible_mask_loaded = np.load(paths["eligible_mask"], mmap_mode="r")
    bits_all_loaded = np.load(paths["bits_all"], mmap_mode="r")
    return schema_loaded, raw_names_loaded, line_offsets_loaded, eligible_mask_loaded, bits_all_loaded


def _ensure_run_bits_sidecar(run_dir: Path, in_dir: Path, bits_all: np.ndarray) -> Path:
    out_path = _bits_sidecar_path(run_dir)
    if out_path.exists():
        return out_path
    ensure_dir(out_path.parent)
    # Prefer hardlink to avoid extra disk IO; fall back to copy.
    cache_bits = _in_dir_cache_paths(in_dir)["bits_all"]
    if cache_bits.exists():
        try:
            os.link(cache_bits, out_path)
            return out_path
        except OSError:
            shutil.copyfile(cache_bits, out_path)
            return out_path
    np.save(out_path, np.asarray(bits_all, dtype=np.uint64))
    return out_path


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


def build_dataset(
    run_dir: Path,
    in_dir: Path,
    seed: int,
    *,
    rebuild: bool,
    max_blocks: int | None = None,
) -> DatasetCache:
    cached = load_dataset_cache(run_dir)
    if cached and not rebuild:
        return cached

    training_path = in_dir / "training.all.jsonl"
    label_path = in_dir / "labels.all.bin"
    if not training_path.is_file():
        raise FileNotFoundError(f"training.all.jsonl not found: {training_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"labels.all.bin not found: {label_path}")

    total_rows_expected = _labels_record_count(label_path)
    paths = _in_dir_cache_paths(in_dir)
    schema, raw_names, line_offsets, eligible_mask, bits_all = _load_in_dir_cache(
        in_dir, training_path, label_path, total_rows_expected
    )
    if schema is None or raw_names is None or line_offsets is None or eligible_mask is None or bits_all is None:
        print(f"[scan] building in-dir cache at {paths['dir']}", flush=True)
        schema, raw_names, line_offsets, eligible_mask, bits_all = _build_in_dir_cache(
            in_dir, training_path, label_path, total_rows_expected
        )

    bits_path = _ensure_run_bits_sidecar(run_dir, in_dir, bits_all)

    meta = load_json(paths["meta"]) if paths["meta"].exists() else {}
    drop_reasons: dict[str, int] = meta.get("drop_reasons", {})

    eligible_indices = np.flatnonzero(eligible_mask).astype(np.int64, copy=False)
    eligible_rows = int(eligible_indices.shape[0])
    if eligible_rows == 0:
        raise ValueError("no eligible rows found after filtering")

    if max_blocks is not None:
        if max_blocks <= 0:
            raise ValueError("--max-blocks must be positive")
        if eligible_rows > max_blocks:
            rng = np.random.default_rng(seed)
            selected = rng.choice(eligible_indices, size=int(max_blocks), replace=False)
        else:
            selected = eligible_indices
    else:
        selected = eligible_indices

    selected = np.sort(selected)
    kept_rows = int(selected.shape[0])
    if kept_rows == 0:
        raise ValueError("no eligible rows selected")

    raw_dim = len(raw_names)
    pixels = np.empty((kept_rows, RAW_RGB_DIMS), dtype=np.float32)
    raw_numeric = np.empty((kept_rows, raw_dim), dtype=np.float32)

    labels_mm = np.memmap(label_path, dtype=LABEL_RECORD_DTYPE, mode="r", shape=(total_rows_expected,))
    labels_all = labels_mm["labels"]
    label_sel = np.asarray(labels_all[selected], dtype=np.int16)
    labels_best = label_sel[:, :6]
    labels_second = label_sel[:, 6:]

    bits = np.asarray(bits_all[selected], dtype=np.uint64)
    indices = np.asarray(selected, dtype=np.int64)

    build_start = time.time()
    with training_path.open("rb", buffering=8 * 1024 * 1024) as tfp:
        for i, src_idx in enumerate(selected):
            _log_progress("build", i, kept_rows, build_start, every=20_000)
            src_idx_int = int(src_idx)
            start = int(line_offsets[src_idx_int])
            end = int(line_offsets[src_idx_int + 1])
            tfp.seek(start)
            line = tfp.read(end - start)
            if not line or line == b"\n":
                raise ValueError(
                    f"blank line in training.all.jsonl at line {src_idx_int + 1} (alignment would break)"
                )
            row = _loads_line(line)
            pixels[i] = extract_pixels_rgb(row)
            raw_numeric[i] = extract_numeric_row(row, schema)

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
        "total_rows_expected": int(total_rows_expected),
        "total_rows_seen": int(total_rows_expected),
        "eligible_rows": int(eligible_rows),
        "kept_rows": int(kept_rows),
        "max_blocks": None if max_blocks is None else int(max_blocks),
        "in_dir_cache_dir": str(paths["dir"]),
        "in_dir_cache_version": int(IN_DIR_CACHE_VERSION),
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
    build_dataset(run_dir, args.in_dir, args.seed, rebuild=args.rebuild, max_blocks=args.max_blocks)


if __name__ == "__main__":
    main()
