#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tlg_ml_utils import ensure_dir, now_iso, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bits_best/bits_second sidecar from training.all.jsonl")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory containing training.all.jsonl")
    parser.add_argument("--run-id", required=True, type=str, help="Run ID under ml/runs")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing bits.all.npy")
    return parser.parse_args()


def extract_bits_row(row: dict[str, object]) -> tuple[int, int]:
    best = row.get("best")
    second = row.get("second")
    max_u64 = np.iinfo(np.uint64).max
    if best is None:
        bits_best = max_u64
    else:
        bits_best = int(best.get("bits", max_u64))
    if second is None:
        bits_second = max_u64
    else:
        bits_second = int(second.get("bits", max_u64))
    return bits_best, bits_second


def write_bits_sidecar(in_dir: Path, run_dir: Path, *, force: bool) -> Path:
    training_path = in_dir / "training.all.jsonl"
    if not training_path.is_file():
        raise SystemExit(f"training.all.jsonl not found: {training_path}")

    dataset_dir = run_dir / "dataset"
    ensure_dir(dataset_dir)
    out_path = dataset_dir / "bits.all.npy"
    if out_path.exists() and not force:
        raise SystemExit(f"bits.all.npy already exists: {out_path} (use --force to overwrite)")

    bits: list[tuple[int, int]] = []
    with training_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bits.append(extract_bits_row(row))

    arr = np.asarray(bits, dtype=np.uint64)
    np.save(out_path, arr)

    meta = {
        "created_at": now_iso(),
        "training_all_jsonl": str(training_path),
        "count": int(arr.shape[0]),
        "path": str(out_path),
    }
    save_json(dataset_dir / "bits.meta.json", meta)
    return out_path


def main() -> None:
    args = parse_args()
    run_dir = args.run_root / args.run_id
    write_bits_sidecar(args.in_dir, run_dir, force=args.force)


if __name__ == "__main__":
    main()
