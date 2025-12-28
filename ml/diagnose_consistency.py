#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dataset_builder import (
    IN_DIR_CACHE_VERSION,
    _labels_record_count,
    _in_dir_cache_paths,
    _load_in_dir_cache,
    _loads_line,
)
from tlg_ml_utils import ensure_dir, now_iso, read_label_record, split_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ML dataset / label / bits consistency")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory with training.all.jsonl, labels.all.bin")
    parser.add_argument("--run-id", required=True, type=str, help="Run ID under ml/runs (for writing report)")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed")
    parser.add_argument("--samples", type=int, default=2048, help="Number of random indices to check")
    parser.add_argument("--require-eligible", action="store_true", help="Sample only eligible rows")
    return parser.parse_args()


def _stat_fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _row_tuple(row: dict[str, Any], key: str) -> tuple[int, int, int, int, int]:
    obj = row.get(key) if isinstance(row, dict) else None
    if not isinstance(obj, dict):
        return (-1, -1, -1, -1, -1)
    pred = _as_int(obj.get("predictor"))
    filt = _as_int(obj.get("filter"))
    re = _as_int(obj.get("reorder"))
    inter = _as_int(obj.get("interleave"))
    bits = _as_int(obj.get("bits"))
    return (pred, filt, re, inter, bits)


def _labels_tuple(lab: dict[str, int], which: str) -> tuple[int, int, int, int, int, int]:
    if which == "best":
        return (
            int(lab["best_predictor"]),
            int(lab["best_filter_perm"]),
            int(lab["best_filter_primary"]),
            int(lab["best_filter_secondary"]),
            int(lab["best_reorder"]),
            int(lab["best_interleave"]),
        )
    if which == "second":
        return (
            int(lab["second_predictor"]),
            int(lab["second_filter_perm"]),
            int(lab["second_filter_primary"]),
            int(lab["second_filter_secondary"]),
            int(lab["second_reorder"]),
            int(lab["second_interleave"]),
        )
    raise ValueError("which must be best or second")


def _is_eligible_recompute(row: dict[str, Any], lab: dict[str, int]) -> bool:
    if row.get("block_size") != [8, 8]:
        return False
    components = _as_int(row.get("components"), 0)
    if components not in (3, 4):
        return False
    best = row.get("best")
    second = row.get("second")
    if not isinstance(best, dict) or not isinstance(second, dict):
        return False
    if _as_int(best.get("entropy"), -1) != 0 or _as_int(second.get("entropy"), -1) != 0:
        return False
    if int(lab["second_predictor"]) == -1:
        return False
    if _labels_tuple(lab, "best") == _labels_tuple(lab, "second"):
        return False
    return True


def main() -> None:
    args = parse_args()
    run_dir = args.run_root / args.run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")

    training_path = args.in_dir / "training.all.jsonl"
    label_path = args.in_dir / "labels.all.bin"
    if not training_path.is_file():
        raise SystemExit(f"training.all.jsonl not found: {training_path}")
    if not label_path.is_file():
        raise SystemExit(f"labels.all.bin not found: {label_path}")

    t0 = time.time()
    paths = _in_dir_cache_paths(args.in_dir)
    total_rows_expected = _labels_record_count(label_path)
    schema, raw_names, line_offsets, eligible_mask, bits_all = _load_in_dir_cache(
        args.in_dir,
        training_path,
        label_path,
        total_rows_expected,
    )
    if schema is None or raw_names is None or line_offsets is None or eligible_mask is None or bits_all is None:
        raise SystemExit(
            f"in-dir cache v{IN_DIR_CACHE_VERSION} missing at {paths['dir']}; run dataset build once to create it"
        )
    total_rows = int(eligible_mask.shape[0])

    if args.require_eligible:
        pool = np.flatnonzero(eligible_mask).astype(np.int64, copy=False)
    else:
        pool = np.arange(total_rows, dtype=np.int64)
    if pool.size == 0:
        raise SystemExit("no rows to sample")
    samples = min(int(args.samples), int(pool.size))
    rng = np.random.default_rng(int(args.seed))
    chosen = np.sort(rng.choice(pool, size=samples, replace=False))

    # Read labels sequentially for speed (random-access would require seeking).
    chosen_set = set(int(v) for v in chosen.tolist())
    labels_by_idx: dict[int, dict[str, int]] = {}
    with label_path.open("rb") as lfp:
        for idx in range(total_rows):
            lab = read_label_record(lfp)
            if idx in chosen_set:
                labels_by_idx[idx] = lab
            if len(labels_by_idx) == samples:
                break

    mismatch = {
        "label_vs_json_best": 0,
        "label_vs_json_second": 0,
        "bits_best": 0,
        "bits_second": 0,
        "eligible_mask": 0,
        "missing_best_or_second": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = {k: [] for k in mismatch.keys()}
    max_examples = 5

    with training_path.open("rb", buffering=8 * 1024 * 1024) as tfp:
        for idx in chosen.tolist():
            idx_i = int(idx)
            lab = labels_by_idx.get(idx_i)
            if lab is None:
                raise SystemExit(f"failed to load label for idx={idx_i}")
            start = int(line_offsets[idx_i])
            end = int(line_offsets[idx_i + 1])
            tfp.seek(start)
            line = tfp.read(end - start)
            row = _loads_line(line)

            best_pred, best_filter, best_reorder, best_inter, best_bits = _row_tuple(row, "best")
            sec_pred, sec_filter, sec_reorder, sec_inter, sec_bits = _row_tuple(row, "second")
            if best_pred < 0 or sec_pred < 0:
                mismatch["missing_best_or_second"] += 1
                if len(examples["missing_best_or_second"]) < max_examples:
                    examples["missing_best_or_second"].append({"idx": idx_i, "best": row.get("best"), "second": row.get("second")})
                continue

            best_perm, best_prim, best_sec = split_filter(int(best_filter))
            sec_perm, sec_prim, sec_sec = split_filter(int(sec_filter))
            lbest = _labels_tuple(lab, "best")
            lsec = _labels_tuple(lab, "second")
            if (best_pred, best_perm, best_prim, best_sec, best_reorder, best_inter) != lbest:
                mismatch["label_vs_json_best"] += 1
                if len(examples["label_vs_json_best"]) < max_examples:
                    examples["label_vs_json_best"].append(
                        {
                            "idx": idx_i,
                            "json_best": (best_pred, best_filter, best_reorder, best_inter),
                            "labels_best": lbest,
                        }
                    )
            if (sec_pred, sec_perm, sec_prim, sec_sec, sec_reorder, sec_inter) != lsec:
                mismatch["label_vs_json_second"] += 1
                if len(examples["label_vs_json_second"]) < max_examples:
                    examples["label_vs_json_second"].append(
                        {
                            "idx": idx_i,
                            "json_second": (sec_pred, sec_filter, sec_reorder, sec_inter),
                            "labels_second": lsec,
                        }
                    )

            bits_row_best = int(best_bits) & ((1 << 64) - 1)
            bits_row_second = int(sec_bits) & ((1 << 64) - 1)
            bits_cache_best = int(bits_all[idx_i, 0])
            bits_cache_second = int(bits_all[idx_i, 1])
            if bits_row_best != bits_cache_best:
                mismatch["bits_best"] += 1
                if len(examples["bits_best"]) < max_examples:
                    examples["bits_best"].append(
                        {"idx": idx_i, "json_bits_best": bits_row_best, "cache_bits_best": bits_cache_best}
                    )
            if bits_row_second != bits_cache_second:
                mismatch["bits_second"] += 1
                if len(examples["bits_second"]) < max_examples:
                    examples["bits_second"].append(
                        {"idx": idx_i, "json_bits_second": bits_row_second, "cache_bits_second": bits_cache_second}
                    )

            elig_re = _is_eligible_recompute(row, lab)
            elig_cache = bool(int(eligible_mask[idx_i]) != 0)
            if elig_re != elig_cache:
                mismatch["eligible_mask"] += 1
                if len(examples["eligible_mask"]) < max_examples:
                    examples["eligible_mask"].append(
                        {"idx": idx_i, "eligible_recompute": bool(elig_re), "eligible_cache": bool(elig_cache)}
                    )

    report = {
        "created_at": now_iso(),
        "elapsed_sec": float(time.time() - t0),
        "in_dir": str(args.in_dir),
        "files": {
            "training_all_jsonl": _stat_fingerprint(training_path),
            "labels_all_bin": _stat_fingerprint(label_path),
        },
        "cache_dir": str(paths["dir"]),
        "cache_version": int(IN_DIR_CACHE_VERSION),
        "total_rows": int(total_rows),
        "sampled_rows": int(samples),
        "require_eligible": bool(args.require_eligible),
        "mismatch_counts": mismatch,
        "examples": examples,
    }
    out_path = run_dir / "artifacts" / "diagnostics_consistency.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["mismatch_counts"], sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
