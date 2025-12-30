#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tlg_ml_utils import ensure_dir, now_iso, save_json


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        )
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote an existing bundle into a new ml/runs/<run_id>/ directory.")
    parser.add_argument("--src-bundle", required=True, type=Path, help="Path to an existing bundle.json")
    parser.add_argument("--run-id", required=True, type=str, help="New run id to create under ml/runs/")
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy *.pt files into the new run (recommended for portability).",
    )
    parser.add_argument(
        "--no-copy-weights",
        dest="copy_weights",
        action="store_false",
        help="Do not copy weights; keep original paths in the promoted bundle.",
    )
    parser.set_defaults(copy_weights=True)
    parser.add_argument("--force", action="store_true", help="Overwrite the destination directory if it exists.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "ml" / "runs" / args.run_id
    if run_dir.exists():
        if not args.force:
            raise SystemExit(f"destination already exists: {run_dir} (use --force to overwrite)")
        shutil.rmtree(run_dir)

    ensure_dir(run_dir / "artifacts" / "trial_0000")
    ensure_dir(run_dir / "splits")

    src_bundle = _load_json(args.src_bundle)

    promoted_bundle = dict(src_bundle)
    promoted_bundle["run_id"] = args.run_id
    promoted_bundle["trial_id"] = 0
    promoted_bundle["promoted_from"] = {
        "src_bundle": str(args.src_bundle),
        "timestamp": now_iso(),
        "git": _git_hash(),
    }

    if args.copy_weights:
        for head, info in promoted_bundle.get("heads", {}).items():
            if not isinstance(info, dict) or "path" not in info:
                continue
            src_path = repo_root / str(info["path"])
            if not src_path.exists():
                raise SystemExit(f"missing weight file for head={head}: {src_path}")
            dst_path = run_dir / "artifacts" / "trial_0000" / f"{head}.pt"
            shutil.copy2(src_path, dst_path)
            info["path"] = str(dst_path.relative_to(repo_root))

        bundle_dst = run_dir / "artifacts" / "trial_0000" / "bundle.json"
    else:
        bundle_dst = run_dir / "artifacts" / "trial_0000" / "bundle.json"

    save_json(bundle_dst, promoted_bundle)

    src_splits = args.src_bundle.parent.parent.parent / "splits" / "split.json"
    if src_splits.exists():
        shutil.copy2(src_splits, run_dir / "splits" / "split.json")

    cfg = {
        "kind": "promote_bundle",
        "run_id": args.run_id,
        "timestamp": now_iso(),
        "git": _git_hash(),
        "src_bundle": str(args.src_bundle),
        "copy_weights": bool(args.copy_weights),
    }
    save_json(run_dir / "config.json", cfg)

    progress = {
        "trial_id": 0,
        "timestamp": now_iso(),
        "git": _git_hash(),
        "event": "promote_bundle",
        "src_bundle": str(args.src_bundle),
        "bundle_path": str(bundle_dst),
        "valid_hit_rates_at": promoted_bundle.get("valid_hit_rates_at", {}),
        "params": promoted_bundle.get("params", {}),
        "heads": {k: v.get("hidden_sizes") for k, v in (promoted_bundle.get("heads") or {}).items() if isinstance(v, dict)},
    }
    with (run_dir / "progress.jsonl").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(progress, ensure_ascii=False) + "\n")

    save_json(
        run_dir / "best.json",
        {
            "best_trial_id": 0,
            "bundle_path": str(bundle_dst),
            "valid_hit_rates_at": promoted_bundle.get("valid_hit_rates_at", {}),
        },
    )

    print(str(bundle_dst))


if __name__ == "__main__":
    main()

