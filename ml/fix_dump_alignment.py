#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix tlgconv dump alignment (training jsonl line count vs label records)")
    p.add_argument("--training-root", type=Path, required=True, help="Root containing *.training.jsonl")
    p.add_argument("--label-root", type=Path, required=True, help="Root containing *.label_cache.meta.json")
    p.add_argument("--glob", type=str, default="**/*.label_cache.meta.json", help="Glob under --label-root")
    p.add_argument("--dry-run", action="store_true", help="Do not modify files")
    return p.parse_args()


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fp:
        for _ in fp:
            n += 1
    return n


def truncate_jsonl(path: Path, keep_lines: int, *, dry_run: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if dry_run:
        return
    with path.open("rb") as src, tmp.open("wb") as dst:
        n = 0
        for line in src:
            if n >= keep_lines:
                break
            if line.endswith(b"\n"):
                dst.write(line)
            else:
                dst.write(line + b"\n")
            n += 1
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    metas = sorted(args.label_root.glob(args.glob))
    if not metas:
        raise SystemExit(f"no meta files found under {args.label_root} with glob {args.glob!r}")

    fixed = 0
    ok = 0
    skipped = 0
    for meta_path in metas:
        rel = meta_path.relative_to(args.label_root).as_posix()
        if not rel.endswith(".label_cache.meta.json"):
            continue
        relpath = rel[: -len(".label_cache.meta.json")]
        training_path = args.training_root / (relpath + ".training.jsonl")
        if not training_path.is_file():
            print(f"WARN missing training jsonl: {training_path}")
            skipped += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = int(meta.get("record_count", -1))
        if expected <= 0:
            print(f"WARN bad record_count in {meta_path}")
            skipped += 1
            continue
        actual = count_lines(training_path)
        if actual == expected:
            ok += 1
            continue
        if actual < expected:
            print(f"WARN too few lines: relpath={relpath} lines={actual} expected={expected}")
            skipped += 1
            continue
        print(f"FIX relpath={relpath} lines={actual} -> {expected}")
        truncate_jsonl(training_path, expected, dry_run=args.dry_run)
        fixed += 1

    print(f"done: ok={ok} fixed={fixed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

