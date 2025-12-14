"""Run tlgconv over many BMP files in parallel and dump training/features/labels.

Behavior (kept):
- Recursively finds *.bmp under --input-dir (sorted).
- For each file, runs:
    tlgconv <input.bmp> <temp.tlg8> --tlg8-dump-mode=<mode> [dump options...]
- Output files are written under --out-training-json and/or --out-label-cache with
  mirrored relative paths from --input-dir.
- Skips work if all required outputs exist (unless --force).
- Supports parallel execution with --jobs.
- Writes failures to ml/run_tlgconv_dump.log (stdout/stderr tail).

Important fix:
- Temp outputs under --tlg8-temp-dir mirror the input relative path too, to avoid
  collisions when basenames match across subdirectories:
    <tlg8-temp-dir>/<relpath_without_ext>.tlg8

Examples:
  # Features/training JSONL only
  python ml/run_tlgconv_dump.py \\
    --input-dir data/images \\
    --tlgconv build/tlgconv \\
    --tlg8-temp-dir out/tmp_tlg8 \\
    --mode features \\
    --out-training-json out/training_jsonl \\
    --jobs 8

  # Label-cache only
  python ml/run_tlgconv_dump.py \\
    --input-dir data/images \\
    --tlgconv build/tlgconv \\
    --tlg8-temp-dir out/tmp_tlg8 \\
    --mode labels \\
    --out-label-cache out/label_cache \\
    --jobs 8

  # Both label-cache and training JSONL
  python ml/run_tlgconv_dump.py \\
    --input-dir data/images \\
    --tlgconv build/tlgconv \\
    --tlg8-temp-dir out/tmp_tlg8 \\
    --mode both \\
    --out-training-json out/training_jsonl \\
    --out-label-cache out/label_cache \\
    --jobs 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

LOG_PATH = Path("ml/run_tlgconv_dump.log")
MAX_LOG_OUTPUT = 2000


@dataclass
class JobResult:
    """A minimal container for a single input file result."""

    path: str
    status: str  # "success" | "failure" | "skipped" | "dry-run"
    command: str
    returncode: int | None = None
    output_tail: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tlgconv のダンプをまとめて実行する")
    parser.add_argument("--input-dir", required=True, type=Path, help="*.bmp を探すルートディレクトリ")
    parser.add_argument("--tlgconv", required=True, type=Path, help="tlgconv バイナリのパス")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="並列実行するジョブ数")
    parser.add_argument("--mode", choices=["features", "labels", "both"], default="both", help="ダンプモード")
    parser.add_argument("--out-training-json", type=Path, help="--tlg8-dump-training の出力先ルート")
    parser.add_argument("--out-label-cache", type=Path, help="--label-cache 出力先ルート")
    parser.add_argument("--tlg8-temp-dir", required=True, type=Path, help="tlgconv の一時出力先ルート")
    parser.add_argument("--tlgconv-args", type=str, default="", help="tlgconv に渡す追加引数 (シェル形式)")
    parser.add_argument("--force", action="store_true", help="既存出力を上書きして実行")
    parser.add_argument("--dry-run", action="store_true", help="実行せずにコマンドのみ表示")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir がディレクトリではありません: {args.input_dir}")

    if not args.tlgconv.exists():
        raise SystemExit(f"--tlgconv が存在しません: {args.tlgconv}")
    if not args.tlgconv.is_file():
        raise SystemExit(f"--tlgconv がファイルではありません: {args.tlgconv}")
    if not os.access(args.tlgconv, os.X_OK):
        raise SystemExit(f"--tlgconv が実行可能ではありません: {args.tlgconv}")

    if args.jobs <= 0:
        raise SystemExit("--jobs は 1 以上にしてください")

    if args.mode == "features":
        if not args.out_training_json:
            raise SystemExit('mode="features" では --out-training-json が必須です（何も出力されません）')
        return

    if args.mode in {"labels", "both"}:
        if not args.out_label_cache and not args.out_training_json:
            raise SystemExit(
                f'mode="{args.mode}" では --out-label-cache と --out-training-json のどちらかが必須です（何も出力されません）'
            )
        if args.mode == "labels" and not args.out_label_cache:
            print(
                'WARNING: mode="labels" ですが --out-label-cache が未指定です（training JSONL のみ出力します）'
            )


def find_bmp_files(root: Path) -> List[Path]:
    """Find *.bmp under the given root (sorted)."""

    return sorted(p for p in root.rglob("*.bmp") if p.is_file())


def rel_without_suffix(path: Path, root: Path) -> Path:
    """Return the path relative to root, without the filename suffix."""

    return path.relative_to(root).with_suffix("")


def build_required_outputs(
    rel_path: Path, *, mode: str, out_training_json: Path | None, out_label_cache: Path | None
) -> List[Path]:
    """Build the list of required output paths for a given input file."""

    outputs: List[Path] = []
    if out_training_json:
        outputs.append(out_training_json / (str(rel_path) + ".training.jsonl"))
    if out_label_cache and mode in {"labels", "both"}:
        base = out_label_cache / rel_path
        outputs.append(base.with_suffix(".label_cache.bin"))
        outputs.append(base.with_suffix(".label_cache.meta.json"))
    return outputs


def ensure_parent_dirs(paths: Iterable[str]) -> None:
    """Create parent directories for the given file paths."""

    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def should_skip(outputs: Sequence[Path], force: bool) -> bool:
    """Skip work if all required outputs already exist (unless forced)."""

    if force or not outputs:
        return False
    return all(p.exists() for p in outputs)


def temp_tlg8_path(tlg8_temp_dir: Path, rel_rootless: Path) -> Path:
    return (tlg8_temp_dir / rel_rootless).with_suffix(".tlg8")


def build_command(
    *,
    tlgconv: Path,
    input_path: Path,
    temp_output_path: Path,
    mode: str,
    rel_rootless: Path,
    out_training_json: Path | None,
    out_label_cache: Path | None,
    extra_args: Sequence[str],
) -> List[str]:
    """Build a tlgconv command line for one input file."""

    cmd = [str(tlgconv), str(input_path), str(temp_output_path), f"--tlg8-dump-mode={mode}"]

    if out_training_json:
        training_path = out_training_json / (str(rel_rootless) + ".training.jsonl")
        cmd.append(f"--tlg8-dump-training={training_path}")
    if out_label_cache and mode in {"labels", "both"}:
        base = out_label_cache / rel_rootless
        cmd.append(f"--label-cache-bin={base.with_suffix('.label_cache.bin')}")
        cmd.append(f"--label-cache-meta={base.with_suffix('.label_cache.meta.json')}")
    cmd.extend(list(extra_args))
    return cmd


def run_job(input_path: str, command: List[str], mkdir_targets: List[str], dry_run: bool) -> JobResult:
    """Run a single tlgconv job.

    Note: We intentionally use ThreadPoolExecutor and pass only primitives/lists here.
    This avoids pickling issues and avoids passing argparse.Namespace into workers.
    """

    cmd_str = shlex.join(command)
    if dry_run:
        return JobResult(input_path, "dry-run", cmd_str)

    try:
        ensure_parent_dirs(mkdir_targets)
        completed = subprocess.run(command, capture_output=True, text=True)
    except (FileNotFoundError, PermissionError, OSError) as e:
        msg = f"{type(e).__name__}: {e}"
        return JobResult(input_path, "failure", cmd_str, returncode=-1, output_tail=msg[-MAX_LOG_OUTPUT:])

    if completed.returncode == 0:
        return JobResult(input_path, "success", cmd_str, returncode=0)

    merged_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    tail = merged_output[-MAX_LOG_OUTPUT:]
    return JobResult(input_path, "failure", cmd_str, completed.returncode, tail)


def write_log(results: List[JobResult], *, started_at: _dt.datetime) -> None:
    """Write the summary and failures to the log file (overwrite)."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        success = sum(1 for r in results if r.status == "success")
        failure = sum(1 for r in results if r.status == "failure")
        skipped = sum(1 for r in results if r.status == "skipped")
        dry_run = sum(1 for r in results if r.status == "dry-run")

        f.write(f"timestamp: {started_at.isoformat(timespec='seconds')}\n")
        f.write("counts:\n")
        f.write(f"  success: {success}\n")
        f.write(f"  failure: {failure}\n")
        f.write(f"  skipped: {skipped}\n")
        f.write(f"  dry-run: {dry_run}\n")
        f.write("\n")

        for res in results:
            if res.status != "failure":
                continue
            f.write(f"[FAIL] {res.path}\n")
            f.write(f"  cmd: {res.command}\n")
            if res.returncode is not None:
                f.write(f"  returncode: {res.returncode}\n")
            if res.output_tail:
                f.write("  output (tail):\n")
                f.write(res.output_tail + "\n")
            f.write("\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started_at = _dt.datetime.now()
    args.tlg8_temp_dir.mkdir(parents=True, exist_ok=True)

    files = find_bmp_files(args.input_dir)
    if not files:
        print("入力ディレクトリに *.bmp が見つかりませんでした")
        return 1

    extra_args = shlex.split(args.tlgconv_args)
    runnable: List[tuple[str, List[str], List[str]]] = []
    skipped: List[JobResult] = []

    for p in files:
        rel = rel_without_suffix(p, args.input_dir)
        required_outputs = build_required_outputs(
            rel, mode=args.mode, out_training_json=args.out_training_json, out_label_cache=args.out_label_cache
        )
        if should_skip(required_outputs, args.force):
            skipped.append(JobResult(str(p), "skipped", ""))
            continue

        temp_out = temp_tlg8_path(args.tlg8_temp_dir, rel)
        cmd = build_command(
            tlgconv=args.tlgconv,
            input_path=p,
            temp_output_path=temp_out,
            mode=args.mode,
            rel_rootless=rel,
            out_training_json=args.out_training_json,
            out_label_cache=args.out_label_cache,
            extra_args=extra_args,
        )
        mkdir_targets = [str(temp_out)] + [str(x) for x in required_outputs]
        runnable.append((str(p), cmd, mkdir_targets))

    print(f"対象ファイル数: {len(files)}, 実行: {len(runnable)}, スキップ: {len(skipped)}")
    results: List[JobResult] = []

    if args.dry_run:
        for input_path, cmd, mkdir_targets in runnable:
            res = run_job(input_path, cmd, mkdir_targets, dry_run=True)
            print(res.command)
            results.append(res)
    else:
        # Threads are sufficient here because the heavy work is external subprocess calls,
        # and this avoids pickling/argparse.Namespace issues in multiprocessing.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(run_job, input_path, cmd, mkdir_targets, False)
                for (input_path, cmd, mkdir_targets) in runnable
            ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                if res.status == "failure":
                    print(f"FAIL: {res.path} (code={res.returncode})")
                else:
                    print(f"OK: {res.path}")

    results.extend(skipped)
    write_log(results, started_at=started_at)

    success = sum(1 for r in results if r.status == "success")
    failure = sum(1 for r in results if r.status == "failure")
    skip = sum(1 for r in results if r.status == "skipped")
    dry_run = sum(1 for r in results if r.status == "dry-run")

    print("--- summary ---")
    print(f"success: {success}")
    print(f"failure: {failure}")
    print(f"skipped: {skip}")
    print(f"dry-run: {dry_run}")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
