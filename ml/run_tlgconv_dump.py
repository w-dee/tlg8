"""tlgconv を複数の BMP ファイルに対して並列実行する簡易スクリプト。

- 入力ディレクトリ以下の *.bmp を再帰的に処理する。
- モード(features/labels/both)に応じて tlgconv のダンプオプションを切り替える。
- 出力先は入力ルートからの相対パス構造を保つ。
- 既存出力が揃っていればスキップし、--force で上書き実行する。
- --dry-run で実行せずにコマンドのみ表示する。
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
    """各ファイルに対する実行結果を保持するシンプルな構造体。"""

    path: Path
    status: str  # "success" | "failure" | "skipped"
    command: str
    returncode: int | None = None
    output_tail: str | None = None


@dataclass
class Job:
    """実行に必要な情報をひとまとめにするだけの入れ物。"""

    input_path: Path
    rel_rootless: Path
    outputs: List[Path]


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


def find_bmp_files(root: Path) -> List[Path]:
    """入力ディレクトリ以下の *.bmp をソートして返す。"""

    return sorted(p for p in root.rglob("*.bmp") if p.is_file())


def rel_without_suffix(path: Path, root: Path) -> Path:
    """入力ルートからの相対パスを拡張子なしで返す。"""

    return path.relative_to(root).with_suffix("")


def build_outputs(rel_path: Path, args: argparse.Namespace) -> List[Path]:
    """モードと出力指定に応じて必要な出力パス一覧を生成する。"""

    outputs: List[Path] = []
    if args.out_training_json:
        outputs.append(args.out_training_json / (str(rel_path) + ".training.jsonl"))
    if args.out_label_cache and args.mode in {"labels", "both"}:
        base = args.out_label_cache / rel_path
        outputs.append(base.with_suffix(".label_cache.bin"))
        outputs.append(base.with_suffix(".label_cache.meta.json"))
    return outputs


def ensure_parent_dirs(paths: Iterable[Path]) -> None:
    """出力ファイルの親ディレクトリを作成する。"""

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def should_skip(outputs: Sequence[Path], force: bool) -> bool:
    """全ての出力が存在する場合にスキップする。"""

    if force or not outputs:
        return False
    return all(p.exists() for p in outputs)


def build_command(job: Job, extra_args: List[str], mode: str, args: argparse.Namespace) -> List[str]:
    """tlgconv 実行コマンドを組み立てる。"""

    output_path = args.tlg8_temp_dir / job.input_path.with_suffix(".tlg8").name
    cmd = [str(args.tlgconv), str(job.input_path), str(output_path), f"--tlg8-dump-mode={mode}"]
    if args.out_training_json:
        training_path = args.out_training_json / (str(job.rel_rootless) + ".training.jsonl")
        cmd.append(f"--tlg8-dump-training={training_path}")
    if args.out_label_cache and mode in {"labels", "both"}:
        base = args.out_label_cache / job.rel_rootless
        cmd.append(f"--label-cache-bin={base.with_suffix('.label_cache.bin')}")
        cmd.append(f"--label-cache-meta={base.with_suffix('.label_cache.meta.json')}")
    cmd.extend(extra_args)
    return cmd


def run_job(job: Job, mode: str, extra_args: List[str], dry_run: bool, args: argparse.Namespace) -> JobResult:
    """単一ファイルに対して tlgconv を実行する。"""

    command = build_command(job, extra_args, mode, args)
    if dry_run:
        return JobResult(job.input_path, "skipped", " ".join(command))

    ensure_parent_dirs(job.outputs)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        return JobResult(job.input_path, "success", " ".join(command))

    merged_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    tail = merged_output[-MAX_LOG_OUTPUT:]
    return JobResult(job.input_path, "failure", " ".join(command), completed.returncode, tail)


def write_log(results: List[JobResult]) -> None:
    """ログファイルに結果を追記する。"""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
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
    args.tlg8_temp_dir.mkdir(parents=True, exist_ok=True)
    files = find_bmp_files(args.input_dir)
    if not files:
        print("入力ディレクトリに *.bmp が見つかりませんでした")
        return 1

    extra_args = shlex.split(args.tlgconv_args)
    jobs: List[Job] = []
    skipped: List[JobResult] = []

    for p in files:
        rel = rel_without_suffix(p, args.input_dir)
        outputs = build_outputs(rel, args)
        if should_skip(outputs, args.force):
            skipped.append(JobResult(p, "skipped", ""))
            continue
        jobs.append(Job(p, rel, outputs))

    print(f"対象ファイル数: {len(files)}, 実行: {len(jobs)}, スキップ: {len(skipped)}")
    results: List[JobResult] = []

    if args.dry_run:
        for job in jobs:
            res = run_job(job, args.mode, extra_args, dry_run=True, args=args)
            print(res.command)
            results.append(res)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_job = {
                executor.submit(run_job, job, args.mode, extra_args, False, args): job for job in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                res = future.result()
                results.append(res)
                if res.status == "failure":
                    print(f"FAIL: {res.path} (code={res.returncode})")
                else:
                    print(f"OK: {res.path}")

    results.extend(skipped)
    write_log(results)

    success = sum(1 for r in results if r.status == "success")
    failure = sum(1 for r in results if r.status == "failure")
    skip = sum(1 for r in results if r.status == "skipped")

    print("--- summary ---")
    print(f"success: {success}")
    print(f"failure: {failure}")
    print(f"skipped: {skip}")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
