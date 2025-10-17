#!/usr/bin/env python3
"""TLG8 学習用データを並列に生成してマージする補助スクリプト。"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="build/tlgconv を並列実行し、学習用 JSONL を随時マージする"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="同時に起動する build/tlgconv の数 (既定: 1)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("test/images"),
        help="処理する BMP 画像が配置されたディレクトリ",
    )
    parser.add_argument(
        "--tlgconv",
        type=Path,
        default=Path("build/tlgconv"),
        help="tlgconv バイナリのパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tlg8_training.jsonl"),
        help="マージ先の JSONL ファイル",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/tlg8_training_parts"),
        help="一時的な JSONL を配置するディレクトリ",
    )
    parser.add_argument(
        "--images-glob",
        default="*.bmp",
        help="処理対象を絞り込むための glob パターン",
    )
    return parser.parse_args()


def discover_images(images_dir: Path, pattern: str) -> List[Path]:
    """指定ディレクトリから対象画像を列挙する。"""
    images = sorted(images_dir.glob(pattern))
    return [img for img in images if img.is_file()]


def run_tlgconv(
    tlgconv: Path,
    image: Path,
    tlg_output: Path,
    jsonl_output: Path,
) -> None:
    """build/tlgconv を 1 回実行する。"""
    cmd = [
        str(tlgconv),
        str(image),
        str(tlg_output),
        "--tlg-version=8",
        f"--tlg8-dump-training={jsonl_output}",
        f"--tlg8-training-tag={image.name}",
    ]
    subprocess.run(cmd, check=True)


def worker(
    job_queue: "queue.Queue[Tuple[int, Path]]",
    result_queue: "queue.Queue[Tuple[int, Path]]",
    tlgconv: Path,
    temp_dir: Path,
) -> None:
    """ジョブを取り出して tlgconv を実行し、結果 JSONL を報告する。"""
    while True:
        item = job_queue.get()
        if item is None:
            job_queue.task_done()
            break
        index, image = item
        tag = image.name
        tlg_path = Path("/tmp") / f"{image.stem}.tlg8"
        jsonl_path = temp_dir / f"{tag}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        run_tlgconv(tlgconv, image, tlg_path, jsonl_path)
        result_queue.put((index, jsonl_path))
        job_queue.task_done()


def append_jsonl(destination: Path, sources: Iterable[Path]) -> None:
    """JSONL ファイルを順番にマージする。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as dest:
        for src in sources:
            with src.open("rb") as data:
                shutil.copyfileobj(data, dest)


def main() -> int:
    """エントリーポイント。"""
    args = parse_args()
    images = discover_images(args.images_dir, args.images_glob)
    if not images:
        print("処理対象の画像が見つかりません", file=sys.stderr)
        return 1
    if args.threads < 1:
        print("--threads は 1 以上を指定してください", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"")

    job_queue: "queue.Queue[Tuple[int, Path] | None]" = queue.Queue()
    result_queue: "queue.Queue[Tuple[int, Path]]" = queue.Queue()

    workers = []
    for _ in range(args.threads):
        thread = threading.Thread(
            target=worker,
            args=(job_queue, result_queue, args.tlgconv, args.temp_dir),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    for idx, image in enumerate(images):
        job_queue.put((idx, image))

    for _ in workers:
        job_queue.put(None)

    pending: dict[int, Path] = {}
    next_index = 0
    completed = 0
    total = len(images)

    while completed < total:
        index, jsonl_path = result_queue.get()
        pending[index] = jsonl_path
        while next_index in pending:
            path = pending.pop(next_index)
            append_jsonl(args.output, [path])
            path.unlink(missing_ok=True)
            next_index += 1
            completed += 1
            print(f"[{completed}/{total}] {path.name} をマージしました")

    job_queue.join()

    for thread in workers:
        thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
