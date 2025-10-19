#!/usr/bin/env python3
"""TLG8 学習用データを並列に生成してマージする補助スクリプト。"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
    parser.add_argument(
        "--label-cache-bin",
        type=Path,
        default=None,
        help="ラベルキャッシュのバイナリを書き出すパス (省略可)",
    )
    parser.add_argument(
        "--label-cache-meta",
        type=Path,
        default=None,
        help="ラベルキャッシュのメタデータを書き出すパス (省略可)",
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
    label_cache_bin: Optional[Path],
    label_cache_meta: Optional[Path],
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
    if label_cache_bin is not None and label_cache_meta is not None:
        cmd.append(f"--label-cache-bin={label_cache_bin}")
        cmd.append(f"--label-cache-meta={label_cache_meta}")
    subprocess.run(cmd, check=True)


def worker(
    job_queue: "queue.Queue[Tuple[int, Path]]",
    result_queue: "queue.Queue[Tuple[int, Path, Optional[Path], Optional[Path]]]",
    tlgconv: Path,
    temp_dir: Path,
    enable_label_cache: bool,
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
        label_bin_path: Optional[Path] = None
        label_meta_path: Optional[Path] = None
        if enable_label_cache:
            label_bin_path = temp_dir / f"{tag}.labels.bin"
            label_meta_path = temp_dir / f"{tag}.labels.meta.json"
        run_tlgconv(tlgconv, image, tlg_path, jsonl_path, label_bin_path, label_meta_path)
        result_queue.put((index, jsonl_path, label_bin_path, label_meta_path))
        job_queue.task_done()


def append_jsonl(destination: Path, sources: Iterable[Path]) -> None:
    """JSONL ファイルを順番にマージする。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as dest:
        for src in sources:
            with src.open("rb") as data:
                shutil.copyfileobj(data, dest)


def append_label_cache(destination: Path, sources: Iterable[Path]) -> None:
    """ラベルキャッシュのバイナリを順に連結する。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as dest:
        for src in sources:
            with src.open("rb") as data:
                shutil.copyfileobj(data, dest)


def read_label_cache_meta(meta_path: Path) -> int:
    """部分的なラベルキャッシュメタデータからレコード数を得る。"""
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            data: Dict[str, object] = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"メタデータ {meta_path} の読み込みに失敗しました: {exc}") from exc

    schema = data.get("schema")
    if schema != 1:
        raise RuntimeError(f"メタデータ {meta_path} の schema が想定と異なります: {schema}")

    record_size = data.get("record_size")
    if record_size != 128:
        raise RuntimeError(f"メタデータ {meta_path} の record_size が 128 ではありません: {record_size}")

    record_count = data.get("record_count")
    if not isinstance(record_count, int):
        raise RuntimeError(f"メタデータ {meta_path} の record_count が不正です: {record_count}")

    return record_count


def compute_file_sha256(path: Path) -> str:
    """ファイルの SHA-256 を計算する。"""
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataset_sha256(inputs: List[Dict[str, object]]) -> str:
    """Python 実装と同じ手順でデータセットのハッシュを求める。"""
    hasher = hashlib.sha256()
    for item in inputs:
        sha_hex = item.get("sha256")
        size = item.get("size")
        path = item.get("path")
        if not isinstance(sha_hex, str) or not isinstance(size, int) or not isinstance(path, str):
            raise RuntimeError("データセットハッシュ計算用の入力情報が不正です")
        hasher.update(bytes.fromhex(sha_hex))
        hasher.update(size.to_bytes(8, "little"))
        hasher.update(path.encode("utf-8"))
    return hasher.hexdigest()


def write_label_cache_meta(
    meta_path: Path,
    bin_path: Path,
    record_count: int,
    training_jsonl: Path,
) -> None:
    """最終的なラベルキャッシュのメタデータを書き出す。"""
    bin_size = bin_path.stat().st_size
    expected_size = record_count * 128
    if bin_size != expected_size:
        raise RuntimeError(
            f"ラベルキャッシュバイナリのサイズ {bin_size} がレコード数 {record_count} と一致しません"
        )

    training_path = training_jsonl.resolve()
    stats = training_path.stat()
    sha256_hex = compute_file_sha256(training_path)
    inputs: List[Dict[str, object]] = [
        {
            "path": str(training_path),
            "size": int(stats.st_size),
            "mtime": stats.st_mtime,
            "sha256": sha256_hex,
        }
    ]
    dataset_sha = compute_dataset_sha256(inputs)

    meta = {
        "schema": 1,
        "record_size": 128,
        "record_count": record_count,
        "inputs": inputs,
        "dataset_sha256": dataset_sha,
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.parent / f"{meta_path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    tmp_path.replace(meta_path)


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

    label_cache_requested = args.label_cache_bin is not None or args.label_cache_meta is not None
    if label_cache_requested and (args.label_cache_bin is None or args.label_cache_meta is None):
        print("ラベルキャッシュを出力する場合は bin と meta の両方を指定してください", file=sys.stderr)
        return 1

    if label_cache_requested:
        args.label_cache_bin.parent.mkdir(parents=True, exist_ok=True)
        args.label_cache_bin.write_bytes(b"")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"")

    job_queue: "queue.Queue[Tuple[int, Path] | None]" = queue.Queue()
    result_queue: "queue.Queue[Tuple[int, Path, Optional[Path], Optional[Path]]]" = queue.Queue()

    workers = []
    for _ in range(args.threads):
        thread = threading.Thread(
            target=worker,
            args=(job_queue, result_queue, args.tlgconv, args.temp_dir, label_cache_requested),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    for idx, image in enumerate(images):
        job_queue.put((idx, image))

    for _ in workers:
        job_queue.put(None)

    pending: dict[int, Tuple[Path, Optional[Path], Optional[Path]]] = {}
    next_index = 0
    completed = 0
    total = len(images)
    total_label_records = 0

    while completed < total:
        index, jsonl_path, label_bin_path, label_meta_path = result_queue.get()
        pending[index] = (jsonl_path, label_bin_path, label_meta_path)
        while next_index in pending:
            path, bin_path, meta_path = pending.pop(next_index)
            append_jsonl(args.output, [path])
            path.unlink(missing_ok=True)
            if label_cache_requested:
                if bin_path is None or meta_path is None:
                    print("ラベルキャッシュの一時ファイル情報が欠落しています", file=sys.stderr)
                    return 1
                try:
                    record_count = read_label_cache_meta(meta_path)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                append_label_cache(args.label_cache_bin, [bin_path])
                total_label_records += record_count
                bin_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            next_index += 1
            completed += 1
            print(f"[{completed}/{total}] {path.name} をマージしました")

    job_queue.join()

    for thread in workers:
        thread.join()

    if label_cache_requested:
        try:
            write_label_cache_meta(
                args.label_cache_meta,
                args.label_cache_bin,
                total_label_records,
                args.output,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
