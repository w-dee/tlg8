#!/usr/bin/env python3
"""TLG8 学習用データを並列に生成してマージする補助スクリプト。"""

from __future__ import annotations

import argparse
import json
import struct
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .io_utils import (
        RECORD_SIZE,
        append_file,
        append_jsonl_and_count,
        count_lines,
        dataset_hash,
        sha256_file,
        validate_label_part,
    )
except ImportError:  # 実行方法によっては相対インポートが利用できない
    from io_utils import (  # type: ignore[no-redef]
        RECORD_SIZE,
        append_file,
        append_jsonl_and_count,
        count_lines,
        dataset_hash,
        sha256_file,
        validate_label_part,
    )


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
    parser.add_argument(
        "--feature-stats-bin",
        type=Path,
        default=None,
        help="特徴量統計のバイナリを書き出すパス (省略可)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="一時ファイルを検証するだけで最終出力を書き出さない",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="最終的なファイル連結のみ省略して挙動を確認する",
    )
    parser.add_argument(
        "--hexdump-head",
        type=int,
        default=0,
        metavar="N",
        help="最終 labels.bin の先頭 N バイトを 16 進で表示",
    )
    return parser.parse_args()


def discover_images(images_dir: Path, pattern: str) -> List[Path]:
    """指定ディレクトリから対象画像を列挙する。"""
    images = sorted(images_dir.glob(pattern))
    return [img for img in images if img.is_file()]


FEATURE_STATS_MAGIC = b"FSC8"
FEATURE_STATS_VERSION = 1
FEATURE_STATS_HEADER = struct.Struct("<4sIIQ")


def read_feature_stats(path: Path) -> tuple[int, List[float], List[float]]:
    """部分的な特徴量統計ファイルを読み取る。"""

    with path.open("rb") as fp:
        header = fp.read(FEATURE_STATS_HEADER.size)
        if len(header) != FEATURE_STATS_HEADER.size:
            raise RuntimeError(f"{path}: 特徴量統計ヘッダーを読み取れませんでした")
        magic, version, dimension, count = FEATURE_STATS_HEADER.unpack(header)
        if magic != FEATURE_STATS_MAGIC or version != FEATURE_STATS_VERSION:
            raise RuntimeError(f"{path}: 特徴量統計ヘッダーが不正です")
        sum_bytes = fp.read(8 * dimension)
        if len(sum_bytes) != 8 * dimension:
            raise RuntimeError(f"{path}: 特徴量統計 sum のサイズが不正です")
        sums = list(struct.unpack("<" + "d" * dimension, sum_bytes)) if dimension else []
        sumsq_bytes = fp.read(8 * dimension)
        if len(sumsq_bytes) != 8 * dimension:
            raise RuntimeError(f"{path}: 特徴量統計 sumsq のサイズが不正です")
        sumsq = list(struct.unpack("<" + "d" * dimension, sumsq_bytes)) if dimension else []
    return count, sums, sumsq


def write_feature_stats(path: Path, count: int, sums: List[float], sumsq: List[float]) -> None:
    """結合した特徴量統計ファイルを書き出す。"""

    if len(sums) != len(sumsq):
        raise RuntimeError("特徴量統計ベクトルの長さが一致しません")
    dimension = len(sums)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("wb") as fp:
        fp.write(FEATURE_STATS_HEADER.pack(FEATURE_STATS_MAGIC, FEATURE_STATS_VERSION, dimension, count))
        if dimension:
            fp.write(struct.pack("<" + "d" * dimension, *sums))
            fp.write(struct.pack("<" + "d" * dimension, *sumsq))
    tmp_path.replace(path)


def run_tlgconv(
    tlgconv: Path,
    image: Path,
    tlg_output: Path,
    jsonl_output: Path,
    label_cache_bin: Optional[Path],
    label_cache_meta: Optional[Path],
    feature_stats_path: Optional[Path],
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
    if feature_stats_path is not None:
        cmd.append(f"--tlg8-training-stats={feature_stats_path}")
    subprocess.run(cmd, check=True)


def worker(
    job_queue: "queue.Queue[Tuple[int, Path]]",
    result_queue: "queue.Queue[Tuple[int, Path, Optional[Path], Optional[Path], Optional[Path]]]",
    tlgconv: Path,
    temp_dir: Path,
    enable_label_cache: bool,
    enable_feature_stats: bool,
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
        stats_path: Optional[Path] = None
        if enable_feature_stats:
            stats_path = temp_dir / f"{tag}.stats.bin"
        run_tlgconv(
            tlgconv,
            image,
            tlg_path,
            jsonl_path,
            label_bin_path,
            label_meta_path,
            stats_path,
        )
        result_queue.put((index, jsonl_path, label_bin_path, label_meta_path, stats_path))
        job_queue.task_done()


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
    if record_size != RECORD_SIZE:
        raise RuntimeError(
            f"メタデータ {meta_path} の record_size が {RECORD_SIZE} ではありません: {record_size}"
        )

    record_count = data.get("record_count")
    if not isinstance(record_count, int):
        raise RuntimeError(f"メタデータ {meta_path} の record_count が不正です: {record_count}")

    return record_count


def write_label_cache_meta(
    meta_path: Path,
    bin_path: Path,
    record_count: int,
    training_jsonl: Path,
) -> None:
    """最終的なラベルキャッシュのメタデータを書き出す。"""
    bin_size = bin_path.stat().st_size
    expected_size = record_count * RECORD_SIZE
    if bin_size != expected_size:
        raise RuntimeError(
            f"ラベルキャッシュバイナリのサイズ {bin_size} がレコード数 {record_count} と一致しません"
        )

    training_path = training_jsonl.resolve()
    stats = training_path.stat()
    sha256_hex = sha256_file(training_path)
    inputs: List[Dict[str, object]] = [
        {
            "path": str(training_path),
            "size": int(stats.st_size),
            "mtime": stats.st_mtime,
            "sha256": sha256_hex,
        }
    ]
    dataset_sha = dataset_hash(inputs)

    meta = {
        "schema": 1,
        "record_size": RECORD_SIZE,
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
    if args.hexdump_head < 0:
        print("--hexdump-head には 0 以上を指定してください", file=sys.stderr)
        return 1

    if args.verify_only:
        args.dry_run = True

    label_cache_requested = args.label_cache_bin is not None or args.label_cache_meta is not None
    feature_stats_requested = args.feature_stats_bin is not None
    if label_cache_requested and (args.label_cache_bin is None or args.label_cache_meta is None):
        print("ラベルキャッシュを出力する場合は bin と meta の両方を指定してください", file=sys.stderr)
        return 1

    write_outputs = not args.verify_only and not args.dry_run

    if write_outputs:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(b"")
        if label_cache_requested:
            args.label_cache_bin.parent.mkdir(parents=True, exist_ok=True)
            args.label_cache_bin.write_bytes(b"")

    job_queue: "queue.Queue[Tuple[int, Path] | None]" = queue.Queue()
    result_queue: "queue.Queue[Tuple[int, Path, Optional[Path], Optional[Path], Optional[Path]]]" = queue.Queue()

    workers = []
    for _ in range(args.threads):
        thread = threading.Thread(
            target=worker,
            args=(
                job_queue,
                result_queue,
                args.tlgconv,
                args.temp_dir,
                label_cache_requested,
                feature_stats_requested,
            ),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    for idx, image in enumerate(images):
        job_queue.put((idx, image))

    for _ in workers:
        job_queue.put(None)

    pending: dict[int, Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]] = {}
    next_index = 0
    completed = 0
    total = len(images)
    total_label_records = 0
    total_jsonl_lines = 0
    stats_dim: Optional[int] = None
    stats_sum: Optional[List[float]] = None
    stats_sumsq: Optional[List[float]] = None
    stats_total_count = 0

    while completed < total:
        index, jsonl_path, label_bin_path, label_meta_path, stats_path = result_queue.get()
        pending[index] = (jsonl_path, label_bin_path, label_meta_path, stats_path)
        while next_index in pending:
            path, bin_path, meta_path, part_stats_path = pending.pop(next_index)
            try:
                if write_outputs:
                    part_lines = append_jsonl_and_count(args.output, path)
                else:
                    part_lines = count_lines(path)
            except Exception as exc:  # noqa: BLE001
                print(f"JSONL の追記に失敗しました ({path}): {exc}", file=sys.stderr)
                return 1
            total_jsonl_lines += part_lines
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
                try:
                    validate_label_part(bin_path, record_count)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                if part_lines != record_count:
                    print(
                        f"JSONL の行数 {part_lines} とラベル数 {record_count} が一致しません ({path.name})",
                        file=sys.stderr,
                    )
                    return 1
                if write_outputs:
                    try:
                        append_file(args.label_cache_bin, bin_path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"ラベルキャッシュの結合に失敗しました ({bin_path}): {exc}", file=sys.stderr)
                        return 1
                total_label_records += record_count
                bin_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            if feature_stats_requested:
                if part_stats_path is None:
                    print("特徴量統計の一時ファイル情報が欠落しています", file=sys.stderr)
                    return 1
                try:
                    part_count, part_sum, part_sumsq = read_feature_stats(part_stats_path)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                if part_lines != part_count:
                    print(
                        f"JSONL の行数 {part_lines} と特徴量統計のサンプル数 {part_count} が一致しません ({path.name})",
                        file=sys.stderr,
                    )
                    return 1
                if stats_dim is None:
                    stats_dim = len(part_sum)
                    stats_sum = [0.0] * stats_dim
                    stats_sumsq = [0.0] * stats_dim
                elif stats_dim != len(part_sum):
                    print("特徴量統計の次元数が一致しません", file=sys.stderr)
                    return 1
                if stats_sum is not None and stats_sumsq is not None:
                    for i in range(stats_dim or 0):
                        stats_sum[i] += part_sum[i]
                        stats_sumsq[i] += part_sumsq[i]
                stats_total_count += part_count
                part_stats_path.unlink(missing_ok=True)
            next_index += 1
            completed += 1
            print(f"[{completed}/{total}] {path.name} をマージしました")

    job_queue.join()

    for thread in workers:
        thread.join()

    if label_cache_requested:
        if total_jsonl_lines != total_label_records:
            print(
                f"マージ後の JSONL 行数 {total_jsonl_lines} とラベル数 {total_label_records} が一致しません",
                file=sys.stderr,
            )
            return 1
        if write_outputs:
            try:
                validate_label_part(args.label_cache_bin, total_label_records)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
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

    if feature_stats_requested:
        if stats_dim is None:
            stats_dim = 0
        if stats_sum is None:
            stats_sum = []
        if stats_sumsq is None:
            stats_sumsq = []
        if stats_total_count != total_jsonl_lines:
            print(
                f"マージ後の JSONL 行数 {total_jsonl_lines} と特徴量統計のサンプル数 {stats_total_count} が一致しません",
                file=sys.stderr,
            )
            return 1
        if write_outputs:
            try:
                write_feature_stats(args.feature_stats_bin, stats_total_count, stats_sum, stats_sumsq)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1

    if args.hexdump_head > 0 and label_cache_requested and args.label_cache_bin is not None:
        try:
            with args.label_cache_bin.open("rb") as fp:
                head = fp.read(args.hexdump_head)
        except FileNotFoundError:
            print("hexdump を要求されましたが labels.bin が存在しません", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"hexdump の取得に失敗しました: {exc}", file=sys.stderr)
            return 1
        print("labels.bin head:", head.hex())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
