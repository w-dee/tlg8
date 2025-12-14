#!/usr/bin/env python3
"""
TLG8 学習用アーティファクト（per-image の小ファイル群）をパックして、ML データロード時の
ファイルシステム・オーバーヘッド（大量の open/stat）を減らすためのスクリプト。

入力（同一 relpath 構造）:
  - <training_json_root>/<relpath>.training.jsonl
  - <label_cache_root>/<relpath>.label_cache.bin
  - <label_cache_root>/<relpath>.label_cache.meta.json

出力（--out-dir）:
  - training.all.jsonl : JSON はパースせず、行をそのまま連結（末尾改行は必ず付与）
  - labels.all.bin     : label_cache.bin をバイナリ連結
  - index.jsonl        : 画像ごとのオフセット（行/レコード単位）
  - labels.meta.json   : 最小メタ + dataset_sha256（パック用に再計算）

これにより、学習時は「少数の大きいファイル + インデックス」でほぼ連続 I/O を行えるようになり、
データロードが高速化します。

実行例:
  python ml/pack_artifacts.py \
    --training-json-root /path/to/training_json \
    --label-cache-root   /path/to/label_cache \
    --out-dir            /path/to/packed_out

  # まずは小さく確認（書き込み無し）
  python ml/pack_artifacts.py \
    --training-json-root /path/to/training_json \
    --label-cache-root   /path/to/label_cache \
    --out-dir            /path/to/packed_out \
    --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, TextIO

try:
    # tqdm が入っていない環境でも動くように optional import にする。
    # 進捗表示は「あるなら使う」方針。
    from tqdm import tqdm  # type: ignore
except Exception:  # ImportError 以外も一旦吸収（壊れた環境でも本体処理は動かす）
    tqdm = None  # type: ignore


LABEL_RECORD_SIZE = 128
META_SCHEMA = 1


@dataclass(frozen=True)
class Pair:
    relpath: str  # 拡張子なし（例: "foo/bar/image.png" のような相対パス）
    training_jsonl: Path
    label_bin: Path
    label_meta: Path


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"真偽値として解釈できません: {value!r}（例: true/false）")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class TeeLogger:
    def __init__(self, fp: Optional[TextIO]):
        self._fp = fp

    def log(self, msg: str) -> None:
        line = f"[{now_ts()}] {msg}"
        print(line, flush=True)
        if self._fp is not None:
            self._fp.write(line + "\n")
            self._fp.flush()


def iter_training_files(training_root: Path, glob_pattern: str) -> list[tuple[str, Path]]:
    suffix = ".training.jsonl"
    seen: dict[str, Path] = {}
    for p in training_root.glob(glob_pattern):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(training_root).as_posix()
        except ValueError:
            # 通常起きない（glob は training_root 配下）
            continue
        if not rel.endswith(suffix):
            continue
        relpath = rel[: -len(suffix)]
        if relpath in seen:
            raise RuntimeError(f"同一 relpath の重複を検出: {relpath} ({seen[relpath]} と {p})")
        seen[relpath] = p
    return sorted(seen.items(), key=lambda x: x[0])


def build_pairs(
    training_root: Path,
    label_root: Path,
    glob_pattern: str,
    strict: bool,
    logger: TeeLogger,
) -> list[Pair]:
    items = iter_training_files(training_root, glob_pattern)
    pairs: list[Pair] = []
    missing = 0
    for relpath, training_path in items:
        label_bin = label_root / f"{relpath}.label_cache.bin"
        label_meta = label_root / f"{relpath}.label_cache.meta.json"
        if not label_bin.is_file() or not label_meta.is_file():
            missing += 1
            msg = (
                f"対応ファイルが見つかりません: relpath={relpath} "
                f"label_bin={label_bin.exists()} label_meta={label_meta.exists()}"
            )
            if strict:
                raise FileNotFoundError(msg)
            logger.log("WARN: " + msg + "（スキップ）")
            continue
        pairs.append(
            Pair(
                relpath=relpath,
                training_jsonl=training_path,
                label_bin=label_bin,
                label_meta=label_meta,
            )
        )
    logger.log(f"training jsonl 検出: {len(items)} 件, ペア作成: {len(pairs)} 件, 欠損: {missing} 件")
    return pairs


def count_jsonl_lines(path: Path) -> int:
    # JSON はパースせず、行数だけ数える（低メモリ、ストリーム処理）
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def validate_label_bin(path: Path) -> int:
    size = path.stat().st_size
    if size % LABEL_RECORD_SIZE != 0:
        raise ValueError(
            f"label_cache.bin のサイズが {LABEL_RECORD_SIZE} の倍数ではありません: {path} size={size}"
        )
    return size // LABEL_RECORD_SIZE


def read_and_validate_label_meta(path: Path, expected_record_count: int) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"label_cache.meta.json の JSON が壊れています: {path} ({e})") from e
    schema = meta.get("schema")
    record_size = meta.get("record_size")
    record_count = meta.get("record_count")
    if schema != META_SCHEMA:
        raise ValueError(f"meta schema 不一致: {path} schema={schema} expected={META_SCHEMA}")
    if record_size != LABEL_RECORD_SIZE:
        raise ValueError(
            f"meta record_size 不一致: {path} record_size={record_size} expected={LABEL_RECORD_SIZE}"
        )
    if record_count != expected_record_count:
        raise ValueError(
            f"meta record_count 不一致: {path} record_count={record_count} expected={expected_record_count}"
        )
    return meta


def sha256_file_bytes(path: Path, chunk_size: int = 8 * 1024 * 1024) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


def copy_training_jsonl(
    src: Path,
    dst: BinaryIO,
    chunked: bool = False,
) -> int:
    # 行単位でコピーしつつ、末尾改行を必ず付与する。
    # chunked は拡張用（現状は行単位のまま）。
    _ = chunked
    line_count = 0
    with src.open("rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                dst.write(line)
            else:
                dst.write(line + b"\n")
            line_count += 1
    return line_count


def copy_label_bin(src: Path, dst: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> bytes:
    # 連結しつつ、入力 bin の sha256(digest bytes) を返す。
    h = hashlib.sha256()
    with src.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            h.update(chunk)
    return h.digest()


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def check_overwrite_safety(out_dir: Path, force: bool, dry_run: bool, logger: TeeLogger) -> None:
    targets = [
        out_dir / "training.all.jsonl",
        out_dir / "labels.all.bin",
        out_dir / "index.jsonl",
        out_dir / "labels.meta.json",
    ]
    existing = [p for p in targets if p.exists()]
    if not existing:
        return
    msg = "出力ファイルが既に存在します: " + ", ".join(str(p) for p in existing)
    if dry_run:
        logger.log("WARN: " + msg + "（dry-run のため書き込みは行いません）")
        return
    if not force:
        raise FileExistsError(msg + "（上書きするには --force）")
    logger.log("WARN: " + msg + "（--force により上書きします）")


def open_log_file(dry_run: bool) -> Optional[TextIO]:
    if dry_run:
        return None
    log_path = Path("ml/pack_artifacts.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("w", encoding="utf-8", newline="\n")


def maybe_tqdm(
    iterable: Iterable,
    *,
    total: int,
    desc: str,
    unit: str,
):
    """
    tqdm が利用可能なら progress bar 付き iterable を返す。
    戻り値: (pbar_or_none, iterable)
    """
    if tqdm is None:
        return None, iterable
    pbar = tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.2,
    )
    return pbar, pbar


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TLG8 学習アーティファクトをパックして連続 I/O 化します。")
    parser.add_argument("--training-json-root", required=True, type=Path)
    parser.add_argument("--label-cache-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--glob-pattern", default="**/*.training.jsonl")
    parser.add_argument("--force", action="store_true", help="出力ファイルを上書きします。")
    parser.add_argument("--dry-run", action="store_true", help="書き込みを行わず、検証と予定出力のみ表示します。")
    parser.add_argument(
        "--strict",
        type=parse_bool,
        default=True,
        help="true: 不整合があれば即失敗 / false: 不正ペアをスキップ（default: true）",
    )
    parser.add_argument("--limit", type=int, default=None, help="先頭 N ペアのみ処理します。")
    args = parser.parse_args(argv)

    training_root: Path = args.training_json_root
    label_root: Path = args.label_cache_root
    out_dir: Path = args.out_dir
    glob_pattern: str = args.glob_pattern
    force: bool = args.force
    dry_run: bool = args.dry_run
    strict: bool = args.strict
    limit: Optional[int] = args.limit

    if not training_root.is_dir():
        raise FileNotFoundError(f"--training-json-root がディレクトリではありません: {training_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"--label-cache-root がディレクトリではありません: {label_root}")

    t0 = time.perf_counter()
    log_fp = open_log_file(dry_run=dry_run)
    try:
        logger = TeeLogger(log_fp)
        logger.log(
            "開始: "
            f"training_root={training_root} label_root={label_root} out_dir={out_dir} "
            f"glob={glob_pattern!r} strict={strict} dry_run={dry_run} force={force} limit={limit}"
        )

        pairs = build_pairs(training_root, label_root, glob_pattern, strict=strict, logger=logger)
        if limit is not None:
            if limit < 0:
                raise ValueError("--limit は 0 以上で指定してください。")
            pairs = pairs[:limit]
            logger.log(f"--limit 適用: 対象 {len(pairs)} 件")

        if not dry_run:
            ensure_out_dir(out_dir)
        check_overwrite_safety(out_dir, force=force, dry_run=dry_run, logger=logger)

        # 検証（strict の場合は最初の不整合で例外）
        validated: list[tuple[Pair, int, int, dict]] = []
        skipped = 0
        vbar, viter = maybe_tqdm(pairs, total=len(pairs), desc="検証", unit="file")
        try:
            for i, pair in enumerate(viter, start=1):
                try:
                    record_count = validate_label_bin(pair.label_bin)
                    meta = read_and_validate_label_meta(pair.label_meta, expected_record_count=record_count)
                    line_count = count_jsonl_lines(pair.training_jsonl)
                    if line_count != record_count:
                        raise ValueError(
                            "training jsonl 行数と label レコード数が不一致: "
                            f"relpath={pair.relpath} lines={line_count} records={record_count}"
                        )
                    validated.append((pair, line_count, record_count, meta))
                except Exception as e:
                    if strict:
                        raise
                    skipped += 1
                    logger.log(f"WARN: 検証失敗のためスキップ: {pair.relpath} ({e})")
                if vbar is not None:
                    vbar.set_postfix(ok=len(validated), skipped=skipped)
                elif i % 500 == 0:
                    logger.log(f"検証進捗: {i}/{len(pairs)} processed, ok={len(validated)} skipped={skipped}")
        finally:
            if vbar is not None:
                vbar.close()

        logger.log(f"検証完了: ok={len(validated)} skipped={skipped}")
        if dry_run:
            total_lines = sum(line_count for _, line_count, _, _ in validated)
            total_records = sum(record_count for _, _, record_count, _ in validated)
            elapsed = time.perf_counter() - t0
            logger.log(
                f"dry-run: 出力予定 training_lines={total_lines} label_records={total_records} elapsed={elapsed:.2f}s"
            )
            logger.log("dry-run: 生成予定ファイル: training.all.jsonl, labels.all.bin, index.jsonl, labels.meta.json")
            return 0

        # 本書き込み（中断時の破損を避けるため tmp に書いて最後に置換）
        tmp_suffix = f".tmp.{os.getpid()}"
        training_out_tmp = out_dir / f"training.all.jsonl{tmp_suffix}"
        labels_out_tmp = out_dir / f"labels.all.bin{tmp_suffix}"
        index_out_tmp = out_dir / f"index.jsonl{tmp_suffix}"
        meta_out_tmp = out_dir / f"labels.meta.json{tmp_suffix}"

        training_line_offset = 0
        label_record_offset = 0
        total_lines = 0
        total_records = 0
        dataset_hasher = hashlib.sha256()
        inputs_min: list[dict] = []

        t_write0 = time.perf_counter()
        with (
            training_out_tmp.open("wb") as training_out,
            labels_out_tmp.open("wb") as labels_out,
            index_out_tmp.open("w", encoding="utf-8", newline="\n") as index_out,
        ):
            wbar, witer = maybe_tqdm(validated, total=len(validated), desc="書き込み", unit="file")
            try:
                for idx, (pair, line_count, record_count, _meta) in enumerate(witer, start=1):
                    # training jsonl
                    copied_lines = copy_training_jsonl(pair.training_jsonl, training_out)
                    if copied_lines != line_count:
                        raise RuntimeError(
                            f"内部エラー: 行数が変化しました: relpath={pair.relpath} expected={line_count} got={copied_lines}"
                        )

                    # label bin（連結 + sha256）
                    label_bin_sha256 = copy_label_bin(pair.label_bin, labels_out)

                    # per-image meta sha256（元 JSON の bytes をハッシュ、軽量で十分）
                    label_meta_sha256 = sha256_file_bytes(pair.label_meta).hex()

                    # dataset_sha256 更新（指定フォーマット）
                    dataset_hasher.update(label_bin_sha256)
                    dataset_hasher.update(struct.pack("<Q", record_count))
                    dataset_hasher.update(pair.relpath.encode("utf-8"))

                    # index
                    entry = {
                        "relpath": pair.relpath,
                        "training_path": f"{pair.relpath}.training.jsonl",
                        "label_bin_path": f"{pair.relpath}.label_cache.bin",
                        "label_meta_path": f"{pair.relpath}.label_cache.meta.json",
                        "training_line_offset": training_line_offset,
                        "training_line_count": line_count,
                        "label_record_offset": label_record_offset,
                        "label_record_count": record_count,
                        "label_record_size": LABEL_RECORD_SIZE,
                    }
                    index_out.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

                    # labels.meta.json の最小 inputs
                    inputs_min.append(
                        {
                            "relpath": pair.relpath,
                            "label_meta_sha256": label_meta_sha256,
                            "label_bin_size": pair.label_bin.stat().st_size,
                            "record_count": record_count,
                        }
                    )

                    training_line_offset += line_count
                    label_record_offset += record_count
                    total_lines += line_count
                    total_records += record_count

                    if wbar is not None:
                        wbar.set_postfix(lines=total_lines, records=total_records)
                    elif idx % 200 == 0:
                        elapsed_write = time.perf_counter() - t_write0
                        logger.log(
                            f"書き込み進捗: {idx}/{len(validated)} "
                            f"lines={total_lines} records={total_records} elapsed={elapsed_write:.2f}s"
                        )
            finally:
                if wbar is not None:
                    wbar.close()

        packed_meta = {
            "schema": META_SCHEMA,
            "record_size": LABEL_RECORD_SIZE,
            "record_count": total_records,
            "inputs": inputs_min,
            "dataset_sha256": dataset_hasher.hexdigest(),
        }
        meta_out_tmp.write_text(json.dumps(packed_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # 最終配置（原子的に置換）
        os.replace(training_out_tmp, out_dir / "training.all.jsonl")
        os.replace(labels_out_tmp, out_dir / "labels.all.bin")
        os.replace(index_out_tmp, out_dir / "index.jsonl")
        os.replace(meta_out_tmp, out_dir / "labels.meta.json")

        elapsed = time.perf_counter() - t0
        logger.log(
            f"完了: ok={len(validated)} skipped={skipped} "
            f"total_lines={total_lines} total_records={total_records} elapsed={elapsed:.2f}s"
        )
        return 0
    finally:
        if log_fp is not None:
            log_fp.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # パイプ先が閉じられた場合（例: head）
        raise SystemExit(1)
