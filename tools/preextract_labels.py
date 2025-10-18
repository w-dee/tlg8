#!/usr/bin/env python3
"""TLG8 学習データのラベルを事前抽出して固定長バイナリに保存するツール。"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import math
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import orjson as _fast_json
except Exception:  # pragma: no cover - orjson が無い環境向けフォールバック
    _fast_json = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm が利用不可の場合
    tqdm = None

MAGIC = 0x4C424C38  # 'LBL8'
VERSION = 1
STRUCT_FMT = "<IHH12hI92x"
STRUCT = struct.Struct(STRUCT_FMT)
EXPECTED_RECORD_SIZE = STRUCT.size


class _SimpleProgressBar:
    """tqdm 不在時に利用する簡易プログレスバー。"""

    def __init__(self, total: int, desc: str, unit: str, *, unit_scale: bool = False) -> None:
        self._total = max(0, int(total))
        self._desc = desc
        self._unit = unit
        self._unit_scale = unit_scale
        self._current = 0
        self._printed = False
        self._last_message_len = 0
        if self._total > 0:
            self._render()

    def _format_value(self, value: int) -> str:
        if not self._unit_scale or value <= 0:
            return f"{value} {self._unit}"
        suffixes = ["B", "KB", "MB", "GB", "TB"]
        scaled = float(value)
        idx = 0
        while scaled >= 1024.0 and idx < len(suffixes) - 1:
            scaled /= 1024.0
            idx += 1
        return f"{scaled:.2f} {suffixes[idx]}"

    def _render(self) -> None:
        message = f"{self._desc}: {self._format_value(self._current)}/{self._format_value(self._total)}"
        padding = max(0, self._last_message_len - len(message))
        sys.stderr.write("\r" + message + (" " * padding))
        sys.stderr.flush()
        self._printed = True
        self._last_message_len = len(message)

    def update(self, amount: int) -> None:
        if amount <= 0 or self._total <= 0:
            return
        self._current = min(self._total, self._current + amount)
        self._render()

    def close(self) -> None:
        if self._printed:
            sys.stderr.write("\n")
            sys.stderr.flush()


class ProgressReporter:
    """tqdm を優先しつつ、利用不可時は簡易バーにフォールバックするヘルパー。"""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        unit: str = "件",
        enable: bool = True,
        unit_scale: bool = False,
    ) -> None:
        self._enabled = enable and total > 0
        self._bar = None
        self._simple: Optional[_SimpleProgressBar] = None
        if not self._enabled:
            return
        if tqdm is not None:
            self._bar = tqdm(total=total, desc=desc, unit=unit, unit_scale=unit_scale)
        else:
            self._simple = _SimpleProgressBar(total, desc, unit, unit_scale=unit_scale)

    def update(self, amount: int) -> None:
        if not self._enabled or amount <= 0:
            return
        if self._bar is not None:
            self._bar.update(amount)
        elif self._simple is not None:
            self._simple.update(amount)

    def close(self) -> None:
        if not self._enabled:
            return
        if self._bar is not None:
            self._bar.close()
        elif self._simple is not None:
            self._simple.close()


@dataclass
class FileRef:
    """入力ファイルのメタ情報。"""

    path: Path
    size: int
    mtime: float


class ThreadLocalJsonReader:
    """スレッドごとにファイルハンドルを保持しつつ JSON 行を読み出す補助クラス。"""

    def __init__(self, files: Sequence[FileRef]) -> None:
        self._files = files
        self._local = threading.local()

    def _handle(self, file_id: int) -> io.BufferedReader:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
        handle = handles.get(file_id)
        if handle is None or handle.closed:
            handle = open(self._files[file_id].path, "rb", buffering=1024 * 1024)
            handles[file_id] = handle
        return handle

    def read_line(self, file_id: int, offset: int) -> bytes:
        handle = self._handle(file_id)
        handle.seek(offset)
        return handle.readline()


def discover_inputs(paths: Sequence[str]) -> List[Path]:
    """CLI 引数から JSONL ファイル一覧を決定する。"""

    results: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.rglob("*.jsonl")):
                if child.is_file():
                    results.append(child)
        elif path.is_file():
            results.append(path)
    return sorted({p.resolve() for p in results})


def _load_index_cache(paths: Sequence[Path], cache_path: Path) -> Tuple[Optional[List[FileRef]], Optional[np.ndarray]]:
    """インデックスキャッシュが利用可能なら読み出す。"""

    if not cache_path.exists():
        return None, None
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    try:
        with meta_path.open("r", encoding="utf-8") as meta_fp:
            meta = json.load(meta_fp)
    except Exception:
        return None, None
    files_meta = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files_meta, list) or len(files_meta) != len(paths):
        return None, None
    loaded: List[FileRef] = []
    for path, entry in zip(paths, files_meta):
        if not isinstance(entry, dict):
            return None, None
        try:
            recorded_path = Path(entry["path"]).resolve()
            size = int(entry["size"])
            mtime = float(entry["mtime"])
        except (KeyError, TypeError, ValueError):
            return None, None
        try:
            stat = path.stat()
        except OSError:
            return None, None
        if recorded_path != path.resolve():
            return None, None
        if int(stat.st_size) != size:
            return None, None
        if not math.isclose(stat.st_mtime, mtime, rel_tol=0.0, abs_tol=1e-6):
            return None, None
        loaded.append(FileRef(path=path, size=size, mtime=mtime))
    try:
        index = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None, None
    index = np.asarray(index, dtype=np.int64)
    if index.ndim == 1:
        if index.size % 2 != 0:
            return None, None
        index = index.reshape(-1, 2)
    elif index.ndim != 2 or index.shape[1] != 2:
        return None, None
    return loaded, index


def _save_index_cache(cache_path: Path, files: Sequence[FileRef], index: np.ndarray) -> None:
    """インデックスを NPY とメタデータに保存する。"""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, index)
    meta = {
        "files": [
            {
                "path": str(ref.path.resolve()),
                "size": int(ref.size),
                "mtime": float(ref.mtime),
            }
            for ref in files
        ]
    }
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp)


def build_index(paths: Sequence[Path], cache_path: Optional[Path], *, show_progress: bool) -> Tuple[List[FileRef], np.ndarray]:
    """JSONL 群から (file_id, offset) インデックスを構築する。"""

    if cache_path is not None:
        loaded, cached = _load_index_cache(paths, cache_path)
        if loaded is not None and cached is not None:
            return loaded, cached

    files: List[FileRef] = []
    index_list: List[Tuple[int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            print(f"警告: {path} にアクセスできません: {exc}", file=sys.stderr)
            continue
        file_ref = FileRef(path=path, size=int(stat.st_size), mtime=float(stat.st_mtime))
        files.append(file_ref)
        file_id = len(files) - 1
        try:
            with path.open("rb", buffering=1024 * 1024) as fp:
                offset = fp.tell()
                last = offset
                bar = ProgressReporter(
                    file_ref.size,
                    f"{path.name}: インデックス作成",
                    unit="B",
                    enable=show_progress,
                    unit_scale=True,
                )
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    if line.strip():
                        index_list.append((file_id, offset))
                    offset = fp.tell()
                    bar.update(offset - last)
                    last = offset
                if last < file_ref.size:
                    bar.update(file_ref.size - last)
                bar.close()
        except OSError as exc:
            print(f"警告: {path} の読み出しに失敗しました: {exc}", file=sys.stderr)
    if not index_list:
        raise RuntimeError("有効な JSONL レコードが見つかりませんでした")
    index = np.asarray(index_list, dtype=np.int64).reshape(-1, 2)
    if cache_path is not None:
        _save_index_cache(cache_path, files, index)
    return files, index


def compute_file_hashes(files: Sequence[FileRef], *, show_progress: bool) -> Tuple[List[Dict[str, object]], str]:
    """入力ファイルごとの SHA-256 と結合ハッシュを計算する。"""

    total_bytes = sum(ref.size for ref in files)
    progress = ProgressReporter(total_bytes, "入力ハッシュ計算", unit="B", enable=show_progress, unit_scale=True)
    per_file: List[Dict[str, object]] = []
    dataset_hasher = hashlib.sha256()
    for ref in files:
        sha = hashlib.sha256()
        try:
            with ref.path.open("rb", buffering=1024 * 1024) as fp:
                while True:
                    chunk = fp.read(1024 * 1024)
                    if not chunk:
                        break
                    sha.update(chunk)
                    progress.update(len(chunk))
        except OSError as exc:
            progress.close()
            raise RuntimeError(f"{ref.path} のハッシュ計算に失敗しました: {exc}") from exc
        digest = sha.hexdigest()
        per_file.append(
            {
                "path": str(ref.path.resolve()),
                "size": int(ref.size),
                "mtime": float(ref.mtime),
                "sha256": digest,
            }
        )
        dataset_hasher.update(bytes.fromhex(digest))
        dataset_hasher.update(struct.pack("<Q", int(ref.size)))
        dataset_hasher.update(str(ref.path.resolve()).encode("utf-8"))
    progress.close()
    return per_file, dataset_hasher.hexdigest()


def _decode_json(data: bytes) -> Dict[str, object]:
    """JSON バイト列を辞書にデコードする。"""

    text = data.strip()
    if not text:
        return {}
    if _fast_json is not None:
        return _fast_json.loads(text)
    return json.loads(text.decode("utf-8"))


def _split_filter(code: int) -> Tuple[int, int, int]:
    """カラー相関フィルターを 3 要素に分解する。"""

    if code < 0:
        return -1, -1, -1
    perm = ((code >> 4) & 0x7) % 6
    primary = ((code >> 2) & 0x3) % 4
    secondary = (code & 0x3) % 4
    return perm, primary, secondary


def extract_labels(record: Dict[str, object]) -> Tuple[int, ...]:
    """JSON レコードから 12 個のラベル値を抽出する。"""

    def _resolve(entry: Dict[str, object], key: str) -> int:
        value = entry.get(key)
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return -1
        return iv

    best_raw = record.get("best") if isinstance(record, dict) else None
    if not isinstance(best_raw, dict):
        best_raw = {}
    second_raw = record.get("second") if isinstance(record, dict) else None
    if not isinstance(second_raw, dict):
        second_raw = {}

    best_filter = int(best_raw.get("filter", -1)) if "filter" in best_raw else -1
    second_filter = int(second_raw.get("filter", -1)) if "filter" in second_raw else -1

    best_perm, best_primary, best_secondary = _split_filter(best_filter)
    second_perm, second_primary, second_secondary = _split_filter(second_filter)

    best_values = (
        _resolve(best_raw, "predictor"),
        best_perm,
        best_primary,
        best_secondary,
        _resolve(best_raw, "reorder"),
        _resolve(best_raw, "interleave"),
    )
    second_values = (
        _resolve(second_raw, "predictor"),
        second_perm,
        second_primary,
        second_secondary,
        _resolve(second_raw, "reorder"),
        _resolve(second_raw, "interleave"),
    )
    values = []
    for val in best_values + second_values:
        if not isinstance(val, int):
            values.append(-1)
        else:
            if val < -32768 or val > 32767:
                values.append(-1)
            else:
                values.append(int(val))
    return tuple(values)


def run_extraction(
    files: Sequence[FileRef],
    index: np.ndarray,
    *,
    record_size: int,
    bin_path: Path,
    show_progress: bool,
    threads: int,
    skip_bad: bool,
) -> None:
    """並列デコードでラベルを抽出し、バイナリに書き出す。"""

    if record_size != EXPECTED_RECORD_SIZE:
        raise ValueError(f"レコードサイズ {record_size} は想定 {EXPECTED_RECORD_SIZE} と一致しません")

    reader = ThreadLocalJsonReader(files)
    total = index.shape[0]
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as fp:
        fp.truncate(record_size * total)
    progress = ProgressReporter(total, "ラベル書き出し", unit="件", enable=show_progress)

    flush_threshold = max(record_size * 8192, 4 * 1024 * 1024)
    written_since_flush = 0

    lock = threading.Lock()
    error_holder: List[BaseException] = []

    def worker(task: Tuple[int, int, int]) -> Tuple[int, Optional[Tuple[int, ...]]]:
        seq, file_id, offset = task
        try:
            raw = reader.read_line(file_id, offset)
            record = _decode_json(raw)
            values = extract_labels(record)
            return seq, values
        except Exception as exc:  # pragma: no cover - 実運用の異常系
            if skip_bad:
                return seq, tuple([-1] * 12)
            with lock:
                error_holder.append(exc)
            return seq, None

    with bin_path.open("r+b", buffering=0) as fp:
        def writer() -> Iterator[Tuple[int, int, int]]:
            for seq, (file_id, offset) in enumerate(index):
                yield seq, int(file_id), int(offset)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            iterator: Iterable[Tuple[int, Optional[Tuple[int, ...]]]]
            iterator = executor.map(worker, writer(), chunksize=64)
            for seq, values in iterator:
                if values is None:
                    break
                offset = seq * record_size
                data = STRUCT.pack(MAGIC, VERSION, 0, *values, 0)
                fp.seek(offset)
                fp.write(data)
                written_since_flush += record_size
                if written_since_flush >= flush_threshold:
                    fp.flush()
                    os.fsync(fp.fileno())
                    written_since_flush = 0
                progress.update(1)
                if error_holder:
                    break
        progress.close()
        if written_since_flush > 0:
            fp.flush()
            os.fsync(fp.fileno())
    if error_holder:
        raise error_holder[0]


def write_metadata(
    meta_path: Path,
    *,
    record_size: int,
    record_count: int,
    files_meta: Sequence[Dict[str, object]],
    dataset_hash: str,
    index_cache: Optional[Path],
) -> None:
    """メタデータ JSON を出力する。"""

    version = None
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        text = completed.stdout.strip()
        if text:
            version = text
    except Exception:  # pragma: no cover - git 不在時
        version = None
    if not version:
        version = "unknown"

    meta = {
        "schema": 1,
        "record_size": int(record_size),
        "record_count": int(record_count),
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "inputs": list(files_meta),
        "dataset_sha256": dataset_hash,
        "index_cache": {
            "path": str(index_cache) if index_cache else None,
            "exists": bool(index_cache and index_cache.exists()),
            "shape": [int(record_count), 2],
        },
        "tool_version": version,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI 引数を構文解析する。"""

    parser = argparse.ArgumentParser(description="TLG8 ラベル事前抽出ツール")
    parser.add_argument("inputs", nargs="+", help="JSONL ファイルまたはディレクトリ")
    parser.add_argument("--index-cache", type=Path, help="(file_id, offset) インデックスの保存先")
    parser.add_argument("--meta-out", type=Path, required=True, help="メタデータ JSON の出力先")
    parser.add_argument("--bin-out", type=Path, required=True, help="ラベルバイナリの出力先")
    parser.add_argument("--record-size", type=int, default=EXPECTED_RECORD_SIZE, help="レコードバイト数")
    parser.add_argument("--threads", type=int, default=1, help="JSON デコードワーカー数")
    parser.add_argument("--progress", action="store_true", help="進捗バーを表示")
    parser.add_argument("--skip-bad-records", action="store_true", help="壊れた行を -1 で埋めて継続する")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="進捗バーを無効化")
    parser.set_defaults(progress=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    inputs = discover_inputs(args.inputs)
    if not inputs:
        print("エラー: JSONL 入力が見つかりません", file=sys.stderr)
        return 1

    files, index = build_index(inputs, args.index_cache, show_progress=args.progress)
    files_meta, dataset_hash = compute_file_hashes(files, show_progress=args.progress)

    threads = max(1, int(args.threads))
    try:
        run_extraction(
            files,
            index,
            record_size=int(args.record_size),
            bin_path=args.bin_out,
            show_progress=args.progress,
            threads=threads,
            skip_bad=bool(args.skip_bad_records),
        )
    except Exception as exc:
        print(f"エラー: ラベル書き出しに失敗しました: {exc}", file=sys.stderr)
        return 1

    write_metadata(
        args.meta_out,
        record_size=int(args.record_size),
        record_count=int(index.shape[0]),
        files_meta=files_meta,
        dataset_hash=dataset_hash,
        index_cache=args.index_cache,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
