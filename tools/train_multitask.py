#!/usr/bin/env python3
"""TLG8 マルチタスク分類モデルの学習 CLI。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import mmap
import os
import random
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # 高速 JSON デコードが利用可能ならば切り替える
    import orjson as _fastjson  # type: ignore[import]
except Exception:  # pragma: no cover - orjson 未導入環境
    _fastjson = None  # type: ignore[assignment]


if _fastjson is not None:

    def _json_loads(raw: bytes) -> Dict[str, object]:
        """orjson を利用した高速 JSON デコード。"""

        return _fastjson.loads(raw)


else:

    def _json_loads(raw: bytes) -> Dict[str, object]:
        """標準 json によるフォールバック実装。"""

        return json.loads(raw.decode("utf-8"))

try:  # PyTorch を利用する場合にのみ必要
    import torch
    from torch.nn import functional as torch_F
except Exception:  # pragma: no cover - PyTorch 非導入環境
    torch = None  # type: ignore[assignment]
    torch_F = None  # type: ignore[assignment]

from multitask_model import (
    HEAD_ORDER,
    HEAD_SPECS,
    MultiTaskModel,
    TorchMultiTask,
    TrainConfig,
    build_soft_targets,
    compute_metrics,
    pick_device,
    predict_logits_batched,
    train_multitask_model,
)

MAX_COMPONENTS = 4
BLOCK_EDGE = 8
MAX_BLOCK_PIXELS = BLOCK_EDGE * BLOCK_EDGE

# ラベルキャッシュのレコード定義
LABEL_MAGIC = 0x4C424C38  # 'LBL8'
LABEL_VERSION = 1
LABEL_STRUCT = struct.Struct("<IHH12hI92x")
LABEL_RECORD_SIZE = LABEL_STRUCT.size
LABEL_FIELD_COUNT = 12
LABEL_FIELD_OFFSET = 8  # ヘッダー 8 バイト分の後にラベル本体が続く

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm が利用不可の場合のフォールバック
    tqdm = None


class _SimpleProgressBar:
    """tqdm 非導入環境向けの簡易プログレスバー。"""

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
    """tqdm を優先しつつ、利用不可時は簡易バーで進捗を表示するヘルパー。"""

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


# 進捗表示の有効 / 無効を CLI から切り替えるためのフラグ。
ENABLE_PROGRESS = True

# 遅延デコード用のスレッド上限と JSON デコード挙動を実行時に切り替えるためのグローバル。
MAX_DECODE_THREADS = 1
SKIP_BAD_RECORDS = False


def discover_input_files(inputs: Sequence[str]) -> List[Path]:
    """入力パスから学習データファイル一覧を収集する。"""

    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.rglob("*.jsonl")):
                if child.is_file():
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"警告: 入力パス '{item}' は存在しません", file=sys.stderr)
    return sorted({p.resolve() for p in files})


def split_dataset(count: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """データセットを訓練用と評価用に分割する。"""

    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    if count <= 1:
        return indices, indices
    test_count = max(1, int(round(count * test_ratio)))
    test_count = min(test_count, count - 1)
    train_count = count - test_count
    return indices[:train_count], indices[train_count:]


def _split_filter_code(code: int) -> Tuple[int, int, int]:
    """96 種類のカラー相関フィルターコードを 3 要素に分解する。"""

    if code < 0:
        return -1, -1, -1
    perm = ((code >> 4) & 0x7) % 6
    primary = ((code >> 2) & 0x3) % 4
    secondary = (code & 0x3) % 4
    return perm, primary, secondary


@dataclass
class FileRef:
    """JSONL ファイルへの参照情報。"""

    path: Path
    size: int
    mtime: float


def _inputs_match_meta(paths: Sequence[Path], meta: Dict[str, object]) -> bool:
    """キャッシュメタデータと入力ファイル群が一致するか検証する。"""

    files_meta = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files_meta, list) or len(files_meta) != len(paths):
        return False
    for path, entry in zip(paths, files_meta):
        if not isinstance(entry, dict):
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        recorded_path = entry.get("path")
        recorded_size = entry.get("size")
        recorded_mtime = entry.get("mtime")
        if str(path) != str(recorded_path):
            return False
        if int(stat.st_size) != int(recorded_size):
            return False
        try:
            meta_mtime = float(recorded_mtime)
        except (TypeError, ValueError):
            return False
        if not math.isclose(stat.st_mtime, meta_mtime, rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def _ensure_index_matrix(arr: np.ndarray) -> np.ndarray:
    """(N, 2) 形状の int64 配列に正規化する。"""

    index = np.asarray(arr, dtype=np.int64)
    if index.ndim == 1:
        if index.size % 2 != 0:
            raise ValueError("インデックス配列の長さが不正です")
        index = index.reshape(-1, 2)
    elif index.ndim != 2 or index.shape[1] != 2:
        index = index.reshape(-1, 2)
    return index


def build_jsonl_index(
    paths: Sequence[Path],
    cache_path: Optional[Path],
    show_progress: bool,
) -> Tuple[List[FileRef], np.ndarray]:
    """JSONL 群からレコード位置のインデックスを構築する。"""

    if cache_path is not None and cache_path.exists():
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        try:
            with meta_path.open("r", encoding="utf-8") as meta_fp:
                meta = json.load(meta_fp)
        except OSError:
            meta = None
        except json.JSONDecodeError:
            meta = None
        if meta and _inputs_match_meta(paths, meta):
            try:
                cached = np.load(cache_path, allow_pickle=False)
            except (OSError, ValueError):
                cached = None
            if cached is not None:
                files: List[FileRef] = []
                valid = True
                for item in meta.get("files", []):
                    if not isinstance(item, dict):
                        valid = False
                        break
                    try:
                        path_str = str(item["path"])
                        size_val = int(item["size"])
                        mtime_val = float(item["mtime"])
                    except (KeyError, TypeError, ValueError):
                        valid = False
                        break
                    files.append(FileRef(path=Path(path_str), size=size_val, mtime=mtime_val))
                if valid and len(files) == len(paths):
                    return files, _ensure_index_matrix(cached)

    files: List[FileRef] = []
    index_list: List[Tuple[int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
            size = stat.st_size
            files.append(FileRef(path=path, size=size, mtime=stat.st_mtime))
            file_id = len(files) - 1
            with path.open("rb", buffering=1024 * 1024) as fp:
                offset = fp.tell()
                last_pos = offset
                bar = ProgressReporter(
                    size,
                    f"{path.name}: インデックス構築中",
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
                    bar.update(offset - last_pos)
                    last_pos = offset
                if last_pos < size:
                    bar.update(size - last_pos)
                bar.close()
        except OSError as exc:
            print(f"警告: {path} を読み込めません: {exc}", file=sys.stderr)
    if not index_list:
        raise RuntimeError("学習データが読み込めませんでした")
    index = _ensure_index_matrix(index_list)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, index)
        meta = {
            "files": [
                {
                    "path": str(ref.path),
                    "size": int(ref.size),
                    "mtime": float(ref.mtime),
                }
                for ref in files
            ]
        }
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as meta_fp:
            json.dump(meta, meta_fp)
    return files, index


def compute_label_cache_hashes(
    files: Sequence[FileRef], *, show_progress: bool
) -> Tuple[List[Dict[str, object]], str]:
    """ラベルキャッシュ検証用に各入力の SHA-256 と結合ハッシュを計算する。"""

    total = sum(ref.size for ref in files)
    progress = ProgressReporter(
        total,
        "ラベルキャッシュ照合: ハッシュ計算",
        unit="B",
        enable=show_progress,
        unit_scale=True,
    )
    dataset_hasher = hashlib.sha256()
    results: List[Dict[str, object]] = []
    for ref in files:
        sha = hashlib.sha256()
        resolved = ref.path.resolve()
        try:
            with resolved.open("rb", buffering=1024 * 1024) as fp:
                while True:
                    chunk = fp.read(1024 * 1024)
                    if not chunk:
                        break
                    sha.update(chunk)
                    progress.update(len(chunk))
        except OSError as exc:
            progress.close()
            raise RuntimeError(f"{resolved} のハッシュ計算に失敗しました: {exc}") from exc
        digest = sha.hexdigest()
        results.append(
            {
                "path": str(resolved),
                "size": int(ref.size),
                "mtime": float(ref.mtime),
                "sha256": digest,
            }
        )
        dataset_hasher.update(bytes.fromhex(digest))
        dataset_hasher.update(struct.pack("<Q", int(ref.size)))
        dataset_hasher.update(str(resolved).encode("utf-8"))
    progress.close()
    return results, dataset_hasher.hexdigest()


def try_load_label_cache(
    meta_path: Path,
    bin_path: Path,
    files: Sequence[FileRef],
    expected_records: int,
    *,
    show_progress: bool,
) -> Optional[FastLabelStore]:
    """ラベルキャッシュが利用可能なら FastLabelStore を初期化する。"""

    if not meta_path.exists() or not bin_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("ラベルキャッシュメタデータを読み取れません: %s", exc)
        return None

    try:
        schema_val = int(meta.get("schema", 0))
    except (TypeError, ValueError):
        logging.warning("ラベルキャッシュのスキーマが解釈できません")
        return None
    if schema_val != 1:
        logging.warning("ラベルキャッシュのスキーマバージョンが一致しません")
        return None
    try:
        record_size = int(meta.get("record_size", 0))
        record_count = int(meta.get("record_count", 0))
    except (TypeError, ValueError):
        logging.warning("ラベルキャッシュメタデータのレコード情報が不正です")
        return None
    if record_size != LABEL_RECORD_SIZE:
        logging.warning("ラベルレコードサイズが想定値と異なるためキャッシュを無視します")
        return None
    if record_count != expected_records or record_count <= 0:
        logging.warning("ラベルキャッシュの件数が現在のデータセットと一致しません")
        return None

    inputs_meta = meta.get("inputs")
    if not isinstance(inputs_meta, list) or len(inputs_meta) != len(files):
        logging.warning("ラベルキャッシュの入力ファイル一覧が不正です")
        return None

    try:
        computed_meta, dataset_hash = compute_label_cache_hashes(files, show_progress=show_progress)
    except RuntimeError as exc:
        logging.warning("ラベルキャッシュ照合中にハッシュ計算へ失敗しました: %s", exc)
        return None

    for recorded, current in zip(inputs_meta, computed_meta):
        if not isinstance(recorded, dict):
            logging.warning("ラベルキャッシュの入力情報が壊れています")
            return None
        try:
            recorded_path = Path(recorded["path"]).resolve()
            recorded_size = int(recorded["size"])
            recorded_mtime = float(recorded["mtime"])
            recorded_sha = str(recorded["sha256"])
        except (KeyError, TypeError, ValueError):
            logging.warning("ラベルキャッシュの入力情報を解釈できません")
            return None
        if recorded_path != Path(current["path"]).resolve():
            logging.warning("ラベルキャッシュの入力パスが一致しません: %s", recorded_path)
            return None
        if recorded_size != int(current["size"]):
            logging.warning("ラベルキャッシュのサイズ情報が一致しません: %s", recorded_path)
            return None
        if not math.isclose(recorded_mtime, float(current["mtime"]), rel_tol=0.0, abs_tol=1e-6):
            logging.warning("ラベルキャッシュの更新時刻が一致しません: %s", recorded_path)
            return None
        if recorded_sha != str(current["sha256"]):
            logging.warning("ラベルキャッシュのハッシュが一致しません: %s", recorded_path)
            return None

    recorded_dataset_hash = str(meta.get("dataset_sha256", ""))
    if recorded_dataset_hash != dataset_hash:
        logging.warning("ラベルキャッシュの結合ハッシュが一致しません")
        return None

    try:
        size = bin_path.stat().st_size
    except OSError as exc:
        logging.warning("ラベルキャッシュのバイナリを確認できません: %s", exc)
        return None
    if size != record_size * record_count:
        logging.warning("ラベルキャッシュのファイルサイズが不一致です")
        return None

    try:
        store = FastLabelStore(bin_path, record_count, record_size)
    except Exception as exc:
        logging.warning("ラベルキャッシュの初期化に失敗しました: %s", exc)
        return None
    logging.info("ラベルキャッシュを利用します (%s)", bin_path)
    return store


def invoke_preextractor(
    inputs: Sequence[str],
    *,
    index_cache: Optional[Path],
    meta_out: Path,
    bin_out: Path,
    record_size: int,
    threads: int,
    progress: bool,
    skip_bad: bool,
) -> int:
    """preextract_labels.py をサブプロセスとして実行する。"""

    script = Path(__file__).with_name("preextract_labels.py")
    if not script.exists():
        logging.warning("preextract_labels.py が見つかりません")
        return 1
    cmd: List[str] = [sys.executable, str(script)]
    cmd.extend(str(item) for item in inputs)
    if index_cache is not None:
        cmd.extend(["--index-cache", str(index_cache)])
    cmd.extend(
        [
            "--meta-out",
            str(meta_out),
            "--bin-out",
            str(bin_out),
            "--record-size",
            str(int(record_size)),
            "--threads",
            str(max(1, int(threads))),
        ]
    )
    cmd.append("--progress" if progress else "--no-progress")
    if skip_bad:
        cmd.append("--skip-bad-records")
    logging.info("preextract_labels.py を起動します: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


class JsonlReader:
    """ファイル・オフセット指定で JSON レコードを読み取る補助クラス。"""

    def __init__(self, files: List[FileRef]) -> None:
        self._files = files
        # スレッドごとにファイルハンドルを分離してシーク競合を防ぐ。
        self._local = threading.local()

    def _handle(self, file_id: int) -> io.BufferedReader:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
        handle = handles.get(file_id)
        if handle is None or handle.closed:
            path = self._files[file_id].path
            handle = open(path, "rb", buffering=1024 * 1024)
            handles[file_id] = handle
        return handle

    def read_line(self, file_id: int, offset: int) -> Dict[str, object]:
        """指定位置の JSON レコードを辞書として返す。"""

        handle = self._handle(file_id)
        handle.seek(offset)
        raw = handle.readline()
        try:
            return _json_loads(raw.strip())
        except Exception:
            handle.seek(offset)
            raw_retry = handle.readline()
            try:
                return _json_loads(raw_retry.strip())
            except Exception as exc:
                if SKIP_BAD_RECORDS:
                    logging.warning(
                        "Skipping corrupt JSON at file_id=%d offset=%d",
                        file_id,
                        offset,
                    )
                    return {
                        "pixels": [],
                        "block_size": [0, 0],
                        "components": 0,
                        "best": {},
                        "second": {},
                    }
                raise exc

    def read(self, file_id: int, offset: int) -> Dict[str, object]:
        """後方互換用エイリアス。"""

        return self.read_line(file_id, offset)


def record_to_feature(record: Dict[str, object]) -> np.ndarray:
    """単一レコードから特徴量ベクトルを構築する。"""

    try:
        pixels = record["pixels"]  # type: ignore[index]
        block_w, block_h = record["block_size"]  # type: ignore[index]
        components = record["components"]  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"レコードの必須フィールドが不足しています: {exc}") from exc

    if not isinstance(pixels, list):
        raise ValueError("pixels フィールドが配列ではありません")
    block_w = int(block_w)
    block_h = int(block_h)
    components = int(components)
    expected_len = block_w * block_h * components
    if len(pixels) != expected_len:
        raise ValueError("pixels の長さがブロック構成と一致しません")

    padded = np.zeros((MAX_COMPONENTS, BLOCK_EDGE, BLOCK_EDGE), dtype=np.float32)
    arr = np.asarray(pixels, dtype=np.uint8, order="C")
    arr = arr.reshape(block_h, block_w, components).transpose(2, 0, 1)
    padded[:components, :block_h, :block_w] = arr / 255.0
    extra = np.array([block_w / 8.0, block_h / 8.0, components / 4.0], dtype=np.float32)
    feature = np.concatenate([padded.reshape(-1), extra])
    return feature


def _normalize_indices(idx: object, length: int) -> Tuple[np.ndarray, bool]:
    """インデックス指定を正規化し、スカラー指定かどうかを返す。"""

    if isinstance(idx, (int, np.integer)):
        return np.asarray([int(idx)], dtype=np.int64), True
    if isinstance(idx, slice):
        rng = range(*idx.indices(length))
        return np.fromiter(rng, dtype=np.int64), False
    if isinstance(idx, (list, tuple)):
        arr = np.asarray(idx, dtype=np.int64)
        return arr, False
    if isinstance(idx, np.ndarray):
        if idx.ndim != 1:
            raise TypeError("1 次元の整数インデックスのみ対応しています")
        return idx.astype(np.int64, copy=False), False
    raise TypeError("対応していないインデックス指定です")


class FastLabelStore:
    """ラベルバイナリを mmap して高速アクセス用ビューを提供する。"""

    def __init__(self, bin_path: Path, record_count: int, record_size: int) -> None:
        self._path = Path(bin_path)
        self._record_count = int(record_count)
        self._record_size = int(record_size)
        self._file = open(self._path, "rb")
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise
        try:
            mapped_size = self._mmap.size()
        except (AttributeError, OSError):  # pragma: no cover - プラットフォーム差異
            mapped_size = len(self._mmap)
        expected_size = self._record_count * self._record_size
        if mapped_size < expected_size:
            self.close()
            raise ValueError("ラベルバイナリのサイズがメタデータと一致しません")
        try:
            magic, version, _reserved = struct.unpack_from("<IHH", self._mmap, 0)
        except struct.error as exc:
            self.close()
            raise ValueError("ラベルバイナリのヘッダーを読み取れません") from exc
        if magic != LABEL_MAGIC or version != LABEL_VERSION:
            self.close()
            raise ValueError("ラベルバイナリのマジックまたはバージョンが不正です")
        self._fields: Dict[int, np.ndarray] = {}
        stride = self._record_size
        base = LABEL_FIELD_OFFSET
        for idx in range(LABEL_FIELD_COUNT):
            offset = base + idx * 2
            arr = np.ndarray(
                shape=(self._record_count,),
                dtype=np.int16,
                buffer=self._mmap,
                offset=offset,
                strides=(stride,),
            )
            self._fields[idx] = arr

    def field_array(self, idx: int) -> np.ndarray:
        return self._fields[idx]

    @property
    def record_count(self) -> int:
        return self._record_count

    def make(self, head: str, *, use_second: bool) -> "FastLabels":
        return FastLabels(self, head, use_second=use_second)

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            self._file.close()


class FastLabels:
    """FastLabelStore 上の単一ヘッドビュー。"""

    FIELD_INDEX = {
        ("predictor", False): 0,
        ("filter_perm", False): 1,
        ("filter_primary", False): 2,
        ("filter_secondary", False): 3,
        ("reorder", False): 4,
        ("interleave", False): 5,
        ("predictor", True): 6,
        ("filter_perm", True): 7,
        ("filter_primary", True): 8,
        ("filter_secondary", True): 9,
        ("reorder", True): 10,
        ("interleave", True): 11,
    }

    def __init__(self, store: FastLabelStore, head: str, *, use_second: bool) -> None:
        self._store = store
        try:
            self._field_idx = self.FIELD_INDEX[(head, use_second)]
        except KeyError as exc:
            raise KeyError(f"未知のヘッド名です: {head}") from exc

    def __len__(self) -> int:
        return self._store.record_count

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        field = self._store.field_array(self._field_idx)
        values = field[rows]
        out = values.astype(np.int32, copy=True)
        if scalar:
            return int(out[0])
        return out


class LazyFeatures:
    """JSONL からオンデマンドで特徴量を読み出す遅延配列。"""

    def __init__(
        self,
        reader: JsonlReader,
        index: np.ndarray,
        feature_dim: int,
        *,
        max_workers: int = 1,
    ) -> None:
        self._reader = reader
        self._index = index
        self._shape = (index.shape[0], feature_dim)
        self._max_workers = max(1, int(max_workers))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        batch = np.empty((rows.shape[0], self._shape[1]), dtype=np.float32)
        if self._max_workers <= 1 or rows.shape[0] <= 1:
            for out_idx, rec_idx in enumerate(rows):
                batch[out_idx] = self._read_and_pack(int(rec_idx))
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [
                    executor.submit(self._read_and_pack_with_index, out_idx, int(rec_idx))
                    for out_idx, rec_idx in enumerate(rows)
                ]
                for future in futures:
                    out_idx, feature = future.result()
                    batch[out_idx] = feature
        if scalar:
            return batch[0]
        return batch

    def _read_and_pack(self, rec_idx: int) -> np.ndarray:
        file_id, offset = self._index[int(rec_idx)]
        record = self._reader.read_line(int(file_id), int(offset))
        feature = record_to_feature(record)
        if feature.dtype != np.float32:
            feature = feature.astype(np.float32, copy=False)
        return feature

    def _read_and_pack_with_index(self, out_idx: int, rec_idx: int) -> Tuple[int, np.ndarray]:
        feature = self._read_and_pack(rec_idx)
        return out_idx, feature


class LazyLabels:
    """JSONL からオンデマンドでラベルを読み出す遅延ベクトル。"""

    def __init__(self, reader: JsonlReader, index: np.ndarray, head: str, *, use_second: bool = False) -> None:
        self._reader = reader
        self._index = index
        self._head = head
        self._use_second = use_second

    def __len__(self) -> int:
        return self._index.shape[0]

    def _resolve_value(self, record: Dict[str, object]) -> int:
        key = "second" if self._use_second else "best"
        entry = record.get(key, {})  # type: ignore[assignment]
        if not isinstance(entry, dict):
            entry = {}
        default_scalar = -1 if self._use_second else 0
        default_filter = -1 if self._use_second else 0
        if self._head == "predictor":
            return int(entry.get("predictor", default_scalar))
        if self._head == "reorder":
            return int(entry.get("reorder", default_scalar))
        if self._head == "interleave":
            return int(entry.get("interleave", default_scalar))
        if self._head in ("filter_perm", "filter_primary", "filter_secondary"):
            code = int(entry.get("filter", default_filter))
            perm, primary, secondary = _split_filter_code(code)
            mapping = {
                "filter_perm": perm,
                "filter_primary": primary,
                "filter_secondary": secondary,
            }
            return mapping[self._head]
        raise KeyError(f"未知のヘッド名です: {self._head}")

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        out = np.empty((rows.shape[0],), dtype=np.int32)
        desc = "best" if not self._use_second else "second"
        progress = ProgressReporter(
            rows.shape[0],
            f"{self._head}({desc}) ラベル読み出し",
            unit="サンプル",
            enable=ENABLE_PROGRESS and rows.shape[0] >= 1024,
        )
        pending = 0
        for out_idx, rec_idx in enumerate(rows):
            file_id, offset = self._index[int(rec_idx)]
            record = self._reader.read_line(int(file_id), int(offset))
            out[out_idx] = self._resolve_value(record)
            pending += 1
            if pending >= 256:
                progress.update(pending)
                pending = 0
        if pending:
            progress.update(pending)
        progress.close()
        if scalar:
            return out[0]
        return out


class LazyFeatureSubset:
    """LazyFeatures の部分集合ビュー。"""

    def __init__(self, base: LazyFeatures, indices: np.ndarray) -> None:
        self._base = base
        self._indices = np.asarray(indices, dtype=np.int64)
        self._shape = (self._indices.shape[0], base.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._indices.shape[0]

    def __getitem__(self, idx: object) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            base_idx = self._indices[int(idx)]
            return self._base[int(base_idx)]
        rows, scalar = _normalize_indices(idx, len(self))
        base_rows = self._indices[rows]
        batch = self._base[base_rows]
        if scalar:
            if isinstance(batch, np.ndarray) and batch.ndim == 2 and batch.shape[0] == 1:
                return batch[0]
        return batch


def open_indexed_dataset(
    paths: Sequence[Path],
    cache_path: Optional[Path],
    show_progress: bool,
) -> Tuple[LazyFeatures, Dict[str, LazyLabels], Dict[str, LazyLabels], int]:
    """JSONL データセットをインデックス化し、遅延ビューを返す。"""

    files, index = build_jsonl_index(paths, cache_path, show_progress)
    reader = JsonlReader(files)
    sample = reader.read_line(int(index[0, 0]), int(index[0, 1]))
    feature_dim = record_to_feature(sample).shape[0]
    features = LazyFeatures(reader, index, feature_dim, max_workers=MAX_DECODE_THREADS)
    best_labels = {name: LazyLabels(reader, index, name, use_second=False) for name in HEAD_ORDER}
    second_labels = {name: LazyLabels(reader, index, name, use_second=True) for name in HEAD_ORDER}
    return features, best_labels, second_labels, index.shape[0]


def compute_feature_scaler_streaming(
    features: LazyFeatures,
    train_idx: np.ndarray,
    batch_size: int = 32768,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """訓練データの遅延読み出しで平均・標準偏差を算出する。"""

    feat_dim = features.shape[1]
    total_sum = np.zeros((feat_dim,), dtype=np.float64)
    total_sumsq = np.zeros((feat_dim,), dtype=np.float64)
    total_count = 0
    progress = ProgressReporter(
        int(train_idx.shape[0]), "特徴量スケーラー算出中", unit="サンプル", enable=show_progress
    )
    for start in range(0, train_idx.shape[0], batch_size):
        batch_indices = train_idx[start : start + batch_size]
        batch = np.asarray(features[batch_indices], dtype=np.float64, order="C")
        total_sum += batch.sum(axis=0)
        total_sumsq += np.square(batch).sum(axis=0)
        total_count += batch.shape[0]
        progress.update(batch.shape[0])
    progress.close()
    if total_count <= 1:
        variance = np.ones_like(total_sum)
    else:
        variance = (total_sumsq - (total_sum ** 2) / total_count) / (total_count - 1)
    mean_fp64 = total_sum / max(total_count, 1)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean = mean_fp64.astype(np.float32)
    logging.info(
        "Scaler: L2(mean)=%.6f, min(std)=%.6f, max(std)=%.6f",
        float(np.linalg.norm(mean_fp64)),
        float(std.min()),
        float(std.max()),
    )
    return mean, std


def slice_labels_lazy(labels: Dict[str, object], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """遅延ラベルから指定インデックスの値をまとめて抽出する。"""

    return {name: lazy[indices] for name, lazy in labels.items()}


def predict_logits_with_progress(
    model: "torch.nn.Module",
    features: object,
    device: "torch.device",
    *,
    batch_size: int,
    amp: str,
    desc: str,
) -> Dict[str, np.ndarray]:
    """predict_logits_batched にプログレスバーを付与したラッパー。"""

    if not ENABLE_PROGRESS:
        return predict_logits_batched(model, features, device, batch_size=batch_size, amp=amp)

    if isinstance(features, np.ndarray):
        total = int(features.shape[0]) if features.ndim > 1 else 1
    else:
        total = int(getattr(features, "shape")[0])  # type: ignore[index]

    progress = ProgressReporter(total, desc, unit="サンプル", enable=ENABLE_PROGRESS)
    if total <= 0:
        progress.close()
        return predict_logits_batched(model, features, device, batch_size=batch_size, amp=amp)

    if isinstance(features, np.ndarray):
        try:
            return predict_logits_batched(model, features, device, batch_size=batch_size, amp=amp)
        finally:
            progress.update(total)
            progress.close()

    total_local = total

    class _ProgressiveFeatures:
        """特徴量読み出し時に進捗を更新する薄いラッパー。"""

        def __init__(self, base: object, reporter: ProgressReporter) -> None:
            self._base = base
            self._reporter = reporter
            self.shape = getattr(base, "shape")

        def __len__(self) -> int:
            if hasattr(self._base, "__len__"):
                return int(len(self._base))  # type: ignore[arg-type]
            return total_local

        def __getitem__(self, item: object) -> np.ndarray:
            result = self._base[item]
            amount = 0
            if isinstance(item, slice):
                start = 0 if item.start is None else int(item.start)
                stop = start if item.stop is None else int(item.stop)
                amount = max(0, stop - start)
            elif isinstance(item, np.ndarray):
                amount = int(item.shape[0])
            elif isinstance(item, Sequence):
                amount = len(item)
            elif isinstance(result, np.ndarray):
                amount = int(result.shape[0]) if result.ndim > 0 else 1
            else:
                amount = 1
            if amount > 0:
                self._reporter.update(amount)
            return result

    wrapped = _ProgressiveFeatures(features, progress)
    try:
        return predict_logits_batched(model, wrapped, device, batch_size=batch_size, amp=amp)
    finally:
        progress.close()


def format_metrics(metrics: Dict[str, float]) -> str:
    """単一ヘッドの指標を整形して返す。"""

    top1 = metrics.get("top1", 0.0) * 100.0
    top2 = metrics.get("top2", 0.0) * 100.0
    top3 = metrics.get("top3", 0.0) * 100.0
    three = metrics.get("three_choice", 0.0) * 100.0
    return f"top1={top1:.2f}% top2={top2:.2f}% top3={top3:.2f}% three={three:.2f}%"


def macro_average(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """ヘッドごとの指標辞書からマクロ平均を計算する。"""

    result: Dict[str, float] = {}
    if not metrics:
        return result
    for key in ("top1", "top2", "top3", "three_choice"):
        values = [metrics[name].get(key, float("nan")) for name in HEAD_ORDER]
        result[key] = float(np.nanmean(values))
    return result


def _amp_context(device: "torch.device", amp_mode: str):
    """AMP 設定に応じて適切なコンテキストを返す。"""

    if torch is None or amp_mode.lower() != "bf16":
        return nullcontext()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.amp.autocast(dtype=torch.bfloat16)  # type: ignore[attr-defined]
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_with_torch_backend(
    args: argparse.Namespace,
    train_features: "LazyFeatureSubset",
    train_best: Dict[str, np.ndarray],
    train_second: Dict[str, np.ndarray],
    val_features: "LazyFeatureSubset",
    val_best: Dict[str, np.ndarray],
    val_second: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[TorchMultiTask, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """PyTorch バックエンドでマルチタスクモデルを学習する。"""

    if torch is None or torch_F is None:
        raise ImportError("PyTorch が利用できません。'pip install torch torchvision torchaudio intel-extension-for-pytorch' を確認してください")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if hasattr(torch, "xpu"):
        try:
            torch.xpu.manual_seed(args.seed)  # type: ignore[attr-defined]
        except Exception:
            pass

    device = pick_device(args.device)
    amp_mode = args.amp.lower()
    if amp_mode == "bf16" and device.type == "cpu":
        if not hasattr(torch, "cpu") or not getattr(torch.cpu, "is_bf16_supported", lambda: False)():  # type: ignore[attr-defined]
            logging.warning("CPU が bfloat16 をサポートしていないため AMP を無効化します")
            amp_mode = "none"

    model = TorchMultiTask(mean, std, dropout=args.dropout)
    model.inference_temperature = max(float(args.temperature), 1e-6)
    model.to(device)

    model_forward = model
    if args.compile:
        if hasattr(torch, "compile"):
            try:
                model_forward = torch.compile(model)  # type: ignore[assignment]
            except Exception as exc:
                logging.warning("torch.compile に失敗したため未コンパイルで継続します: %s", exc)
                model_forward = model
        else:
            logging.warning("この PyTorch では torch.compile が利用できないため通常モードで実行します")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_targets: Dict[str, np.ndarray] = {}
    for name in HEAD_ORDER:
        train_targets[name] = build_soft_targets(
            train_best[name],
            train_second[name],
            HEAD_SPECS[name],
            args.epsilon_soft,
        ).astype(np.float32)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_train_metrics: Dict[str, Dict[str, float]] = {}
    best_val_metrics: Dict[str, Dict[str, float]] = {}
    best_metric = -np.inf
    patience_counter = 0
    train_count = len(train_features)
    rng = np.random.default_rng(args.seed)
    total_batches = max(1, (train_count + args.batch_size - 1) // args.batch_size)
    digits = len(str(total_batches))
    bar_width = 30

    for epoch in range(args.epochs):
        indices = rng.permutation(train_count)
        progress_rendered = False

        def report_progress(batch_idx: int, total: int) -> None:
            nonlocal progress_rendered
            if not ENABLE_PROGRESS:
                return
            progress_rendered = True
            ratio = min(max(batch_idx / total, 0.0), 1.0)
            filled = min(bar_width, max(0, int(round(ratio * bar_width))))
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\rエポック {epoch + 1:03d} {batch_idx:>{digits}}/{total} [{bar}] {ratio * 100:6.2f}%"
            )
            sys.stdout.flush()

        model.train()
        model_forward.train()
        epoch_loss = 0.0
        sample_seen = 0
        for batch_no, start in enumerate(range(0, train_count, args.batch_size), start=1):
            end = min(train_count, start + args.batch_size)
            batch_idx = indices[start:end]
            features_np = np.asarray(train_features[batch_idx], dtype=np.float32)
            xb = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
            target_tensors: Dict[str, torch.Tensor] = {}
            for name in HEAD_ORDER:
                probs = train_targets[name][batch_idx]
                target_tensors[name] = torch.from_numpy(probs).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp_mode):
                logits = model_forward(xb)
                loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
                for name in HEAD_ORDER:
                    log_probs = torch_F.log_softmax(logits[name], dim=1)
                    loss_tensor = loss_tensor + torch_F.kl_div(
                        log_probs, target_tensors[name], reduction="batchmean"
                    )
            loss_tensor.backward()
            optimizer.step()
            batch_size_actual = batch_idx.shape[0]
            epoch_loss += float(loss_tensor.detach().cpu()) * batch_size_actual
            sample_seen += batch_size_actual
            report_progress(batch_no, total_batches)

        if progress_rendered:
            sys.stdout.write("\n")

        mean_loss = epoch_loss / max(1, sample_seen)

        model_forward.eval()
        model.eval()
        eval_batch = min(args.max_batch, 8192)
        train_logits = predict_logits_with_progress(
            model_forward,
            train_features,
            device,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="訓練ロジット算出中",
        )
        val_logits = predict_logits_with_progress(
            model_forward,
            val_features,
            device,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="評価ロジット算出中",
        )

        train_metrics: Dict[str, Dict[str, float]] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for name in HEAD_ORDER:
            train_metrics[name] = compute_metrics(train_logits[name], train_best[name], train_second[name])
            val_metrics[name] = compute_metrics(val_logits[name], val_best[name], val_second[name])

        mean_top3 = float(np.mean([val_metrics[name]["top3"] for name in HEAD_ORDER]))
        mean_three_choice = float(np.mean([val_metrics[name]["three_choice"] for name in HEAD_ORDER]))
        mean_top1 = float(np.mean([val_metrics[name]["top1"] for name in HEAD_ORDER]))
        mean_top2 = float(np.mean([val_metrics[name]["top2"] for name in HEAD_ORDER]))
        print(
            "epoch {epoch:03d}: loss={loss:.4f} val_top3={top3:.2f}% val_three={three:.2f}% val_top1={top1:.2f}% val_top2={top2:.2f}%".format(
                epoch=epoch + 1,
                loss=mean_loss,
                top3=mean_top3 * 100.0,
                three=mean_three_choice * 100.0,
                top1=mean_top1 * 100.0,
                top2=mean_top2 * 100.0,
            )
        )

        if mean_top3 > best_metric + 1e-6:
            best_metric = mean_top3
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("早期終了: 改善が見られませんでした")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if model_forward is not model:
            model_forward.load_state_dict(best_state)

    return model, best_train_metrics, best_val_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLG8 マルチタスク分類モデル学習ツール")
    parser.add_argument("inputs", nargs="+", help="JSONL 形式の学習データまたはディレクトリ")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch", help="学習バックエンド")
    parser.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正則化係数")
    parser.add_argument("--batch-size", type=int, default=512, help="ミニバッチサイズ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[1024, 512, 256], help="(NumPy バックエンド用) 隠れ層ユニット数")
    parser.add_argument("--epsilon-soft", type=float, default=0.2, help="第 2 候補に割り当てる確率質量")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="評価データ比率")
    parser.add_argument("--patience", type=int, default=20, help="早期終了の待機エポック数")
    parser.add_argument("--export-dir", type=Path, help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--temperature", type=float, default=1.0, help="推論時の温度パラメータ")
    parser.add_argument("--index-cache", type=str, default=None, help="(file_id, offset) を格納する NPY キャッシュパス")
    parser.add_argument("--labels-meta", type=Path, default=Path(".cache/labels.meta.json"), help="ラベルキャッシュのメタデータパス")
    parser.add_argument("--labels-bin", type=Path, default=Path(".cache/labels.bin"), help="ラベルキャッシュのバイナリパス")
    parser.add_argument("--build-label-cache", action="store_true", help="キャッシュ不整合時に事前抽出を自動実行")
    parser.add_argument("--no-label-cache", action="store_true", help="JSONL からのラベル読み出しを強制")
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="重い処理でプログレスバーを表示 (デフォルト ON)",
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="プログレスバーを無効化")
    parser.add_argument("--skip-bad-records", action="store_true", help="JSON デコードに失敗した行をスキップ")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8192,
        help="遅延読み出し時の実効ミニバッチ上限",
    )
    parser.add_argument(
        "--scaler-batch",
        type=int,
        default=32768,
        help="特徴量スケーラー算出時のバッチサイズ",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=0,
        help="遅延デコードに利用する最大スレッド数 (0 で自動設定)",
    )
    parser.add_argument("--device", choices=["xpu", "cpu"], default="xpu", help="PyTorch バックエンド用デバイス優先度")
    parser.add_argument(
        "--amp",
        choices=["bf16", "none"],
        default=None,
        help="PyTorch バックエンドで利用する自動混合精度",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile を試行する")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    backend = args.backend.lower()
    if args.amp is None:
        args.amp = "bf16" if backend == "torch" and args.device == "xpu" else "none"

    if args.max_batch <= 0:
        print("エラー: --max-batch には 1 以上を指定してください", file=sys.stderr)
        return 1

    if args.scaler_batch <= 0:
        print("エラー: --scaler-batch には 1 以上を指定してください", file=sys.stderr)
        return 1

    global MAX_DECODE_THREADS, SKIP_BAD_RECORDS, ENABLE_PROGRESS
    ENABLE_PROGRESS = bool(args.progress)
    SKIP_BAD_RECORDS = bool(args.skip_bad_records)
    threads = int(args.max_threads)
    if threads <= 0:
        threads = max(1, min(8, os.cpu_count() or 1))
    MAX_DECODE_THREADS = max(1, threads)

    # 代表的な数値演算ライブラリ向けにスレッド数を環境変数で制御する。
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = str(MAX_DECODE_THREADS)
    # NumPy 自身がスレッド制御 API を提供していれば併用する。
    if hasattr(np, "set_num_threads"):
        try:
            np.set_num_threads(MAX_DECODE_THREADS)
        except Exception:
            pass
    try:
        import mkl  # type: ignore

        try:
            mkl.set_num_threads(MAX_DECODE_THREADS)
        except Exception:
            pass
    except ImportError:
        pass

    input_paths = discover_input_files(args.inputs)
    if not input_paths:
        print("エラー: 入力ファイルが見つかりません", file=sys.stderr)
        return 1

    cache_path = Path(args.index_cache) if args.index_cache else None
    file_refs, index = build_jsonl_index(input_paths, cache_path, bool(args.progress))

    fast_label_store: Optional[FastLabelStore] = None
    if not args.no_label_cache:
        fast_label_store = try_load_label_cache(
            args.labels_meta,
            args.labels_bin,
            file_refs,
            index.shape[0],
            show_progress=bool(args.progress),
        )
        if fast_label_store is None and args.build_label_cache:
            rc = invoke_preextractor(
                args.inputs,
                index_cache=cache_path,
                meta_out=args.labels_meta,
                bin_out=args.labels_bin,
                record_size=LABEL_RECORD_SIZE,
                threads=MAX_DECODE_THREADS,
                progress=bool(args.progress),
                skip_bad=bool(args.skip_bad_records),
            )
            if rc == 0:
                fast_label_store = try_load_label_cache(
                    args.labels_meta,
                    args.labels_bin,
                    file_refs,
                    index.shape[0],
                    show_progress=bool(args.progress),
                )
            else:
                logging.warning("ラベルキャッシュ生成に失敗しました (exit=%d)", rc)

    reader = JsonlReader(file_refs)
    sample = reader.read_line(int(index[0, 0]), int(index[0, 1]))
    feature_dim = record_to_feature(sample).shape[0]
    features = LazyFeatures(reader, index, feature_dim, max_workers=MAX_DECODE_THREADS)

    if fast_label_store is not None:
        best_labels = {name: fast_label_store.make(name, use_second=False) for name in HEAD_ORDER}
        second_labels = {name: fast_label_store.make(name, use_second=True) for name in HEAD_ORDER}
    else:
        best_labels = {name: LazyLabels(reader, index, name, use_second=False) for name in HEAD_ORDER}
        second_labels = {name: LazyLabels(reader, index, name, use_second=True) for name in HEAD_ORDER}

    total = int(index.shape[0])
    train_idx, val_idx = split_dataset(total, args.test_ratio, args.seed)

    mean, std = compute_feature_scaler_streaming(
        features,
        train_idx,
        batch_size=int(args.scaler_batch),
        show_progress=ENABLE_PROGRESS,
    )
    train_features = LazyFeatureSubset(features, train_idx)
    val_features = LazyFeatureSubset(features, val_idx)
    train_best = slice_labels_lazy(best_labels, train_idx)
    train_second = slice_labels_lazy(second_labels, train_idx)
    val_best = slice_labels_lazy(best_labels, val_idx)
    val_second = slice_labels_lazy(second_labels, val_idx)

    args.batch_size = min(int(args.batch_size), int(args.max_batch))

    if backend == "numpy":
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            epsilon_soft=args.epsilon_soft,
            patience=args.patience,
            seed=args.seed,
        )

        model = MultiTaskModel(features.shape[1], args.hidden_dims, args.dropout, mean, std)
        model.inference_temperature = max(float(args.temperature), 1e-6)
        model, train_metrics, val_metrics = train_multitask_model(
            model,
            train_features,
            train_best,
            train_second,
            val_features,
            val_best,
            val_second,
            config,
        )

        print("\n=== 訓練データ精度 ===")
        for name in HEAD_ORDER:
            print(f"{name:16s}: {format_metrics(train_metrics.get(name, {}))}")
        train_macro = macro_average(train_metrics)
        if train_macro:
            print(
                f"{'macro':16s}: top1={train_macro['top1']*100:.2f}% "
                f"top2={train_macro['top2']*100:.2f}% top3={train_macro['top3']*100:.2f}% "
                f"three={train_macro['three_choice']*100:.2f}%"
            )

        print("\n=== 評価データ精度 ===")
        for name in HEAD_ORDER:
            print(f"{name:16s}: {format_metrics(val_metrics.get(name, {}))}")
        val_macro = macro_average(val_metrics)
        if val_macro:
            print(
                f"{'macro':16s}: top1={val_macro['top1']*100:.2f}% "
                f"top2={val_macro['top2']*100:.2f}% top3={val_macro['top3']*100:.2f}% "
                f"three={val_macro['three_choice']*100:.2f}%"
            )

        if args.export_dir:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.export_dir / "multitask_model.npz"
            model.save(str(out_path))
            print(f"\nモデルを {out_path} に保存しました。")
    else:
        try:
            torch_model, train_metrics, val_metrics = train_with_torch_backend(
                args,
                train_features,
                train_best,
                train_second,
                val_features,
                val_best,
                val_second,
                mean,
                std,
            )
        except ImportError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1

        print("\n=== 訓練データ精度 ===")
        for name in HEAD_ORDER:
            print(f"{name:16s}: {format_metrics(train_metrics.get(name, {}))}")
        train_macro = macro_average(train_metrics)
        if train_macro:
            print(
                f"{'macro':16s}: top1={train_macro['top1']*100:.2f}% "
                f"top2={train_macro['top2']*100:.2f}% top3={train_macro['top3']*100:.2f}% "
                f"three={train_macro['three_choice']*100:.2f}%"
            )

        print("\n=== 評価データ精度 ===")
        for name in HEAD_ORDER:
            print(f"{name:16s}: {format_metrics(val_metrics.get(name, {}))}")
        val_macro = macro_average(val_metrics)
        if val_macro:
            print(
                f"{'macro':16s}: top1={val_macro['top1']*100:.2f}% "
                f"top2={val_macro['top2']*100:.2f}% top3={val_macro['top3']*100:.2f}% "
                f"three={val_macro['three_choice']*100:.2f}%"
            )

        if args.export_dir:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.export_dir / "multitask_best.pt"
            torch_model.save_torch(str(out_path))
            print(f"\nPyTorch モデルを {out_path} に保存しました。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
