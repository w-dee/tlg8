#!/usr/bin/env python3
"""TLG8 マルチタスク分類モデルの学習 CLI。"""

from __future__ import annotations

import argparse
import inspect
import atexit
import hashlib
import io
import json
import logging
import math
import mmap
import os
import queue
import random
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAST_LABELCACHE_TOOL = PROJECT_ROOT / "build" / "tlg8_labelcache"
FAST_FEATCACHE_TOOL = PROJECT_ROOT / "build" / "tlg8_featcache"
RANKER_DEFAULT_HEADS = ["predictor", "reorder", "interleave", "filter_primary", "filter_secondary"]
RANKER_DEFAULT_TOPK = 2

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

    # Ampere 以降の TF32 制御を明示し、高速な行列演算を常時有効化する
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
except Exception:  # pragma: no cover - PyTorch 非導入環境
    torch = None  # type: ignore[assignment]
    torch_F = None  # type: ignore[assignment]

from features import compute_orientation_features
from io_utils import find_features_path, sha256_file
from losses import FocalLoss
from make_feature_stats import compute_feature_mean_std, load_feature_matrix
from multitask_model import (
    HEAD_ORDER,
    HEAD_SPECS,
    MultiTaskModel,
    TorchMultiTask,
    TrainConfig,
    build_soft_targets,
    compute_metrics,
    conditioned_extra_dim,
    _deterministic_topk_indices,
    pick_device,
    predict_logits_batched,
    train_multitask_model,
)

MAX_COMPONENTS = 4
BLOCK_EDGE = 8
MAX_BLOCK_PIXELS = BLOCK_EDGE * BLOCK_EDGE


def _str_to_bool(raw: str) -> bool:
    """CLI 文字列を真偽値へ変換する。"""

    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"真偽値に変換できない文字列です: {raw}")

# ラベルキャッシュのレコード定義
LABEL_MAGIC = 0x4C424C38  # 'LBL8'
LABEL_VERSION = 1
LABEL_STRUCT = struct.Struct("<IHH12hI92x")
LABEL_RECORD_SIZE = LABEL_STRUCT.size
LABEL_FIELD_COUNT = 12
LABEL_FIELD_OFFSET = 8  # ヘッダー 8 バイト分の後にラベル本体が続く

RANKER_LABEL_HEADER_STRUCT = struct.Struct("<8sIIQQQQ")
RANKER_LABEL_HEAD_STRUCT = struct.Struct("<IIII")
RANKER_LABEL_MAGIC = b"TLG8LBL\0"
RANKER_TOPK_MAGIC = b"TLG8TK\0"

FEATURE_STATS_MAGIC = b"FSC8"
FEATURE_STATS_VERSION = 1
FEATURE_STATS_HEADER = struct.Struct("<4sIIQ")


class FeatureStatsDimensionError(ValueError):
    """特徴量統計の次元不一致を通知する例外。"""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"特徴量統計の次元数が入力データと一致しません (expected={expected}, actual={actual})"
        )
        self.expected = int(expected)
        self.actual = int(actual)


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


def parse_condition_head_list(raw: str) -> Tuple[str, ...]:
    """条件付き入力として利用するヘッド一覧を解析する。"""

    if not raw:
        return ()
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    if not entries:
        return ()
    allowed = set(HEAD_ORDER)
    result: List[str] = []
    seen: set[str] = set()
    for name in entries:
        if name not in allowed:
            raise ValueError(f"未知のヘッド名が指定されました: {name}")
        if name in seen:
            continue
        result.append(name)
        seen.add(name)
    return tuple(result)


def parse_head_loss_weights(raw: str) -> Dict[str, float]:
    """ヘッド別損失重み指定文字列を辞書に変換する。"""

    if not raw:
        return {}
    weights: Dict[str, float] = {}
    for token in raw.split(","):
        entry = token.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"不正なヘッド損失指定です: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"ヘッド損失重みの数値が不正です: {entry}") from exc
    return weights


def parse_head_list(raw: str, *, allow_default: bool = False) -> Tuple[str, ...]:
    """ヘッド名のカンマ区切り文字列を正規化する。"""

    if not raw:
        return ()
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    if not entries:
        return ()
    allowed = set(HEAD_ORDER)
    result: List[str] = []
    seen: set[str] = set()
    for name in entries:
        if allow_default and name == "default":
            if name not in seen:
                result.append(name)
                seen.add(name)
            continue
        if name not in allowed:
            raise ValueError(f"未知のヘッド名が指定されました: {name}")
        if name in seen:
            continue
        result.append(name)
        seen.add(name)
    return tuple(result)


def parse_head_value_map(raw: str, *, allow_default: bool = False) -> Dict[str, float]:
    """ヘッド別の浮動小数点指定を辞書化する。"""

    if not raw:
        return {}
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    if not entries:
        return {}
    allowed = set(HEAD_ORDER)
    values: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"不正なヘッド指定です: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if allow_default and key == "default":
            values[key] = float(value)
            continue
        if key not in allowed:
            raise ValueError(f"未知のヘッド名が指定されました: {key}")
        values[key] = float(value)
    return values


def parse_optional_float(value: object) -> float | None:
    """None 指定を許容する float 変換。"""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("none", "null", ""):
        return None
    return float(text)


def effective_class_weights(counts: np.ndarray, beta: float) -> "torch.Tensor":
    """有効サンプル数に基づくクラス重みを算出する。"""

    if torch is None:
        raise ImportError("PyTorch が必要です")
    counts64 = np.asarray(counts, dtype=np.float64).clip(min=1.0)
    beta = float(beta)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts64))
    weights = weights * (counts64.size / np.sum(weights))
    return torch.from_numpy(weights.astype(np.float32))


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


def _load_ranker_cache_from_meta(
    meta_path: Path,
    meta: Dict[str, object],
    bin_path: Path,
    files: Sequence[FileRef],
    expected_records: int,
) -> RankerLabelCache:
    """C++ フォーマットのラベルキャッシュをメタ情報から初期化する。"""

    try:
        record_count = int(meta.get("n_samples", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("ラベルキャッシュメタデータのサンプル数が解釈できません") from exc
    if record_count != expected_records:
        raise ValueError("ラベルキャッシュの件数が現在のデータセットと一致しません")
    if not _feature_cache_files_match(files, meta.get("source_files")):
        raise ValueError("ラベルキャッシュの入力ファイル情報が一致しません")
    topk_info = meta.get("topk")
    topk_path: Optional[Path] = None
    topk_k = 0
    if isinstance(topk_info, dict):
        try:
            topk_k = int(topk_info.get("k", 0))
        except (TypeError, ValueError):
            topk_k = 0
        path_val = topk_info.get("path")
        if isinstance(path_val, str) and path_val:
            candidate = Path(path_val)
            if not candidate.is_absolute():
                candidate = meta_path.parent / candidate
            topk_path = candidate
    return RankerLabelCache(meta, bin_path, topk_path=topk_path, topk_k=topk_k)


def try_load_label_cache(
    meta_path: Path,
    bin_path: Path,
    files: Sequence[FileRef],
    expected_records: int,
    *,
    show_progress: bool,
) -> Optional[object]:
    """ラベルキャッシュが利用可能なら FastLabelStore を初期化する。"""

    if not meta_path.exists() or not bin_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("ラベルキャッシュメタデータを読み取れません: %s", exc)
        return None

    format_val = meta.get("format")
    if isinstance(format_val, str) and format_val == "tlg8-ranker-labels":
        try:
            cache = _load_ranker_cache_from_meta(meta_path, meta, bin_path, files, expected_records)
        except Exception as exc:
            logging.warning("C++ ラベルキャッシュの読み込みに失敗しました: %s", exc)
            return None
        logging.info("C++ ラベルキャッシュを利用します (%s)", bin_path)
        return cache

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


def ensure_ranker_label_cache_cpp(
    tool_path: Path,
    inputs: Sequence[Path],
    *,
    meta_path: Path,
    bin_path: Path,
    topk_path: Path,
    heads: Sequence[str],
    topk: int,
) -> None:
    """C++ 製ラベルキャッシュを生成するユーティリティ。"""

    if not tool_path.exists():
        raise FileNotFoundError(f"tlg8_labelcache が見つかりません: {tool_path}")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if topk > 0:
        topk_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [str(tool_path)]
    for path in inputs:
        cmd.extend(["--jsonl", str(path)])
    cmd.extend(["--out-bin", str(bin_path), "--out-meta", str(meta_path)])
    if heads:
        cmd.extend(["--heads", ",".join(heads)])
    if topk > 0:
        cmd.extend(["--out-topk", str(topk_path), "--topk", str(int(topk))])
    logging.info("tlg8_labelcache を起動します: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tlg8_labelcache の実行に失敗しました (exit={result.returncode})")


def ensure_ranker_feature_cache_cpp(
    tool_path: Path,
    inputs: Sequence[Path],
    *,
    out_npy: Path,
    out_scaler: Path,
    out_meta: Path,
    out_idx: Optional[Path] = None,
) -> None:
    """C++ 製特徴量キャッシュを生成するユーティリティ。"""

    if not tool_path.exists():
        raise FileNotFoundError(f"tlg8_featcache が見つかりません: {tool_path}")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_idx is not None:
        out_idx.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [str(tool_path)]
    for path in inputs:
        cmd.extend(["--jsonl", str(path)])
    cmd.extend(["--out-npy", str(out_npy), "--out-scaler", str(out_scaler), "--out-meta", str(out_meta)])
    if out_idx is not None:
        cmd.extend(["--out-idx", str(out_idx)])
    logging.info("tlg8_featcache を起動します: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tlg8_featcache の実行に失敗しました (exit={result.returncode})")


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
    normalized = arr.astype(np.float32) / 255.0
    padded[:components, :block_h, :block_w] = normalized
    orientation = compute_orientation_features(padded[:components, :block_h, :block_w])
    extra = np.array([block_w / 8.0, block_h / 8.0, components / 4.0], dtype=np.float32)
    feature = np.concatenate([padded.reshape(-1), extra, orientation])
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


class RankerLabelView:
    """C++ 版ラベルキャッシュ上の単一ヘッドビュー。"""

    def __init__(self, array: np.ndarray, invalid_value: int) -> None:
        self._array = array
        self._invalid_value = int(invalid_value)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        values = self._array[rows]
        out = values.astype(np.int32, copy=True)
        out[out == self._invalid_value] = -1
        if scalar:
            return int(out[0])
        return out


class RankerLabelCache:
    """C++ 実装のラベルキャッシュを numpy.memmap として提供する。"""

    def __init__(
        self,
        meta: Dict[str, object],
        bin_path: Path,
        *,
        topk_path: Optional[Path],
        topk_k: int,
    ) -> None:
        self._meta = meta
        self._bin_path = Path(bin_path)
        self._topk_path = topk_path
        self._topk_k = int(max(0, topk_k))
        self._record_count = int(meta.get("n_samples", 0))
        if self._record_count <= 0:
            raise ValueError("ラベルキャッシュのサンプル数が不正です")
        self._invalid_value = int(meta.get("invalid_value", 0xFFFF))

        header, head_descs = _read_ranker_label_header(self._bin_path, expect_magic=RANKER_LABEL_MAGIC)
        if header["version"] != 1:
            raise ValueError("ラベルキャッシュのバージョンが不正です")
        if header["n_samples"] != self._record_count:
            raise ValueError("ラベルキャッシュのサンプル数がメタデータと一致しません")
        try:
            meta_hash = int(meta.get("jsonl_hash", header["jsonl_hash"]))
        except (TypeError, ValueError):
            meta_hash = header["jsonl_hash"]
        if int(header["jsonl_hash"]) != meta_hash:
            logging.warning("ラベルキャッシュの JSONL ハッシュがメタデータと一致しません")
        heads_meta = meta.get("heads")
        if not isinstance(heads_meta, list) or len(heads_meta) != int(header["n_heads"]):
            raise ValueError("ラベルキャッシュのヘッド情報が不正です")

        sample_stride = 0
        for desc in head_descs:
            sample_stride = max(sample_stride, int(desc["offset"]) + int(desc["stride"]))
        if sample_stride <= 0:
            raise ValueError("ラベルキャッシュのストライド計算に失敗しました")
        if sample_stride % 2 != 0:
            raise ValueError("ラベルキャッシュのストライドが 2 バイト境界になっていません")

        self._raw = np.memmap(
            str(self._bin_path),
            dtype=np.uint8,
            mode="r",
            offset=int(header["data_off"]),
            shape=(self._record_count, sample_stride),
        )
        self._u16 = self._raw.view("<u2")

        self._best_arrays: Dict[str, np.ndarray] = {}
        self._second_arrays: Dict[str, np.ndarray] = {}

        topk_arrays: Optional[np.ndarray] = None
        topk_descs: Optional[List[Dict[str, int]]] = None
        if self._topk_path is not None and self._topk_k > 0 and self._topk_path.exists():
            topk_header, tk_descs = _read_ranker_label_header(self._topk_path, expect_magic=RANKER_TOPK_MAGIC)
            if topk_header["version"] != 1:
                raise ValueError("topK ラベルキャッシュのバージョンが不正です")
            if topk_header["n_samples"] != self._record_count:
                raise ValueError("topK ラベルキャッシュのサンプル数が一致しません")
            if int(topk_header["n_heads"]) != len(head_descs):
                raise ValueError("topK ラベルキャッシュのヘッド数が一致しません")
            topk_stride = 0
            for desc in tk_descs:
                topk_stride = max(topk_stride, int(desc["offset"]) + int(desc["stride"]))
            if topk_stride <= 0:
                raise ValueError("topK ラベルキャッシュのストライド計算に失敗しました")
            if topk_stride % 2 != 0:
                raise ValueError("topK ラベルキャッシュのストライドが 2 バイト境界になっていません")
            self._topk_raw = np.memmap(
                str(self._topk_path),
                dtype=np.uint8,
                mode="r",
                offset=int(topk_header["data_off"]),
                shape=(self._record_count, topk_stride),
            )
            topk_arrays = self._topk_raw.view("<u2")
            topk_descs = tk_descs
        else:
            self._topk_raw = None

        for idx, entry in enumerate(heads_meta):
            if not isinstance(entry, dict):
                raise ValueError("ラベルキャッシュのヘッド記述子が壊れています")
            name = str(entry.get("name"))
            desc = head_descs[idx]
            offset_bytes = int(desc["offset"])
            stride_bytes = int(desc["stride"])
            if stride_bytes <= 0 or stride_bytes % 2 != 0:
                raise ValueError("ラベルキャッシュのヘッド stride が不正です")
            start = offset_bytes // 2
            width = stride_bytes // 2
            if start + width > self._u16.shape[1]:
                raise ValueError("ラベルキャッシュのヘッド範囲が配列を超えています")
            view = self._u16[:, start : start + width]
            if width >= 1:
                self._best_arrays[name] = view[:, 0]
            else:
                self._best_arrays[name] = np.full((self._record_count,), self._invalid_value, dtype=np.uint16)
            second_array: np.ndarray
            if topk_arrays is not None and topk_descs is not None:
                tk_desc = topk_descs[idx]
                tk_offset = int(tk_desc["offset"]) // 2
                tk_width = max(0, int(tk_desc["stride"]) // 2)
                if tk_offset + tk_width > topk_arrays.shape[1]:
                    raise ValueError("topK ラベルキャッシュのヘッド範囲が不正です")
                if tk_width >= 2:
                    second_array = topk_arrays[:, tk_offset + 1]
                else:
                    second_array = np.full((self._record_count,), self._invalid_value, dtype=np.uint16)
            else:
                second_array = np.full((self._record_count,), self._invalid_value, dtype=np.uint16)
            self._second_arrays[name] = second_array

    def make(self, head: str, *, use_second: bool) -> RankerLabelView:
        try:
            if use_second:
                return RankerLabelView(self._second_arrays[head], self._invalid_value)
            return RankerLabelView(self._best_arrays[head], self._invalid_value)
        except KeyError as exc:
            raise KeyError(f"未知のヘッド名です: {head}") from exc

    @property
    def record_count(self) -> int:
        return int(self._record_count)

    def close(self) -> None:
        if isinstance(getattr(self, "_raw", None), np.memmap):
            try:
                self._raw._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(getattr(self, "_topk_raw", None), np.memmap):
            try:
                self._topk_raw._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass


def _read_ranker_label_header(path: Path, *, expect_magic: bytes) -> Tuple[Dict[str, int], List[Dict[str, int]]]:
    """C++ 版ラベルキャッシュヘッダーを読み取り構造化する。"""

    with Path(path).open("rb") as fp:
        header_raw = fp.read(RANKER_LABEL_HEADER_STRUCT.size)
        if len(header_raw) != RANKER_LABEL_HEADER_STRUCT.size:
            raise ValueError(f"ラベルキャッシュヘッダーが不足しています: {path}")
        magic, version, n_heads, n_samples, head_off, data_off, jsonl_hash = RANKER_LABEL_HEADER_STRUCT.unpack(
            header_raw
        )
        expected = expect_magic.rstrip(b"\0")
        actual = magic.rstrip(b"\0")
        if actual != expected:
            raise ValueError(f"ラベルキャッシュのマジックが不正です (expected={expected!r}, actual={actual!r})")
        if n_heads <= 0:
            raise ValueError("ラベルキャッシュのヘッド数が不正です")
        fp.seek(int(head_off))
        head_descs: List[Dict[str, int]] = []
        for _ in range(int(n_heads)):
            data = fp.read(RANKER_LABEL_HEAD_STRUCT.size)
            if len(data) != RANKER_LABEL_HEAD_STRUCT.size:
                raise ValueError("ラベルキャッシュのヘッド記述子が不足しています")
            name_id, n_classes, stride, offset = RANKER_LABEL_HEAD_STRUCT.unpack(data)
            head_descs.append(
                {
                    "name_id": int(name_id),
                    "n_classes": int(n_classes),
                    "stride": int(stride),
                    "offset": int(offset),
                }
            )
    header = {
        "magic": actual,
        "version": int(version),
        "n_heads": int(n_heads),
        "n_samples": int(n_samples),
        "head_off": int(head_off),
        "data_off": int(data_off),
        "jsonl_hash": int(jsonl_hash),
    }
    return header, head_descs


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
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._max_workers > 1:
            # バッチごとにスレッドプールを張り直すとオーバーヘッドが大きいので再利用する
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="lazyfeat")
            atexit.register(self._shutdown_executor)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        batch = np.empty((rows.shape[0], self._shape[1]), dtype=np.float32)
        executor = self._executor
        if executor is None or rows.shape[0] <= 1:
            for out_idx, rec_idx in enumerate(rows):
                batch[out_idx] = self._read_and_pack(int(rec_idx))
        else:
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

    def _shutdown_executor(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None

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


class _ArrayFeatureSource:
    """事前計算済み特徴量行列を遅延読み出し可能な形で扱う簡易ラッパー。"""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 2:
            raise ValueError("特徴量行列は 2 次元である必要があります")
        self._array = array
        self._shape = (int(array.shape[0]), int(array.shape[1]))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, len(self))
        batch = np.asarray(self._array[rows], dtype=np.float32)
        if scalar and batch.ndim == 2 and batch.shape[0] == 1:
            return batch[0]
        return batch


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


def _prefetch_training_batches(
    train_features: LazyFeatures,
    train_targets: Dict[str, np.ndarray],
    active_heads: Sequence[str],
    batches: Sequence[np.ndarray],
    *,
    max_prefetch: int = 2,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """学習バッチを非同期に読み出してGPUの待ち時間を削減する。"""

    stop_token: object = object()
    error_token: object = object()
    work_queue: queue.Queue[object] = queue.Queue(max(1, int(max_prefetch)))

    def _worker() -> None:
        try:
            for batch_idx in batches:
                features_np = np.ascontiguousarray(train_features[batch_idx], dtype=np.float32)
                targets_np = {
                    name: np.ascontiguousarray(train_targets[name][batch_idx], dtype=np.float32)
                    for name in active_heads
                }
                work_queue.put((batch_idx, features_np, targets_np))
        except Exception as exc:
            work_queue.put((error_token, exc))
        finally:
            work_queue.put(stop_token)

    thread = threading.Thread(target=_worker, name="prefetch_batches", daemon=True)
    thread.start()

    while True:
        item = work_queue.get()
        if item is stop_token:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] is error_token:
            _, exc = item
            raise exc
        if isinstance(item, tuple) and len(item) == 3:
            batch_idx, features_np, targets_np = item
            yield batch_idx, features_np, targets_np
            continue
        raise RuntimeError("prefetch バックグラウンドスレッドから予期しない値を受信しました")

    thread.join()


def _encode_onehot(ids: np.ndarray, class_count: int) -> np.ndarray:
    """1 次元 ID 配列をワンホット表現に変換する。"""

    ids32 = ids.astype(np.int32, copy=False)
    out = np.zeros((ids32.shape[0], class_count), dtype=np.float32)
    if ids32.size == 0:
        return out
    valid = (ids32 >= 0) & (ids32 < class_count)
    if not np.any(valid):
        return out
    rows = np.nonzero(valid)[0]
    out[rows, ids32[valid]] = 1.0
    return out


class AugmentedFeatures:
    """条件付きヘッドの情報を結合した特徴量ビュー。"""

    def __init__(
        self,
        base: object,
        cond_arrays: Dict[str, np.ndarray],
        cond_specs: Sequence[Tuple[str, int]],
        *,
        encoding: str = "onehot",
    ) -> None:
        self._base = base
        self._specs = list(cond_specs)
        self._encoding = (encoding or "onehot").lower()
        if self._encoding not in ("onehot", "id"):
            raise ValueError(f"未知の条件エンコーディングです: {encoding}")
        if hasattr(base, "__len__"):
            self._length = int(len(base))  # type: ignore[arg-type]
        else:
            self._length = int(getattr(base, "shape")[0])  # type: ignore[index]
        base_shape = getattr(base, "shape")
        base_dim = int(base_shape[1])  # type: ignore[index]
        self._cond_arrays: Dict[str, np.ndarray] = {
            name: np.asarray(arr, dtype=np.int16)
            for name, arr in cond_arrays.items()
        }
        for name, _classes in self._specs:
            if name not in self._cond_arrays:
                raise KeyError(f"条件付き配列が見つかりません: {name}")
            if self._cond_arrays[name].shape[0] != self._length:
                raise ValueError("条件付き配列の長さが特徴量と一致しません")
        extra = sum((cls if self._encoding == "onehot" else 1) for _, cls in self._specs)
        self._shape = (self._length, base_dim + extra)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: object) -> np.ndarray:
        rows, scalar = _normalize_indices(idx, self._length)
        base_batch = self._base[rows]
        batch = np.asarray(base_batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch[None, :]
        extras: List[np.ndarray] = []
        for name, classes in self._specs:
            ids = np.asarray(self._cond_arrays[name][rows], dtype=np.int32)
            if ids.ndim == 0:
                ids = ids.reshape(1)
            if self._encoding == "onehot":
                extras.append(_encode_onehot(ids, classes))
            else:
                extras.append(ids.astype(np.float32, copy=False).reshape(-1, 1))
        if extras:
            batch = np.concatenate([batch] + extras, axis=1)
        if scalar:
            return batch[0]
        return batch


FEATURE_CACHE_META_VERSION = 1


def _write_export_metadata(export_dir: Path, meta: Dict[str, object], config: Dict[str, object]) -> None:
    """学習結果の補助メタデータを書き出す。"""

    export_dir.mkdir(parents=True, exist_ok=True)
    meta_path = export_dir / "meta.json"
    with meta_path.open('w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2, sort_keys=True)
    config_path = export_dir / "config.json"
    with config_path.open('w', encoding='utf-8') as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2, sort_keys=True)


def _feature_cache_meta_path(cache_path: Path) -> Path:
    """特徴量キャッシュメタデータのパスを生成する。"""

    if cache_path.suffix:
        return cache_path.with_suffix(cache_path.suffix + ".meta.json")
    return cache_path.parent / f"{cache_path.name}.meta.json"


def _feature_cache_files_match(files: Sequence[FileRef], meta_files: object) -> bool:
    """メタデータと実際の入力ファイル構成が一致するか確認する。"""

    if not isinstance(meta_files, list) or len(meta_files) != len(files):
        return False
    for ref, entry in zip(files, meta_files):
        if not isinstance(entry, dict):
            return False
        try:
            recorded_path = Path(entry["path"]).resolve()
            recorded_size = int(entry["size"])
            recorded_mtime = float(entry["mtime"])
        except (KeyError, TypeError, ValueError):
            return False
        if recorded_path != ref.path.resolve():
            return False
        if recorded_size != int(ref.size):
            return False
        if not math.isclose(float(ref.mtime), recorded_mtime, rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def _try_load_feature_cache(
    cache_path: Path,
    files: Sequence[FileRef],
    *,
    expected_records: int,
    expected_dim: int,
    cond_heads: Sequence[str],
    cond_encoding: str,
    meta_path: Optional[Path] = None,
) -> Optional[np.memmap]:
    """既存の特徴量キャッシュが利用可能なら mmap して返す。"""

    meta_path = meta_path or _feature_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("特徴量キャッシュメタデータの読み込みに失敗しました (%s)", exc)
        return None

    format_val = meta.get("format")
    if isinstance(format_val, str) and format_val == "tlg8-ranker-features":
        try:
            record_count = int(meta.get("n_samples", 0))
            feature_dim = int(meta.get("feature_dim", 0))
        except (TypeError, ValueError):
            logging.warning("C++ 特徴量キャッシュのメタデータが不正です")
            return None
        if record_count != expected_records or feature_dim != expected_dim:
            logging.warning(
                "C++ 特徴量キャッシュの形状が一致しません (records=%d/%d, dim=%d/%d)",
                record_count,
                expected_records,
                feature_dim,
                expected_dim,
            )
            return None
        if cond_heads:
            logging.warning("条件付き特徴量指定時は C++ 特徴量キャッシュを利用できません")
            return None
        if not _feature_cache_files_match(files, meta.get("source_files")):
            logging.warning("C++ 特徴量キャッシュの入力ファイル構成が一致しません")
            return None
        try:
            arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            logging.warning("C++ 特徴量キャッシュの読み込みに失敗しました: %s", exc)
            return None
        if arr.ndim != 2 or arr.shape[0] != expected_records or arr.shape[1] != expected_dim:
            logging.warning("C++ 特徴量キャッシュの形状が不正です (shape=%s)", arr.shape)
            return None
        logging.info("C++ 特徴量キャッシュを利用します: %s (shape=%d×%d)", cache_path, arr.shape[0], arr.shape[1])
        return arr

    try:
        version = int(meta.get("version", 0))
        record_count = int(meta.get("record_count", 0))
        feature_dim = int(meta.get("feature_dim", 0))
    except (TypeError, ValueError):
        logging.warning("特徴量キャッシュメタデータの基本情報が不正です")
        return None

    if version != FEATURE_CACHE_META_VERSION:
        logging.warning("特徴量キャッシュのメタデータバージョンが一致しません (meta=%d)", version)
        return None
    if record_count != expected_records or feature_dim != expected_dim:
        logging.warning(
            "特徴量キャッシュの形状が現在の設定と一致しません (records=%d/%d, dim=%d/%d)",
            record_count,
            expected_records,
            feature_dim,
            expected_dim,
        )
        return None

    recorded_heads = meta.get("condition_heads", [])
    recorded_encoding = str(meta.get("condition_encoding", "")).lower()
    if list(recorded_heads) != list(cond_heads) or recorded_encoding != cond_encoding.lower():
        logging.warning("特徴量キャッシュの条件付き設定が現在の CLI と一致しません")
        return None

    if not _feature_cache_files_match(files, meta.get("files")):
        logging.warning("特徴量キャッシュの入力ファイル構成が一致しません。再構築します。")
        return None

    try:
        arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        logging.warning("特徴量キャッシュ %s の読み込みに失敗しました: %s", cache_path, exc)
        return None

    if arr.ndim != 2 or arr.shape[0] != expected_records or arr.shape[1] != expected_dim:
        logging.warning("特徴量キャッシュファイルの形状が不正です (shape=%s)", arr.shape)
        return None

    logging.info("特徴量キャッシュを利用します: %s (shape=%d×%d)", cache_path, arr.shape[0], arr.shape[1])
    return arr


def _build_feature_cache(
    cache_path: Path,
    files: Sequence[FileRef],
    base_features: LazyFeatures,
    *,
    cond_arrays: Dict[str, np.ndarray],
    cond_specs: Sequence[Tuple[str, int]],
    cond_encoding: str,
    cond_heads: Sequence[str],
    total: int,
    expected_dim: int,
    chunk_size: int,
    meta_path: Optional[Path] = None,
) -> np.memmap:
    """特徴量キャッシュを新規作成して mmap を返す。"""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_target = meta_path or _feature_cache_meta_path(cache_path)
    feature_source: object
    if cond_specs:
        feature_source = AugmentedFeatures(base_features, cond_arrays, cond_specs, encoding=cond_encoding)
    else:
        feature_source = base_features

    from numpy.lib.format import open_memmap
    mm = open_memmap(str(cache_path), mode="w+", dtype=np.float32, shape=(total, expected_dim))

    # ヘッダ付き .npy を作る。これなら直後の np.load(..., mmap_mode="r") と整合する
    progress = ProgressReporter(
        total,
        "特徴量キャッシュ構築中",
        unit="サンプル",
        enable=ENABLE_PROGRESS and total > 0,
    )
    step = max(1, chunk_size)
    try:
        for start in range(0, total, step):
            end = min(total, start + step)
            batch = np.asarray(feature_source[start:end], dtype=np.float32)
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)
            if batch.shape[1] != expected_dim:
                raise ValueError(f"キャッシュ構築中に特徴量次元が一致しません (expected={expected_dim}, actual={batch.shape[1]})")
            mm[start:end, :] = batch
            if progress is not None:
                progress.update(end - start)
    finally:
        progress.close()
        mm.flush()

    meta = {
        "version": FEATURE_CACHE_META_VERSION,
        "record_count": int(total),
        "feature_dim": int(expected_dim),
        "condition_heads": list(cond_heads),
        "condition_encoding": cond_encoding.lower(),
        "files": [
            {
                "path": str(ref.path.resolve()),
                "size": int(ref.size),
                "mtime": float(ref.mtime),
            }
            for ref in files
        ],
    }
    with meta_target.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp)

    del mm  # mmap を明示的にクローズする
    arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    logging.info("特徴量キャッシュを構築しました: %s (shape=%d×%d)", cache_path, arr.shape[0], arr.shape[1])
    return arr


def load_or_build_feature_cache(
    cache_path: Path,
    files: Sequence[FileRef],
    base_features: LazyFeatures,
    *,
    cond_arrays: Dict[str, np.ndarray],
    cond_specs: Sequence[Tuple[str, int]],
    cond_encoding: str,
    cond_heads: Sequence[str],
    total: int,
    expected_dim: int,
    chunk_size: int = 65536,
    meta_path: Optional[Path] = None,
) -> np.memmap:
    """特徴量キャッシュをロードまたは構築して mmap を返す。"""

    cache = _try_load_feature_cache(
        cache_path,
        files,
        expected_records=total,
        expected_dim=expected_dim,
        cond_heads=cond_heads,
        cond_encoding=cond_encoding,
        meta_path=meta_path,
    )
    if cache is not None:
        return cache
    logging.info(
        "特徴量キャッシュ %s を再構築します (records=%d, dim=%d)",
        cache_path,
        total,
        expected_dim,
    )
    return _build_feature_cache(
        cache_path,
        files,
        base_features,
        cond_arrays=cond_arrays,
        cond_specs=cond_specs,
        cond_encoding=cond_encoding,
        cond_heads=cond_heads,
        total=total,
        expected_dim=expected_dim,
        chunk_size=max(1, int(chunk_size)),
        meta_path=meta_path,
    )


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


def load_feature_scaler_from_binary(path: Path, expected_dim: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """C++ で事前計算した特徴量統計から平均・標準偏差を復元する。"""

    with Path(path).open("rb") as fp:
        header = fp.read(FEATURE_STATS_HEADER.size)
        if len(header) != FEATURE_STATS_HEADER.size:
            raise ValueError(f"特徴量統計ファイルのヘッダーが不完全です: {path}")
        magic, version, dimension, count = FEATURE_STATS_HEADER.unpack(header)
        if magic != FEATURE_STATS_MAGIC or version != FEATURE_STATS_VERSION:
            raise ValueError(f"特徴量統計ファイルのマジック/バージョンが不正です: {path}")
        if dimension != expected_dim:
            raise FeatureStatsDimensionError(expected_dim, dimension)
        sum_bytes = fp.read(8 * dimension)
        if len(sum_bytes) != 8 * dimension:
            raise ValueError("特徴量統計ファイルの sum 部が不足しています")
        sums = np.frombuffer(sum_bytes, dtype="<f8").astype(np.float64, copy=True)
        sumsq_bytes = fp.read(8 * dimension)
        if len(sumsq_bytes) != 8 * dimension:
            raise ValueError("特徴量統計ファイルの sumsq 部が不足しています")
        sumsq = np.frombuffer(sumsq_bytes, dtype="<f8").astype(np.float64, copy=True)

    if count <= 1:
        variance = np.ones_like(sums)
    else:
        variance = (sumsq - (sums ** 2) / count) / (count - 1)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean = (sums / max(count, 1)).astype(np.float32)
    return mean, std, int(count)


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
    allow_cpu_transfer: bool = False,
) -> Dict[str, np.ndarray]:
    """predict_logits_batched にプログレスバーを付与したラッパー。"""

    if not ENABLE_PROGRESS:
        return predict_logits_batched(
            model,
            features,
            device,
            batch_size=batch_size,
            amp=amp,
            allow_cpu_transfer=allow_cpu_transfer,
        )

    if isinstance(features, np.ndarray):
        total = int(features.shape[0]) if features.ndim > 1 else 1
    else:
        total = int(getattr(features, "shape")[0])  # type: ignore[index]

    progress = ProgressReporter(total, desc, unit="サンプル", enable=ENABLE_PROGRESS)
    if total <= 0:
        progress.close()
        return predict_logits_batched(
            model,
            features,
            device,
            batch_size=batch_size,
            amp=amp,
            allow_cpu_transfer=allow_cpu_transfer,
        )

    if isinstance(features, np.ndarray):
        try:
            return predict_logits_batched(
                model,
                features,
                device,
                batch_size=batch_size,
                amp=amp,
                allow_cpu_transfer=allow_cpu_transfer,
            )
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
        return predict_logits_batched(
            model,
            wrapped,
            device,
            batch_size=batch_size,
            amp=amp,
            allow_cpu_transfer=allow_cpu_transfer,
        )
    finally:
        progress.close()


def compute_metrics_on_device_with_progress(
    model: "torch.nn.Module",
    features: object,
    device: "torch.device",
    best: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
    *,
    batch_size: int,
    amp: str,
    desc: str,
    condition_dim: int = 0,
    scramble_cond: bool = False,
) -> Dict[str, Dict[str, float]]:
    """GPU 上でロジット指標を集計しつつプログレスを表示する。"""

    if torch is None:
        raise ImportError("PyTorch が利用できません")

    total = 0
    if isinstance(features, np.ndarray):
        total = int(features.shape[0]) if features.ndim > 1 else 1
    else:
        total = int(getattr(features, "shape")[0])  # type: ignore[index]

    progress = ProgressReporter(total, desc, unit="サンプル", enable=ENABLE_PROGRESS and total > 0)
    try:
        return _compute_metrics_on_device(
            model,
            features,
            device,
            best,
            second,
            batch_size=batch_size,
            amp=amp,
            progress=progress if ENABLE_PROGRESS and total > 0 else None,
            condition_dim=condition_dim,
            scramble_cond=scramble_cond,
        )
    finally:
        progress.close()


def _compute_metrics_on_device(
    model: "torch.nn.Module",
    features: object,
    device: "torch.device",
    best: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
    *,
    batch_size: int,
    amp: str,
    progress: Optional[ProgressReporter] = None,
    condition_dim: int = 0,
    scramble_cond: bool = False,
) -> Dict[str, Dict[str, float]]:
    """GPU 上で top-k 指標を集計する内部実装。"""

    if torch is None:
        raise ImportError("PyTorch が利用できません")

    if batch_size <= 0:
        raise ValueError("batch_size は正の整数である必要があります")

    active_heads = tuple(getattr(model, "active_heads", tuple(best.keys())))
    if not active_heads:
        return {}

    if isinstance(features, np.ndarray):
        arr = np.asarray(features, dtype=np.float32)
        total = int(arr.shape[0]) if arr.ndim > 1 else 1
        fetch = lambda start, end: np.ascontiguousarray(arr[start:end], dtype=np.float32)
    else:
        total = int(getattr(features, "shape")[0])  # type: ignore[index]
        fetch = lambda start, end: np.ascontiguousarray(features[start:end], dtype=np.float32)  # type: ignore[index]

    if total == 0:
        return {
            name: {
                "top1": float("nan"),
                "top2": float("nan"),
                "top3": float("nan"),
                "three_choice": float("nan"),
            }
            for name in active_heads
        }

    was_training = bool(model.training)
    model.eval()

    non_blocking = device.type == "cuda"
    metrics_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in active_heads:
        metrics_buf[name] = {
            "total": torch.zeros((), device=device, dtype=torch.long),
            "top1": torch.zeros((), device=device, dtype=torch.long),
            "top2": torch.zeros((), device=device, dtype=torch.long),
            "top3": torch.zeros((), device=device, dtype=torch.long),
            "three_choice": torch.zeros((), device=device, dtype=torch.long),
        }

    try:
        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(total, start + batch_size)
                batch_np = fetch(start, end)
                if batch_np.ndim == 1:
                    batch_np = batch_np[None, :]
                xb = torch.from_numpy(batch_np)
                if non_blocking:
                    xb = xb.pin_memory()
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                if scramble_cond and condition_dim > 0 and xb.shape[1] >= condition_dim:
                    cond_slice = xb[:, -condition_dim:]
                    if cond_slice.shape[0] > 1:
                        perm = torch.randperm(cond_slice.shape[0], device=cond_slice.device)
                        cond_scrambled = cond_slice.index_select(0, perm)
                        xb[:, -condition_dim:] = cond_scrambled
                with _amp_context(device, amp):
                    logits = model(xb)
                batch_size_actual = xb.shape[0]
                if progress is not None:
                    progress.update(batch_size_actual)
                for name in active_heads:
                    head_logits = logits[name]
                    stats = metrics_buf[name]
                    stats["total"] += batch_size_actual
                    best_slice = np.ascontiguousarray(best[name][start:end], dtype=np.int64)
                    second_slice = np.ascontiguousarray(second[name][start:end], dtype=np.int64)
                    best_tensor = torch.from_numpy(best_slice)
                    second_tensor = torch.from_numpy(second_slice)
                    if non_blocking:
                        best_tensor = best_tensor.pin_memory()
                        second_tensor = second_tensor.pin_memory()
                    best_dev = best_tensor.to(device=device, dtype=torch.long, non_blocking=non_blocking)
                    second_dev = second_tensor.to(device=device, dtype=torch.long, non_blocking=non_blocking)
                    classes = head_logits.shape[1]
                    k = min(3, classes)
                    topk = torch.topk(head_logits, k=k, dim=1)
                    topk_indices = topk.indices
                    pred1 = topk_indices[:, 0]
                    stats["top1"] += (pred1 == best_dev).sum()
                    if classes >= 2:
                        in_top2 = (topk_indices[:, : min(2, k)] == best_dev.unsqueeze(1)).any(dim=1)
                    else:
                        in_top2 = pred1 == best_dev
                    stats["top2"] += in_top2.sum()
                    in_top3 = (topk_indices[:, :k] == best_dev.unsqueeze(1)).any(dim=1)
                    stats["top3"] += in_top3.sum()
                    valid_second = second_dev >= 0
                    match_three = (pred1 == best_dev) | (valid_second & (pred1 == second_dev))
                    stats["three_choice"] += match_three.sum()
    finally:
        if was_training:
            model.train()

    metrics: Dict[str, Dict[str, float]] = {}
    for name in active_heads:
        stats = metrics_buf[name]
        total_count = int(stats["total"].item())
        if total_count == 0:
            metrics[name] = {
                "top1": float("nan"),
                "top2": float("nan"),
                "top3": float("nan"),
                "three_choice": float("nan"),
            }
            continue
        metrics[name] = {
            "top1": float(stats["top1"].item() / total_count),
            "top2": float(stats["top2"].item() / total_count),
            "top3": float(stats["top3"].item() / total_count),
            "three_choice": float(stats["three_choice"].item() / total_count),
        }
    return metrics


def dump_logits_to_npz(
    model: "torch.nn.Module",
    features: object,
    device: "torch.device",
    *,
    batch_size: int,
    amp: str,
    desc: str,
    output_path: Path,
) -> None:
    """ロジットを明示的に保存する (CPU 転送を伴う) ユーティリティ。"""

    logging.info("%s: CPU 転送を伴うロジット書き出しを開始します -> %s", desc, output_path)
    logits = predict_logits_with_progress(
        model,
        features,
        device,
        batch_size=batch_size,
        amp=amp,
        desc=desc,
        allow_cpu_transfer=True,
    )
    np.savez(output_path, **logits)
    logging.info("%s: ロジットを書き出しました", desc)


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
    heads = [name for name in HEAD_ORDER if name in metrics]
    if not heads:
        heads = list(metrics.keys())
    for key in ("top1", "top2", "top3", "three_choice"):
        values = [metrics[name].get(key, float("nan")) for name in heads]
        result[key] = float(np.nanmean(values))
    return result


def _amp_context(device: "torch.device", amp_mode: str):
    """AMP 設定に応じて適切なコンテキストを返す。"""

    if torch is None:
        return nullcontext()
    mode = (amp_mode or "none").lower()
    if mode not in ("bf16", "fp16"):
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    if device.type == "cpu":
        return nullcontext()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.amp.autocast(dtype=dtype)  # type: ignore[attr-defined]
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_with_torch_backend(
    args: argparse.Namespace,
    train_features: object,
    train_best: Dict[str, np.ndarray],
    train_second: Dict[str, np.ndarray],
    val_features: object,
    val_best: Dict[str, np.ndarray],
    val_second: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    disabled_heads: Sequence[str],
    condition_heads: Sequence[str],
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

    device = getattr(args, "device_resolved", None)
    if device is None:
        device = pick_device(args.device)
    logging.info("PyTorch デバイス: %s", device)
    amp_arg = (args.amp or "auto").lower()
    if amp_arg == "auto":
        amp_mode = "bf16" if device.type == "xpu" else "none"
    else:
        amp_mode = amp_arg
    if amp_mode not in ("none", "bf16", "fp16"):
        logging.warning("未知の AMP 指定 %s のため無効化します", amp_mode)
        amp_mode = "none"
    if device.type == "cpu" and amp_mode in ("bf16", "fp16"):
        logging.warning("CPU では指定された AMP モード %s を利用できないため無効化します", amp_mode)
        amp_mode = "none"
    if device.type == "cuda" and amp_mode == "bf16":
        is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not is_supported:
            logging.warning("この CUDA デバイスは bfloat16 をサポートしていないため AMP を無効化します")
            amp_mode = "none"
    if device.type == "xpu" and amp_mode == "fp16":
        logging.warning("XPU では fp16 AMP をサポートしていないため無効化します")
        amp_mode = "none"

    import numpy as _np

    train_mean = None
    train_std = None

    try:
        scaler_path = Path(args.scaler) if args.scaler else None
        if scaler_path is not None and scaler_path.exists():
            with _np.load(scaler_path) as _sc:
                if "mean" in _sc and "std" in _sc:
                    train_mean = _np.asarray(_sc["mean"], dtype=_np.float32)
                    train_std = _np.asarray(_sc["std"], dtype=_np.float32)
                    logger.info("スケーラーから train_mean/std を復元 (dims=%d)", train_mean.shape[0])
    except Exception as _exc:
        logger.warning("スケーラーからの mean/std 復元に失敗: %s", _exc)

    if (train_mean is None or train_std is None) and args.feature_cache:
        try:
            feats = _np.load(args.feature_cache, mmap_mode="r")
            train_mean = _np.asarray(feats.mean(axis=0), dtype=_np.float32)
            train_std = _np.asarray(feats.std(axis=0) + 1e-8, dtype=_np.float32)
            logger.info("特徴量キャッシュから train_mean/std を推定 (dims=%d)", train_mean.shape[0])
        except Exception as _exc:
            logger.warning("特徴量キャッシュからの mean/std 推定に失敗: %s", _exc)

    if train_mean is None or train_std is None:
        train_mean = _np.asarray(mean, dtype=_np.float32)
        train_std = _np.asarray(std, dtype=_np.float32)
        logger.info("既存 mean/std を利用して train_mean/std を初期化 (dims=%d)", train_mean.shape[0])

    reorder_head_config = {
        "use_cosine": bool(args.reorder_cosine),
        "scale": float(max(1e-6, args.reorder_arcface_s)),
        "arcface_m": float(max(0.0, args.reorder_arcface_m)),
    }
    cond_extra_dim = conditioned_extra_dim(condition_heads, args.condition_encoding)
    final_input_dim = int(train_mean.shape[0])
    base_dim = max(0, final_input_dim - cond_extra_dim)
    model = TorchMultiTask(
        train_mean,
        train_std,
        dropout=args.dropout,
        disabled_heads=disabled_heads,
        reorder_head_config=reorder_head_config,
        base_dim=base_dim,
        condition_dim=cond_extra_dim,
        logger=logger,
    ).to(device)
    if hasattr(model, "set_condition_heads"):
        model.set_condition_heads(condition_heads)
    # 既存モデルから初期化（保存形式を自動判別）
    if args.init_from is not None:
        import torch as _torch
        import torch.serialization as _ser
        from collections.abc import Mapping

        def _safe_load_any(path, map_location):
            # (A) safe_globals を追加して weights_only=True を試す
            try:
                import numpy as _np
                _ser.add_safe_globals([_np._core.multiarray._reconstruct])
                return _torch.load(path, map_location=map_location, weights_only=True)
            except Exception:
                # (B) フォールバック（信頼済みファイル前提）
                return _torch.load(path, map_location=map_location, weights_only=False)

        payload = _safe_load_any(str(args.init_from), map_location=device)

        # 3 形式に対応：
        # (1) TorchMultiTask オブジェクト
        # (2) {"state_dict": ..., <meta...>} の辞書
        # (3) 純 state_dict（= そのまま key->Tensor）
        if isinstance(payload, TorchMultiTask):
            state = payload.state_dict()
            model.inference_temperature = getattr(payload, "inference_temperature", 1.0)
        elif isinstance(payload, Mapping) and "state_dict" in payload:
            state = payload["state_dict"]
            model.inference_temperature = payload.get("inference_temperature", getattr(model, "inference_temperature", 1.0))
        else:
            # 純 state_dict とみなす
            state = payload

        partial_msgs: List[str] = []
        if isinstance(state, Mapping):
            # 形状不一致パラメータは strict=False でも例外になるため事前に間引く
            model_state = model.state_dict()
            pruned_keys: List[str] = []
            try:
                state = state.copy()  # type: ignore[assignment]
            except AttributeError:
                state = dict(state)  # type: ignore[assignment]

            def _coerce_like(value: object, reference: torch.Tensor) -> torch.Tensor:
                if isinstance(value, torch.Tensor):
                    return value.detach().to(device=reference.device, dtype=reference.dtype)
                arr = np.asarray(value)
                tensor = torch.as_tensor(arr, dtype=reference.dtype)
                return tensor.to(device=reference.device)

            def _partial_fc1_weight() -> None:
                key = "fc1.weight"
                if key not in state or key not in model_state:
                    return
                src_tensor = state[key]
                tgt_tensor = model_state[key]
                if not hasattr(src_tensor, "shape") or not hasattr(tgt_tensor, "shape"):
                    return
                if tuple(src_tensor.shape) == tuple(tgt_tensor.shape):
                    return
                src_aligned = _coerce_like(src_tensor, tgt_tensor)
                new_tensor = tgt_tensor.clone()
                overlap = min(src_aligned.shape[1], new_tensor.shape[1])
                if overlap > 0:
                    new_tensor[:, :overlap] = src_aligned[:, :overlap]
                state[key] = new_tensor
                partial_msgs.append(f"fc1.weight:{overlap}/{new_tensor.shape[1]}")

            dropped_scaler = []
            for scaler_key in ("feature_mean", "feature_std"):
                if scaler_key in state:
                    state.pop(scaler_key)
                    dropped_scaler.append(scaler_key)
            if dropped_scaler:
                partial_msgs.append(f"scaler_replaced:{','.join(dropped_scaler)}")
            _partial_fc1_weight()

            for key in list(state.keys()):
                if key not in model_state:
                    continue
                src_tensor = state[key]
                tgt_tensor = model_state[key]
                src_shape = tuple(src_tensor.shape) if hasattr(src_tensor, "shape") else None  # type: ignore[attr-defined]
                tgt_shape = tuple(tgt_tensor.shape) if hasattr(tgt_tensor, "shape") else None  # type: ignore[attr-defined]
                if src_shape is None or tgt_shape is None:
                    continue
                if src_shape != tgt_shape:
                    pruned_keys.append(f"{key}: {src_shape}->{tgt_shape}")
                    state.pop(key)
            if pruned_keys:
                logger.warning(
                    "[init-from] 形状不一致のため %d 個のパラメータを読み込み対象から除外しました (例: %s)",
                    len(pruned_keys),
                    ", ".join(pruned_keys[:4]),
                )

        missing, unexpected = model.load_state_dict(state, strict=False)
        missing_filtered = [key for key in missing if key not in {"feature_mean", "feature_std"}]
        if missing_filtered:
            logger.warning(f"[init-from] missing keys: {len(missing_filtered)} e.g. {missing_filtered[:8]}")
        if unexpected:
            logger.warning(f"[init-from] unexpected keys: {len(unexpected)} e.g. {unexpected[:8]}")
        if partial_msgs:
            sample = ", ".join(partial_msgs[:4])
            if len(partial_msgs) > 4:
                sample += " ..."
            logger.warning("[init-from] 入力次元差異のため部分的に再初期化したテンソル: %s", sample)

    else:
        model.inference_temperature = max(float(args.temperature), 1e-6)
    model.apply_feature_stats(train_mean, train_std)
    # 試運転で動的拡張の整合性を確認する
    with torch.no_grad():
        health = torch.zeros((1, model.input_dim), device=device, dtype=torch.float32)
        _ = model(health)
    logger.info(model.input_shape_summary())
    if device.type == "cuda":
        first_param = next(model.parameters(), None)
        if first_param is not None:
            assert first_param.is_cuda, "Model is not on CUDA"

    if device.type == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]

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

    freeze_trunk = bool(getattr(args, "freeze_trunk", False))
    if freeze_trunk and hasattr(model, "trunk_parameters"):
        for param in model.trunk_parameters():
            param.requires_grad_(False)
    param_groups: List[Dict[str, object]] = []
    if hasattr(model, "trunk_parameters"):
        trunk_params = [p for p in model.trunk_parameters() if p.requires_grad]
    else:
        trunk_params = []
    if trunk_params:
        param_groups.append({"params": trunk_params, "name": "trunk"})
    if hasattr(model, "head_parameters"):
        head_params = [p for p in model.head_parameters() if p.requires_grad]
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "name": "heads"})
    if not param_groups:
        raise ValueError("学習可能なパラメータが存在しません")
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_kwargs = dict(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        threshold=1e-4,
        min_lr=1e-5,
    )
    # 古い PyTorch では verbose 引数が存在しないため動的に判定する
    try:
        signature = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "verbose" in signature.parameters:
        scheduler_kwargs["verbose"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**scheduler_kwargs)

    active_heads = tuple(getattr(model, "active_heads", HEAD_ORDER))
    if not active_heads:
        raise ValueError("学習対象ヘッドが存在しません")

    head_list = list(active_heads)

    # ヘッド別損失重み (指定が無いヘッドは 1.0)
    weight_config = getattr(args, "head_loss_weights", {})
    head_loss_w = {name: float(weight_config.get(name, 1.0)) for name in active_heads}

    train_best_np = {name: np.ascontiguousarray(train_best[name], dtype=np.int64) for name in active_heads}
    train_second_np = {name: np.ascontiguousarray(train_second[name], dtype=np.int64) for name in active_heads}
    val_best_np = {name: np.ascontiguousarray(val_best[name], dtype=np.int64) for name in active_heads}
    val_second_np = {name: np.ascontiguousarray(val_second[name], dtype=np.int64) for name in active_heads}

    train_targets: Dict[str, np.ndarray] = {}
    epsilon_map = dict(getattr(args, "epsilon_soft_map", {}))
    epsilon_default = float(epsilon_map.get("default", args.epsilon_soft))
    for name in active_heads:
        epsilon_head = float(epsilon_map.get(name, epsilon_default))
        train_targets[name] = build_soft_targets(
            train_best_np[name],
            train_second_np[name],
            HEAD_SPECS[name],
            epsilon_head,
        ).astype(np.float32)
        train_targets[name] = np.ascontiguousarray(train_targets[name], dtype=np.float32)

    perm_weights_tensor: Optional["torch.Tensor"] = None
    perm_weight_device: Optional["torch.Tensor"] = None
    filter_perm_weight = float(weight_config.get("filter_perm", head_loss_w.get("filter_perm", 1.0)))
    if (
        args.perm_class_balance == "effective"
        and filter_perm_weight > 0.0
        and "filter_perm" in train_best_np
    ):
        num_classes_perm = HEAD_SPECS.get("filter_perm")
        if num_classes_perm is None:
            logging.warning("filter_perm のクラス数が未定義のためクラス重みは適用されません")
        else:
            labels_perm = np.asarray(train_best_np["filter_perm"], dtype=np.int64)
            valid_perm = labels_perm[labels_perm >= 0]
            counts = np.bincount(valid_perm, minlength=int(num_classes_perm))
            if counts.size == 0 or valid_perm.size == 0:
                logging.warning("filter_perm のクラス頻度が取得できなかったためクラス重みは適用されません")
            else:
                try:
                    perm_weights_tensor = effective_class_weights(counts, args.perm_class_beta)
                except Exception as exc:
                    logging.warning("filter_perm のクラス重み算出に失敗したため無効化します: %s", exc)
                    perm_weights_tensor = None
                else:
                    logging.info(
                        "filter_perm に有効サンプル数ベースのクラス重みを適用します (beta=%.4f)",
                        float(args.perm_class_beta),
                    )
    if perm_weights_tensor is not None:
        perm_weight_device = perm_weights_tensor.to(device=device)

    class_balance_heads = set(getattr(args, "class_balance_heads", set()))
    head_class_weight_cpu: Dict[str, torch.Tensor] = {}
    if class_balance_heads:
        for name in active_heads:
            if name not in class_balance_heads:
                continue
            classes = HEAD_SPECS.get(name)
            if classes is None:
                logging.warning("%s のクラス数が未定義のためクラス重みは適用されません", name)
                continue
            labels_arr = np.asarray(train_best_np[name], dtype=np.int64)
            valid = labels_arr[labels_arr >= 0]
            if valid.size == 0:
                logging.warning("%s の有効ラベルが不足しているためクラス重みは適用されません", name)
                continue
            counts = np.bincount(valid, minlength=int(classes))
            if counts.size == 0:
                logging.warning("%s のクラス頻度が取得できませんでした", name)
                continue
            try:
                weight_tensor = effective_class_weights(counts, args.class_balance_beta)
            except Exception as exc:
                logging.warning("%s のクラス重み算出に失敗したため無効化します: %s", name, exc)
                continue
            head_class_weight_cpu[name] = weight_tensor
            logging.info(
                "%s に有効サンプル数ベースのクラス重みを適用します (beta=%.4f)",
                name,
                float(args.class_balance_beta),
            )

    head_class_weight_device: Dict[str, torch.Tensor] = {
        name: tensor.to(device=device) for name, tensor in head_class_weight_cpu.items()
    }

    eval_batch_raw = getattr(args, "eval_batch_size", None)
    if eval_batch_raw is None or int(eval_batch_raw) <= 0:
        eval_batch_raw = args.batch_size
    eval_batch = max(1, int(eval_batch_raw))
    eval_batch = min(eval_batch, int(args.max_batch))

    if device.type == "cuda" and args.batch_size < 8192:
        logging.warning("WARNING: Small batch on CUDA; increase --batch-size to reduce host↔device overhead.")

    logging.info(
        "Training start: backend=torch device=%s amp=%s batch=%d eval_batch=%d metrics_on_gpu=yes",
        device,
        amp_mode,
        args.batch_size,
        eval_batch,
    )

    non_blocking = device.type == "cuda"

    focal_heads = set(getattr(args, "loss_focal_heads", set()))
    focal_losses: Dict[str, FocalLoss] = {}
    for name in active_heads:
        if name in focal_heads:
            focal_module = FocalLoss(
                gamma=float(args.focal_gamma),
                alpha=args.focal_alpha,
                reduction="none",
            ).to(device)
            focal_losses[name] = focal_module

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_train_metrics: Dict[str, Dict[str, float]] = {}
    best_val_metrics: Dict[str, Dict[str, float]] = {}
    best_metric = -np.inf
    patience_counter = 0
    train_count = len(train_features)
    rng = np.random.default_rng(args.seed)
    total_batches = max(1, (train_count + args.batch_size - 1) // args.batch_size)

    hard_ratio = float(np.clip(getattr(args, "reorder_hardratio", 0.0), 0.0, 1.0))
    hard_mult = float(max(1.0, getattr(args, "reorder_hardmult", 1.0)))
    use_hard_mining = "reorder" in active_heads and hard_ratio > 0.0 and hard_mult > 1.0
    hard_loss_buffer = np.zeros((train_count,), dtype=np.float32) if use_hard_mining else None
    hard_sample_weights = np.ones((train_count,), dtype=np.float32) if use_hard_mining else None

    for epoch in range(args.epochs):
        indices = rng.permutation(train_count)
        progress = ProgressReporter(
            total_batches,
            f"エポック {epoch + 1:03d}",
            unit="バッチ",
            enable=ENABLE_PROGRESS,
        )

        model.train()
        model_forward.train()
        epoch_loss = 0.0
        sample_seen = 0
        batches = [
            np.ascontiguousarray(indices[start : min(train_count, start + args.batch_size)], dtype=np.int64)
            for start in range(0, train_count, args.batch_size)
        ]
        for batch_no, (batch_idx, features_np, targets_np) in enumerate(
            _prefetch_training_batches(train_features, train_targets, active_heads, batches), start=1
        ):
            xb_cpu = torch.from_numpy(features_np)
            if non_blocking:
                xb_cpu = xb_cpu.pin_memory()
            xb = xb_cpu.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            target_tensors: Dict[str, torch.Tensor] = {}
            for name in active_heads:
                target_cpu = torch.from_numpy(targets_np[name])
                if non_blocking:
                    target_cpu = target_cpu.pin_memory()
                target_tensors[name] = target_cpu.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            if device.type == "cuda":
                assert xb.is_cuda, "Input batch not on CUDA"
                for tensor in target_tensors.values():
                    assert tensor.is_cuda, "Target tensor not on CUDA"
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp_mode):
                logits = model_forward(xb)
                loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
                for name in active_heads:
                    _logits = logits[name]
                    weight = float(head_loss_w.get(name, 1.0))
                    if weight <= 0.0:
                        continue
                    targets_h = target_tensors[name]
                    class_weight = head_class_weight_device.get(name)
                    if name in focal_losses:
                        hard_np = np.asarray(train_best_np[name][batch_idx], dtype=np.int64)
                        hard_cpu = torch.from_numpy(hard_np)
                        if non_blocking:
                            hard_cpu = hard_cpu.pin_memory()
                        hard_labels = hard_cpu.to(device=device, dtype=torch.long, non_blocking=non_blocking)
                        logits_for_loss = _logits
                        if (
                            name == "reorder"
                            and getattr(model, "reorder_use_cosine", False)
                        ):
                            scale = float(getattr(model, "reorder_scale", 1.0))
                            arcface_module = getattr(model, "reorder_arcface", None)
                            if arcface_module is not None and scale > 0.0:
                                cosine = torch.clamp(_logits / scale, -1.0 + 1e-7, 1.0 - 1e-7)
                                logits_for_loss = arcface_module(cosine, hard_labels)
                        loss_vec = focal_losses[name](logits_for_loss, hard_labels, weight=class_weight)
                        if use_hard_mining and name == "reorder" and hard_loss_buffer is not None:
                            hard_loss_buffer[batch_idx] = loss_vec.detach().cpu().numpy()
                        if (
                            use_hard_mining
                            and name == "reorder"
                            and epoch >= 1
                            and hard_sample_weights is not None
                        ):
                            weights_np = hard_sample_weights[batch_idx]
                            weights_cpu = torch.from_numpy(weights_np.astype(np.float32, copy=False))
                            if non_blocking:
                                weights_cpu = weights_cpu.pin_memory()
                            weights_dev = weights_cpu.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                            loss_vec = loss_vec * weights_dev
                        loss_h = loss_vec.mean()
                    else:
                        log_probs = torch_F.log_softmax(_logits, dim=1)
                        loss_vec = torch_F.kl_div(log_probs, targets_h, reduction="none").sum(dim=1)
                        if (
                            name == "filter_perm"
                            and args.perm_class_balance == "effective"
                            and perm_weight_device is not None
                        ):
                            true_ids = torch.argmax(targets_h, dim=1)
                            cls_weight = perm_weight_device[true_ids]
                            loss_vec = loss_vec * cls_weight
                        elif class_weight is not None:
                            true_ids = torch.argmax(targets_h, dim=1)
                            loss_vec = loss_vec * class_weight[true_ids]
                        if use_hard_mining and name == "reorder" and hard_loss_buffer is not None:
                            hard_loss_buffer[batch_idx] = loss_vec.detach().cpu().numpy()
                        if (
                            use_hard_mining
                            and name == "reorder"
                            and epoch >= 1
                            and hard_sample_weights is not None
                        ):
                            weights_np = hard_sample_weights[batch_idx]
                            weights_cpu = torch.from_numpy(weights_np.astype(np.float32, copy=False))
                            if non_blocking:
                                weights_cpu = weights_cpu.pin_memory()
                            weights_dev = weights_cpu.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                            loss_vec = loss_vec * weights_dev
                        loss_h = loss_vec.mean()
                    loss_tensor += weight * loss_h

            loss_tensor.backward()
            optimizer.step()
            batch_size_actual = batch_idx.shape[0]
            loss_scalar = float(loss_tensor.detach().item())
            epoch_loss += loss_scalar * batch_size_actual
            sample_seen += batch_size_actual
            progress.update(1)

        progress.close()

        mean_loss = epoch_loss / max(1, sample_seen)

        logging.info(
            "訓練ロジット算出開始: device=%s amp=%s batch=%d metrics_on_gpu=yes",
            device,
            amp_mode,
            eval_batch,
        )
        train_metrics = compute_metrics_on_device_with_progress(
            model_forward,
            train_features,
            device,
            train_best_np,
            train_second_np,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="訓練ロジット算出中",
            condition_dim=cond_extra_dim,
            scramble_cond=args.eval_scramble_cond,
        )
        for cond_name in condition_heads:
            head_metrics = train_metrics.get(cond_name)
            if head_metrics:
                logging.info(
                    "train %s: top1=%.2f%% top3=%.2f%%",
                    cond_name,
                    head_metrics.get("top1", float("nan")) * 100.0,
                    head_metrics.get("top3", float("nan")) * 100.0,
                )
        logging.info(
            "評価ロジット算出開始: device=%s amp=%s batch=%d metrics_on_gpu=yes",
            device,
            amp_mode,
            eval_batch,
        )
        val_metrics = compute_metrics_on_device_with_progress(
            model_forward,
            val_features,
            device,
            val_best_np,
            val_second_np,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="評価ロジット算出中",
            condition_dim=cond_extra_dim,
            scramble_cond=args.eval_scramble_cond,
        )
        for cond_name in condition_heads:
            head_metrics = val_metrics.get(cond_name)
            if head_metrics:
                logging.info(
                    "val %s: top1=%.2f%% top3=%.2f%%",
                    cond_name,
                    head_metrics.get("top1", float("nan")) * 100.0,
                    head_metrics.get("top3", float("nan")) * 100.0,
                )

        mean_top3 = float(np.mean([val_metrics[name]["top3"] for name in head_list]))
        mean_three_choice = float(np.mean([val_metrics[name]["three_choice"] for name in head_list]))
        mean_top1 = float(np.mean([val_metrics[name]["top1"] for name in head_list]))
        mean_top2 = float(np.mean([val_metrics[name]["top2"] for name in head_list]))
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

        try:
            scheduler.step(mean_top3)
        except Exception:
            pass  # スケジューラ更新で例外が発生した場合は学習を継続する

        if use_hard_mining and hard_loss_buffer is not None and hard_sample_weights is not None:
            valid_mask = np.isfinite(hard_loss_buffer)
            valid_indices = np.nonzero(valid_mask)[0]
            if valid_indices.size == 0:
                hard_sample_weights.fill(1.0)
            else:
                valid_losses = hard_loss_buffer[valid_mask]
                select_count = max(1, int(round(valid_losses.size * hard_ratio)))
                select_count = min(select_count, valid_losses.size)
                if select_count <= 0:
                    hard_sample_weights.fill(1.0)
                else:
                    order = np.argpartition(valid_losses, -select_count)[-select_count:]
                    hard_indices = valid_indices[order]
                    hard_sample_weights.fill(1.0)
                    hard_sample_weights[hard_indices] = hard_mult
                    logging.info(
                        "ハードマイニング: 上位 %.2f%% の %d 件を倍率 %.2f で再重み付け", 
                        hard_ratio * 100.0,
                        int(select_count),
                        hard_mult,
                    )

        if mean_top3 > best_metric + 1e-4:
            best_metric = mean_top3
            state_source = (
                model_forward._orig_mod if hasattr(model_forward, "_orig_mod") else model_forward
            )
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in state_source.state_dict().items()
            }
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("早期終了: 改善が見られませんでした")
                break

    if best_state is not None:
        target_model = (
            model_forward._orig_mod if hasattr(model_forward, "_orig_mod") else model_forward
        )
        target_model.load_state_dict(best_state)
        if model is not target_model:
            model.load_state_dict(best_state)

    dump_dir = getattr(args, "dump_logits", None)
    if dump_dir is not None:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        dump_logits_to_npz(
            model_forward,
            train_features,
            device,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="訓練ロジットダンプ中",
            output_path=dump_path / "train_logits.npz",
        )
        dump_logits_to_npz(
            model_forward,
            val_features,
            device,
            batch_size=eval_batch,
            amp=amp_mode,
            desc="評価ロジットダンプ中",
            output_path=dump_path / "val_logits.npz",
        )

    return model, best_train_metrics, best_val_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLG8 マルチタスク分類モデル学習ツール")
    parser.add_argument("inputs", nargs="+", help="JSONL 形式の学習データまたはディレクトリ")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch", help="学習バックエンド")
    parser.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正則化係数")
    parser.add_argument("--batch-size", type=int, default=512, help="ミニバッチサイズ")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="評価時のミニバッチサイズ (未指定時は学習バッチと同じ)",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")
    parser.add_argument(
        "--condition-heads",
        type=str,
        default="",
        help="条件付き入力として利用するヘッド名をカンマ区切りで指定 (例: interleave)",
    )
    parser.add_argument(
        "--condition-encoding",
        choices=["onehot", "id"],
        default="onehot",
        help="条件付きヘッドを結合する際のエンコーディング方式",
    )
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[1024, 512, 256], help="(NumPy バックエンド用) 隠れ層ユニット数")
    parser.add_argument(
        "--epsilon-soft",
        type=float,
        default=0.2,
        help="ソフトターゲットを一様分布と混合する割合 (0.0-0.5 程度)",
    )
    parser.add_argument(
        "--epsilon-soft-by-head",
        type=str,
        default="",
        help="ヘッド別の epsilon 指定 (例: reorder=0.02,default=0.05)",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="評価データ比率")
    parser.add_argument("--patience", type=int, default=20, help="早期終了の待機エポック数")
    parser.add_argument(
        "--eval-scramble-cond",
        action="store_true",
        help="評価時に条件特徴量をバッチ内でランダムシャッフルしてリークを検知",
    )
    parser.add_argument("--export-dir", type=Path, help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--init-from", type=Path, default=None, help="保存済みPyTorchモデルから初期化")
    parser.add_argument(
        "--head-loss-weights",
        type=str,
        default="predictor=1.2,filter_primary=1.2,filter_secondary=1.0,reorder=0.6,interleave=0.6",
        help="ヘッド別の損失重み (例: predictor=1.0,filter_primary=1.0)",
    )
    parser.add_argument(
        "--loss-focal-heads",
        type=str,
        default="",
        help="Focal Loss を適用するヘッド名 (例: reorder)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.8,
        help="Focal Loss の gamma 値",
    )
    parser.add_argument(
        "--focal-alpha",
        type=str,
        default="0.25",
        help="Focal Loss の alpha 値 (none で無効)",
    )
    parser.add_argument(
        "--perm-class-balance",
        choices=["none", "effective"],
        default="none",
        help="filter_perm ヘッド向けクラス重み方式 (loss 重み > 0 の場合のみ有効)",
    )
    parser.add_argument(
        "--perm-class-beta",
        type=float,
        default=0.999,
        help="有効サンプル数ベースのクラス重みに用いる beta 値",
    )
    parser.add_argument(
        "--class-balance-heads",
        type=str,
        default="",
        help="有効サンプル数によるクラス重みを適用するヘッド名",
    )
    parser.add_argument(
        "--class-balance-beta",
        type=float,
        default=0.999,
        help="クラス重み算出時の beta 値",
    )
    parser.add_argument(
        "--ranker-mode",
        action="store_true",
        help="filter_perm の予測/損失を無効化し、ランカー系ヘッドのみ学習する",
    )

    parser.add_argument("--temperature", type=float, default=1.0, help="推論時の温度パラメータ")
    parser.add_argument("--index-cache", type=str, default=None, help="(file_id, offset) を格納する NPY キャッシュパス")
    parser.add_argument(
        "--features-npy",
        type=str,
        default=None,
        help="事前計算済み特徴量行列 (.npy) のパス",
    )
    parser.add_argument(
        "--features-version",
        type=int,
        default=0,
        help="優先的に利用する特徴量バージョン番号 (0 で自動)",
    )
    parser.add_argument(
        "--features-autodetect",
        type=_str_to_bool,
        default=True,
        help="--features-npy 未指定時に既定パスから自動検出するか",
    )
    parser.add_argument(
        "--features-cache-key",
        type=str,
        default="",
        help="同一特徴量で複数実験を区別する任意のキー",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(".cache/ranker.features.v2.npy"),
        help="JSONL から抽出した特徴量をキャッシュする NPY ファイルへのパス",
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        default=None,
        help="特徴量キャッシュのメタデータパス (未指定時は <feature-cache>.meta.json)",
    )
    parser.add_argument("--labels-meta", type=Path, default=Path(".cache/labels.meta.json"), help="ラベルキャッシュのメタデータパス")
    parser.add_argument("--labels-bin", type=Path, default=Path(".cache/labels.bin"), help="ラベルキャッシュのバイナリパス")
    parser.add_argument(
        "--labels-topk",
        type=Path,
        default=None,
        help="C++ ラベルキャッシュが出力する topK バイナリのパス (未指定時は labels-bin から派生)",
    )
    parser.add_argument("--build-label-cache", action="store_true", help="キャッシュ不整合時に事前抽出を自動実行")
    parser.add_argument("--no-label-cache", action="store_true", help="JSONL からのラベル読み出しを強制")
    parser.add_argument(
        "--feature-stats-bin",
        type=Path,
        default=None,
        help="事前計算した特徴量統計バイナリのパス",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="特徴量スケーラーを再計算して --scaler を上書き保存",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path(".cache/ranker.scaler.npz"),
        help="特徴量スケーラー (.npz) のパス。既存ファイルを読み込み、無効なら再計算して上書き",
    )
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch バックエンド用デバイス優先度 (auto/cuda[:index]/xpu/mps/cpu)",
    )
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "none", "auto"],
        default="auto",
        help="PyTorch バックエンドで利用する自動混合精度",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile を試行する")
    parser.add_argument(
        "--dump-logits",
        type=Path,
        default=None,
        help="評価ロジットを NPZ として保存するディレクトリ (明示指定時のみ CPU 転送)",
    )
    parser.add_argument(
        "--fast-preproc",
        dest="fast_preproc",
        action="store_true",
        default=True,
        help="C++ 製の高速前処理パイプラインを利用する",
    )
    parser.add_argument("--no-fast-preproc", dest="fast_preproc", action="store_false", help="C++ 前処理を無効化")
    parser.add_argument(
        "--reorder-cosine",
        action="store_true",
        help="reorder ヘッドで CosineLinear を利用する",
    )
    parser.add_argument(
        "--reorder-arcface-m",
        type=float,
        default=0.0,
        help="ArcFace のマージン m (0 で無効)",
    )
    parser.add_argument(
        "--reorder-arcface-s",
        type=float,
        default=30.0,
        help="Cosine 出力に掛けるスケール s",
    )
    parser.add_argument(
        "--reorder-hardratio",
        type=float,
        default=0.0,
        help="損失上位サンプルを再重み付けする割合",
    )
    parser.add_argument(
        "--reorder-hardmult",
        type=float,
        default=1.0,
        help="ハードサンプルに掛ける損失倍率",
    )
    parser.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="トランク層を凍結してヘッドのみ更新する",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.feature_meta is None and args.feature_cache is not None:
        args.feature_meta = _feature_cache_meta_path(Path(args.feature_cache))
    if args.labels_topk is None:
        labels_bin_path = Path(args.labels_bin)
        if labels_bin_path.suffix:
            args.labels_topk = labels_bin_path.with_suffix(labels_bin_path.suffix + ".topk.bin")
        else:
            args.labels_topk = labels_bin_path.parent / f"{labels_bin_path.name}.topk.bin"

    try:
        head_loss_dict = parse_head_loss_weights(args.head_loss_weights)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    args.head_loss_weights = head_loss_dict
    known_heads = set(HEAD_ORDER) | {"filter_perm"}
    for name in head_loss_dict:
        if name not in known_heads:
            logging.warning("--head-loss-weights に未知のヘッド名が指定されています: %s", name)

    try:
        focal_heads = parse_head_list(args.loss_focal_heads)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    args.loss_focal_heads = set(focal_heads)

    try:
        epsilon_map = parse_head_value_map(args.epsilon_soft_by_head, allow_default=True)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    args.epsilon_soft_map = epsilon_map

    try:
        class_balance_heads = parse_head_list(args.class_balance_heads)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    args.class_balance_heads = set(class_balance_heads)

    try:
        args.focal_alpha = parse_optional_float(args.focal_alpha)
    except ValueError as exc:
        print(f"エラー: Focal alpha の解析に失敗しました: {exc}", file=sys.stderr)
        return 1

    if args.features_version < 0:
        print("エラー: --features-version には 0 以上を指定してください", file=sys.stderr)
        return 1
    features_npy_arg = (args.features_npy or "").strip()
    args.features_npy = features_npy_arg or None
    args.features_cache_key = (args.features_cache_key or "").strip()

    if args.ranker_mode:
        logging.info("ランカーモードを有効化: filter_perm ヘッドを無効化します")
        args.head_loss_weights["filter_perm"] = 0.0

    backend = args.backend.lower()
    if backend == "torch" and torch is not None:
        args.device_resolved = pick_device(args.device)
    eval_batch_display = args.eval_batch_size or args.batch_size
    print(
        f"INFO: Backend={args.backend}, Device={args.device}, AMP={args.amp}, "
        f"Batch={args.batch_size}, EvalBatch={eval_batch_display}"
    )
    if torch is not None:
        cuda_available = torch.cuda.is_available()
        try:
            cuda_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        except Exception:
            cuda_name = "N/A"
    else:
        cuda_available = False
        cuda_name = "N/A"
    print(f"INFO: torch.cuda.is_available()={cuda_available}, cuda_device={cuda_name}")
    if torch is not None:
        try:
            torch.set_num_threads(1)
        except Exception:
            logging.debug("torch.set_num_threads の設定に失敗しました", exc_info=True)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                logging.debug("torch.set_num_interop_threads の設定に失敗しました", exc_info=True)

    cond_raw = args.condition_heads or ""
    cond_guard = {item.strip() for item in cond_raw.split(",") if item.strip()}
    leak_guard_set = {"filter_perm", "perm", "filter_primary", "filter_secondary", "predictor", "reorder"}
    illegal = cond_guard & leak_guard_set
    if illegal:
        print(
            "エラー: Leak: --condition-heads must not include predictive/unknown labels: "
            f"{sorted(illegal)}",
            file=sys.stderr,
        )
        return 1
    try:
        cond_heads = parse_condition_head_list(args.condition_heads)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    condition_encoding = args.condition_encoding.lower()
    args.condition_encoding = condition_encoding
    disabled_heads_set: set[str] = set()
    active_heads = list(HEAD_ORDER)
    for name in cond_heads:
        if name not in active_heads:
            active_heads.append(name)
    filter_perm_weight = float(args.head_loss_weights.get("filter_perm", 1.0))
    if (args.ranker_mode or filter_perm_weight <= 0.0) and "filter_perm" in HEAD_ORDER:
        disabled_heads_set.add("filter_perm")
    try:
        cond_specs = [(name, HEAD_SPECS[name]) for name in cond_heads]
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    if disabled_heads_set:
        active_heads = [name for name in active_heads if name not in disabled_heads_set]
    disabled_heads_tuple: Tuple[str, ...] = ()
    if disabled_heads_set:
        disabled_heads_tuple = tuple(name for name in HEAD_ORDER if name in disabled_heads_set)
    if cond_heads:
        logging.info(
            "Conditioned heads: %s (encoding: %s)",
            ", ".join(cond_heads),
            condition_encoding,
        )
    else:
        logging.info("Conditioned heads: (none)")
    logging.info("Active heads for training: %s", ", ".join(active_heads))

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

    force_recompute = bool(args.recompute)

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

    fast_label_store: Optional[object] = None
    if not args.no_label_cache:
        fast_label_store = try_load_label_cache(
            args.labels_meta,
            args.labels_bin,
            file_refs,
            index.shape[0],
            show_progress=bool(args.progress),
        )
        if fast_label_store is None and args.fast_preproc:
            try:
                ensure_ranker_label_cache_cpp(
                    FAST_LABELCACHE_TOOL,
                    input_paths,
                    meta_path=args.labels_meta,
                    bin_path=args.labels_bin,
                    topk_path=args.labels_topk,
                    heads=RANKER_DEFAULT_HEADS,
                    topk=RANKER_DEFAULT_TOPK,
                )
            except FileNotFoundError as exc:
                logging.warning("C++ ラベルキャッシュツールが見つかりません: %s", exc)
            except Exception as exc:
                logging.warning("C++ ラベルキャッシュ生成に失敗しました: %s", exc)
            else:
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
    total = int(index.shape[0])
    features_path_resolved: Optional[Path] = None
    features_checksum = ""
    precomputed_features: Optional[np.ndarray] = None
    feature_dim = 0
    if args.features_npy is not None or args.features_version > 0 or bool(args.features_autodetect):
        try:
            resolved_path = find_features_path(args.features_npy, args.features_version, bool(args.features_autodetect))
        except FileNotFoundError as exc:
            if args.features_npy is not None:
                print(f"エラー: 特徴量 NPY が見つかりません: {exc}", file=sys.stderr)
                return 1
            logging.info("特徴量 NPY を自動検出できなかったため JSON から復元します")
        else:
            features_path_resolved = Path(resolved_path)
            try:
                precomputed_features = load_feature_matrix(features_path_resolved, mmap=True)
            except ValueError as exc:
                print(f"エラー: 特徴量 NPY の読み込みに失敗しました: {exc}", file=sys.stderr)
                return 1
            if precomputed_features.shape[0] != total:
                print(
                    f"エラー: 特徴量 NPY の行数 {precomputed_features.shape[0]} がデータセット {total} と一致しません",
                    file=sys.stderr,
                )
                return 1
            feature_dim = int(precomputed_features.shape[1])
            features_checksum = sha256_file(features_path_resolved)
            logging.info("NPY 特徴量を使用します: %s (shape=%d×%d)", features_path_resolved, total, feature_dim)
    if feature_dim <= 0:
        sample = reader.read_line(int(index[0, 0]), int(index[0, 1]))
        feature_dim = record_to_feature(sample).shape[0]
        logging.info("特徴量を JSON から動的に再計算します (dim=%d)", feature_dim)
        features = LazyFeatures(reader, index, feature_dim, max_workers=MAX_DECODE_THREADS)
    else:
        assert precomputed_features is not None
        features = _ArrayFeatureSource(precomputed_features)
        if args.feature_cache is not None:
            logging.info("事前計算済み特徴量を利用するため --feature-cache 指定を無効化します")
            args.feature_cache = None
            args.feature_meta = None

    if fast_label_store is not None:
        best_labels = {name: fast_label_store.make(name, use_second=False) for name in HEAD_ORDER}
        second_labels = {name: fast_label_store.make(name, use_second=True) for name in HEAD_ORDER}
    else:
        best_labels = {name: LazyLabels(reader, index, name, use_second=False) for name in HEAD_ORDER}
        second_labels = {name: LazyLabels(reader, index, name, use_second=True) for name in HEAD_ORDER}
    train_idx, val_idx = split_dataset(total, args.test_ratio, args.seed)

    cond_best_sources = {name: best_labels[name] for name in cond_heads}
    cond_extra_dim = conditioned_extra_dim(cond_heads, args.condition_encoding)
    expected_feature_dim = feature_dim + cond_extra_dim

    feature_cache_array: Optional[np.memmap] = None
    feature_cache_preloaded: Optional[np.memmap] = None
    feature_cache_path = Path(args.feature_cache) if args.feature_cache is not None else None
    feature_meta_path = Path(args.feature_meta) if args.feature_meta is not None else None
    if feature_cache_path is not None and feature_meta_path is None:
        feature_meta_path = _feature_cache_meta_path(feature_cache_path)
    if (
        feature_cache_path is not None
        and feature_meta_path is not None
        and args.fast_preproc
        and not cond_heads
    ):
        feature_cache_preloaded = _try_load_feature_cache(
            feature_cache_path,
            file_refs,
            expected_records=total,
            expected_dim=expected_feature_dim,
            cond_heads=cond_heads,
            cond_encoding=args.condition_encoding,
            meta_path=feature_meta_path,
        )
        if feature_cache_preloaded is None:
            try:
                ensure_ranker_feature_cache_cpp(
                    FAST_FEATCACHE_TOOL,
                    input_paths,
                    out_npy=feature_cache_path,
                    out_scaler=Path(args.scaler),
                    out_meta=feature_meta_path,
                )
            except FileNotFoundError as exc:
                logging.warning("C++ 特徴量キャッシュツールが見つかりません: %s", exc)
            except Exception as exc:
                logging.warning("C++ 特徴量キャッシュ生成に失敗しました: %s", exc)
            else:
                feature_cache_preloaded = _try_load_feature_cache(
                    feature_cache_path,
                    file_refs,
                    expected_records=total,
                    expected_dim=expected_feature_dim,
                    cond_heads=cond_heads,
                    cond_encoding=args.condition_encoding,
                    meta_path=feature_meta_path,
                )
    if feature_cache_path is not None:
        cond_arrays_full: Dict[str, np.ndarray] = {}
        for name in cond_heads:
            cond_arrays_full[name] = np.asarray(cond_best_sources[name][:], dtype=np.int16)
        if feature_cache_preloaded is not None:
            feature_cache_array = feature_cache_preloaded
        else:
            feature_cache_array = load_or_build_feature_cache(
                feature_cache_path,
                file_refs,
                features,
                cond_arrays=cond_arrays_full,
                cond_specs=cond_specs,
                cond_encoding=args.condition_encoding,
                cond_heads=cond_heads,
                total=total,
                expected_dim=expected_feature_dim,
                chunk_size=max(args.batch_size, args.max_batch),
                meta_path=feature_meta_path,
            )
        train_features = LazyFeatureSubset(feature_cache_array, train_idx)
        val_features = LazyFeatureSubset(feature_cache_array, val_idx)
        final_feature_dim = int(feature_cache_array.shape[1])
    else:
        cond_best_train: Dict[str, np.ndarray] = {}
        cond_best_val: Dict[str, np.ndarray] = {}
        for name in cond_heads:
            cond_best_train[name] = np.asarray(cond_best_sources[name][train_idx], dtype=np.int16)
            cond_best_val[name] = np.asarray(cond_best_sources[name][val_idx], dtype=np.int16)

        train_features_base = LazyFeatureSubset(features, train_idx)
        val_features_base = LazyFeatureSubset(features, val_idx)
        train_features = AugmentedFeatures(
            train_features_base,
            cond_best_train,
            cond_specs,
            encoding=args.condition_encoding,
        )
        val_features = AugmentedFeatures(
            val_features_base,
            cond_best_val,
            cond_specs,
            encoding=args.condition_encoding,
        )
        final_feature_dim = int(train_features.shape[1])
        if expected_feature_dim != final_feature_dim:
            logging.debug(
                "条件付き特徴量次元が想定値と一致しません (expected=%d, actual=%d)",
                expected_feature_dim,
                final_feature_dim,
            )
    cond_desc = f"heads={','.join(cond_heads) if cond_heads else '(none)'} encoding={args.condition_encoding}"

    mean_std_loaded = False
    stats_count: Optional[int] = None
    scaler_source = ""
    mean = np.zeros((final_feature_dim,), dtype=np.float32)
    std = np.ones((final_feature_dim,), dtype=np.float32)

    scaler_path: Optional[Path] = None
    scaler_export_required = False
    if args.scaler is not None:
        scaler_path = Path(args.scaler)
        scaler_export_required = True
        if scaler_path.exists():
            try:
                with np.load(scaler_path) as data:
                    if 'mean' not in data or 'std' not in data:
                        raise ValueError("インポートしたスケーラーに mean/std が含まれていません")
                    loaded_mean = np.asarray(data['mean'], dtype=np.float32)
                    loaded_std = np.asarray(data['std'], dtype=np.float32)
            except (OSError, ValueError) as exc:
                logging.warning(
                    "スケーラーファイル %s の読み込みに失敗しました (%s)。再計算します。",
                    scaler_path,
                    exc,
                )
            else:
                if loaded_mean.shape[0] != final_feature_dim or loaded_std.shape[0] != final_feature_dim:
                    logging.warning(
                        "インポートしたスケーラーの次元が一致しません (expected=%d, actual_mean=%d, actual_std=%d, 条件=%s)。再計算します。",
                        final_feature_dim,
                        loaded_mean.shape[0],
                        loaded_std.shape[0],
                        cond_desc,
                    )
                else:
                    mean = loaded_mean
                    std = loaded_std
                    mean_std_loaded = True
                    scaler_export_required = False
                    scaler_source = f"scaler:{scaler_path}"
                    logging.info(
                        "特徴量スケーラーを %s からインポートしました (次元=%d)",
                        scaler_path,
                        final_feature_dim,
                    )
        else:
            logging.info(
                "スケーラーファイル %s が存在しません。再計算して書き出します。",
                scaler_path,
            )

    if not mean_std_loaded and args.feature_stats_bin is not None:
        try:
            loaded_mean, loaded_std, stats_count = load_feature_scaler_from_binary(
                args.feature_stats_bin, final_feature_dim
            )
        except FileNotFoundError:
            logging.warning("特徴量統計 %s が見つかりません", args.feature_stats_bin)
        except FeatureStatsDimensionError as exc:
            logging.warning(
                "特徴量統計の次元が一致しません (expected=%d, actual=%d, 条件=%s)。ストリーミング算出にフォールバックします。",
                exc.expected,
                exc.actual,
                cond_desc,
            )
        except ValueError as exc:
            logging.warning("特徴量統計の読み込みに失敗しました: %s", exc)
        else:
            mean = loaded_mean
            std = loaded_std
            mean_std_loaded = True
            scaler_source = f"feature-stats-bin:{args.feature_stats_bin}"
            logging.info(
                "特徴量スケーラーを %s から読み込みました (サンプル数=%d)",
                args.feature_stats_bin,
                stats_count,
            )
            if stats_count is not None and stats_count != total:
                logging.warning(
                    "特徴量統計のサンプル数 %d がデータセット行数 %d と一致しません",
                    stats_count,
                    total,
                )

    if force_recompute:
        if mean_std_loaded:
            logging.info("--recompute 指定によりスケーラーを再計算します")
        mean_std_loaded = False
        scaler_export_required = True

    if not mean_std_loaded:
        if precomputed_features is not None:
            mean, std, stats_count = compute_feature_mean_std(precomputed_features)
            mean_std_loaded = True
            scaler_export_required = True
            if features_path_resolved is not None:
                scaler_source = f"features-npy:{features_path_resolved}"
            else:
                scaler_source = "features-npy"
            logging.info(
                "特徴量スケーラーを NPY から再計算しました (records=%d, dim=%d)",
                stats_count,
                final_feature_dim,
            )
        else:
            scaler_indices = np.arange(len(train_features), dtype=np.int64)
            mean, std = compute_feature_scaler_streaming(
                train_features,
                scaler_indices,
                batch_size=int(args.scaler_batch),
                show_progress=ENABLE_PROGRESS,
            )
            mean_std_loaded = True
            scaler_source = "streaming"

    if scaler_export_required and scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(scaler_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        logging.info("特徴量スケーラーを %s に書き出しました (次元=%d)", scaler_path, final_feature_dim)

    logging.info("特徴量スケーラーの利用ソース: %s (次元=%d)", scaler_source or "unknown", final_feature_dim)

    scaler_recomputed_flag = bool(
        force_recompute or scaler_export_required or scaler_source in ("streaming", "features-npy")
    )
    features_path_str = str(features_path_resolved) if features_path_resolved is not None else ""
    features_checksum_str = features_checksum
    scaler_path_str = str(scaler_path) if scaler_path is not None else ""
    base_feature_dim = max(0, final_feature_dim - cond_extra_dim)
    export_meta = {
        "features_path": features_path_str,
        "features_count": total,
        "features_dim": final_feature_dim,
        "features_base_dim": base_feature_dim,
        "features_condition_dim": int(cond_extra_dim),
        "features_checksum": features_checksum_str,
        "features_version": int(args.features_version),
        "features_autodetect": bool(args.features_autodetect),
        "features_cache_key": args.features_cache_key,
        "features_arg": args.features_npy or "",
        "scaler_path": scaler_path_str,
        "scaler_source": scaler_source or "unknown",
        "scaler_recomputed": scaler_recomputed_flag,
    }
    export_config = {
        "cli_args": sys.argv[1:],
        "features_path": features_path_str,
        "features_count": total,
        "features_dim": final_feature_dim,
        "features_base_dim": base_feature_dim,
        "features_condition_dim": int(cond_extra_dim),
        "features_version": int(args.features_version),
        "features_autodetect": bool(args.features_autodetect),
        "features_cache_key": args.features_cache_key,
        "features_arg": args.features_npy or "",
        "scaler_path": scaler_path_str,
        "recompute": force_recompute,
        "scaler_source": scaler_source or "unknown",
        "scaler_recomputed": scaler_recomputed_flag,
    }
    if features_checksum_str:
        export_config["features_checksum"] = features_checksum_str
    else:
        export_meta.pop("features_checksum", None)

    active_best_labels = {name: best_labels[name] for name in active_heads}
    active_second_labels = {name: second_labels[name] for name in active_heads}
    train_best = slice_labels_lazy(active_best_labels, train_idx)
    train_second = slice_labels_lazy(active_second_labels, train_idx)
    val_best = slice_labels_lazy(active_best_labels, val_idx)
    val_second = slice_labels_lazy(active_second_labels, val_idx)

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

        model = MultiTaskModel(
            train_features.shape[1],
            args.hidden_dims,
            args.dropout,
            mean,
            std,
            disabled_heads=disabled_heads_tuple,
        )
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
        for name in active_heads:
            print(f"{name:16s}: {format_metrics(train_metrics.get(name, {}))}")
        train_macro = macro_average(train_metrics)
        if train_macro:
            print(
                f"{'macro':16s}: top1={train_macro['top1']*100:.2f}% "
                f"top2={train_macro['top2']*100:.2f}% top3={train_macro['top3']*100:.2f}% "
                f"three={train_macro['three_choice']*100:.2f}%"
            )

        print("\n=== 評価データ精度 ===")
        for name in active_heads:
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
            _write_export_metadata(args.export_dir, export_meta, export_config)
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
                disabled_heads_tuple,
                cond_heads,
            )
        except ImportError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1

        active_heads = tuple(getattr(torch_model, "active_heads", tuple(active_heads)))
        print("\n=== 訓練データ精度 ===")
        for name in active_heads:
            print(f"{name:16s}: {format_metrics(train_metrics.get(name, {}))}")
        train_macro = macro_average(train_metrics)
        if train_macro:
            print(
                f"{'macro':16s}: top1={train_macro['top1']*100:.2f}% "
                f"top2={train_macro['top2']*100:.2f}% top3={train_macro['top3']*100:.2f}% "
                f"three={train_macro['three_choice']*100:.2f}%"
            )

        print("\n=== 評価データ精度 ===")
        for name in active_heads:
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
            _write_export_metadata(args.export_dir, export_meta, export_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
