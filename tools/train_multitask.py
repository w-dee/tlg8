#!/usr/bin/env python3
"""TLG8 マルチタスク分類モデルの学習 CLI。"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from multitask_model import HEAD_ORDER, MultiTaskModel, TrainConfig, train_multitask_model

MAX_COMPONENTS = 4
BLOCK_EDGE = 8
MAX_BLOCK_PIXELS = BLOCK_EDGE * BLOCK_EDGE

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm が利用不可の場合のフォールバック
    tqdm = None

# 遅延デコード用のスレッド上限と JSON デコード挙動を実行時に切り替えるためのグローバル。
MAX_DECODE_THREADS = 1
SKIP_BAD_RECORDS = False


def discover_input_files(inputs: Sequence[str]) -> List[Path]:
    """入力パスから学習データファイル一覧を収集する。"""

    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file():
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"警告: 入力パス '{item}' は存在しません", file=sys.stderr)
    return files


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
                bar = None
                if show_progress and tqdm is not None:
                    bar = tqdm(total=size, unit="B", unit_scale=True, desc=path.name)
                elif show_progress and tqdm is None:
                    print(f"{path.name}: インデックス構築中...", file=sys.stderr)
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    if line.strip():
                        index_list.append((file_id, offset))
                    offset = fp.tell()
                    if bar is not None:
                        bar.update(offset - last_pos)
                        last_pos = offset
                if bar is not None:
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
            return json.loads(raw.decode("utf-8").strip())
        except json.JSONDecodeError:
            handle.seek(offset)
            raw_retry = handle.readline()
            try:
                return json.loads(raw_retry.decode("utf-8").strip())
            except json.JSONDecodeError as exc:
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

    padded = np.zeros((MAX_COMPONENTS, MAX_BLOCK_PIXELS), dtype=np.float32)
    offset = 0
    for by in range(block_h):
        for bx in range(block_w):
            dest = by * BLOCK_EDGE + bx
            for comp in range(components):
                padded[comp, dest] = float(pixels[offset]) / 255.0
                offset += 1
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
        for out_idx, rec_idx in enumerate(rows):
            file_id, offset = self._index[int(rec_idx)]
            record = self._reader.read_line(int(file_id), int(offset))
            out[out_idx] = self._resolve_value(record)
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
    features: LazyFeatures, train_idx: np.ndarray, batch_size: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """訓練データの遅延読み出しで平均・標準偏差を算出する。"""

    feat_dim = features.shape[1]
    mean = np.zeros((feat_dim,), dtype=np.float64)
    m2 = np.zeros((feat_dim,), dtype=np.float64)
    count = 0
    for start in range(0, train_idx.shape[0], batch_size):
        batch_indices = train_idx[start : start + batch_size]
        batch = features[batch_indices].astype(np.float64)
        for row in batch:
            count += 1
            delta = row - mean
            mean += delta / count
            m2 += delta * (row - mean)
    if count <= 1:
        variance = np.ones_like(mean)
    else:
        variance = m2 / (count - 1)
    std = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean32 = mean.astype(np.float32)
    logging.info(
        "Scaler: L2(mean)=%.6f, min(std)=%.6f, max(std)=%.6f",
        float(np.linalg.norm(mean)),
        float(std.min()),
        float(std.max()),
    )
    return mean32, std


def slice_labels_lazy(labels: Dict[str, LazyLabels], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """遅延ラベルから指定インデックスの値をまとめて抽出する。"""

    return {name: lazy[indices] for name, lazy in labels.items()}


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLG8 マルチタスク分類モデル学習ツール")
    parser.add_argument("inputs", nargs="+", help="JSONL 形式の学習データまたはディレクトリ")
    parser.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正則化係数")
    parser.add_argument("--batch-size", type=int, default=512, help="ミニバッチサイズ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[1024, 512, 256], help="隠れ層ユニット数")
    parser.add_argument("--epsilon-soft", type=float, default=0.2, help="第 2 候補に割り当てる確率質量")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="評価データ比率")
    parser.add_argument("--patience", type=int, default=20, help="早期終了の待機エポック数")
    parser.add_argument("--export-dir", type=Path, help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--temperature", type=float, default=1.0, help="推論時の温度パラメータ")
    parser.add_argument("--index-cache", type=str, default=None, help="(file_id, offset) を格納する NPY キャッシュパス")
    parser.add_argument("--progress", action="store_true", help="JSONL インデックス構築時に進捗を表示")
    parser.add_argument("--skip-bad-records", action="store_true", help="JSON デコードに失敗した行をスキップ")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8192,
        help="遅延読み出し時の実効ミニバッチ上限",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=0,
        help="遅延デコードに利用する最大スレッド数 (0 で自動設定)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.max_batch <= 0:
        print("エラー: --max-batch には 1 以上を指定してください", file=sys.stderr)
        return 1

    global MAX_DECODE_THREADS, SKIP_BAD_RECORDS
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

    files = discover_input_files(args.inputs)
    if not files:
        print("エラー: 入力ファイルが見つかりません", file=sys.stderr)
        return 1

    cache_path = Path(args.index_cache) if args.index_cache else None
    features, best_labels, second_labels, total = open_indexed_dataset(
        files, cache_path, bool(args.progress)
    )
    train_idx, val_idx = split_dataset(total, args.test_ratio, args.seed)

    mean, std = compute_feature_scaler_streaming(features, train_idx)
    train_features = LazyFeatureSubset(features, train_idx)
    val_features = LazyFeatureSubset(features, val_idx)
    train_best = slice_labels_lazy(best_labels, train_idx)
    train_second = slice_labels_lazy(second_labels, train_idx)
    val_best = slice_labels_lazy(best_labels, val_idx)
    val_second = slice_labels_lazy(second_labels, val_idx)

    effective_batch = min(int(args.batch_size), int(args.max_batch))

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=effective_batch,
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
        print(
            f"{name:16s}: {format_metrics(train_metrics.get(name, {}))}"
        )
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
