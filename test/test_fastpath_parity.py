"""C++ 高速前処理と Python 実装の整合性を検証するテスト。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# tools/ 配下のモジュールを直接 import できるようにする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_multitask
from train_multitask import (
    HEAD_ORDER,
    JsonlReader,
    LazyFeatures,
    LazyLabels,
    RankerLabelCache,
    build_jsonl_index,
    compute_feature_scaler_streaming,
    record_to_feature,
)

# 進捗バーを抑制し、テスト環境でのスレッド数を固定する。
train_multitask.ENABLE_PROGRESS = False
train_multitask.MAX_DECODE_THREADS = 1


def _pack_filter_code(perm: int, primary: int, secondary: int) -> int:
    """フィルター構成を 1 つの整数にエンコードする。"""

    return ((perm % 6) << 4) | ((primary % 4) << 2) | (secondary % 4)


def _sample_records() -> List[Dict[str, object]]:
    """検証用の最小 JSONL レコード群を生成する。"""

    return [
        {
            "pixels": list(range(12)),
            "block_size": [2, 2],
            "components": 3,
            "best": {
                "predictor": 2,
                "reorder": 5,
                "interleave": 1,
                "filter": _pack_filter_code(3, 1, 2),
            },
            "second": {
                "predictor": 6,
                "reorder": 3,
                "interleave": 0,
                "filter": _pack_filter_code(2, 3, 1),
            },
        },
        {
            "pixels": [255 - i for i in range(12)],
            "block_size": [2, 2],
            "components": 3,
            "best": {
                "predictor": 0,
                "reorder": 7,
                "interleave": 0,
                "filter": _pack_filter_code(1, 2, 3),
            },
            "second": {
                "predictor": 1,
                "reorder": 6,
                "interleave": 1,
                "filter": _pack_filter_code(0, 1, 2),
            },
        },
        {
            "pixels": [10] * 12,
            "block_size": [2, 2],
            "components": 3,
            "best": {
                "predictor": 4,
                "reorder": 2,
                "interleave": 1,
                "filter": _pack_filter_code(5, 0, 1),
            },
            # second を省略して欠損値経路を確認する。
        },
    ]


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    """レコード群を JSONL として書き出す。"""

    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            json.dump(rec, fp, separators=(",", ":"))
            fp.write("\n")


def _close_reader(reader: JsonlReader) -> None:
    """JsonlReader が保持するファイルハンドルを明示的に閉じる。"""

    handles = getattr(reader._local, "handles", None)
    if isinstance(handles, dict):
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass


class FastpathParityTest(unittest.TestCase):
    """C++ と Python 実装の出力が一致するか確認する。"""

    def setUp(self) -> None:
        self._label_tool = PROJECT_ROOT / "build" / "cpp" / "tlg8_labelcache"
        self._feat_tool = PROJECT_ROOT / "build" / "cpp" / "tlg8_featcache"

    def test_label_cache_matches_python(self) -> None:
        """ラベル BIN が Python 実装と一致することを確認する。"""

        if not self._label_tool.exists():
            self.skipTest("tlg8_labelcache がビルドされていません")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            jsonl_path = tmp_path / "sample.jsonl"
            _write_jsonl(jsonl_path, _sample_records())

            out_bin = tmp_path / "ranker.labels.bin"
            out_meta = tmp_path / "ranker.labels.meta.json"
            out_topk = tmp_path / "ranker.topk.bin"
            heads_arg = ",".join(HEAD_ORDER)
            subprocess.run(
                [
                    str(self._label_tool),
                    "--jsonl",
                    str(jsonl_path),
                    "--out-bin",
                    str(out_bin),
                    "--out-meta",
                    str(out_meta),
                    "--out-topk",
                    str(out_topk),
                    "--topk",
                    "2",
                    "--heads",
                    heads_arg,
                ],
                check=True,
            )

            with out_meta.open("r", encoding="utf-8") as fp:
                meta = json.load(fp)

            cache = RankerLabelCache(meta, out_bin, topk_path=out_topk, topk_k=2)
            files, index = build_jsonl_index([jsonl_path], None, False)
            reader = JsonlReader(files)
            rows = np.arange(index.shape[0], dtype=np.int64)

            try:
                for head in HEAD_ORDER:
                    with self.subTest(head=head):
                        py_best = LazyLabels(reader, index, head, use_second=False)[rows]
                        cpp_best = cache.make(head, use_second=False)[rows]
                        np.testing.assert_array_equal(cpp_best, py_best)

                        py_second = LazyLabels(reader, index, head, use_second=True)[rows]
                        cpp_second = cache.make(head, use_second=True)[rows]
                        np.testing.assert_array_equal(cpp_second, py_second)
            finally:
                _close_reader(reader)
                cache.close()

    def test_feature_cache_matches_python(self) -> None:
        """特徴量 NPY とスケーラー NPZ が Python 実装と一致する。"""

        if not self._feat_tool.exists():
            self.skipTest("tlg8_featcache がビルドされていません")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            jsonl_path = tmp_path / "sample.jsonl"
            _write_jsonl(jsonl_path, _sample_records())

            feat_npy = tmp_path / "ranker.features.npy"
            feat_scaler = tmp_path / "ranker.scaler.npz"
            feat_meta = tmp_path / "ranker.features.meta.json"
            feat_idx = tmp_path / "ranker.idx"
            subprocess.run(
                [
                    str(self._feat_tool),
                    "--jsonl",
                    str(jsonl_path),
                    "--out-npy",
                    str(feat_npy),
                    "--out-scaler",
                    str(feat_scaler),
                    "--out-meta",
                    str(feat_meta),
                    "--out-idx",
                    str(feat_idx),
                ],
                check=True,
            )

            files, index = build_jsonl_index([jsonl_path], None, False)
            reader = JsonlReader(files)
            sample = reader.read_line(int(index[0, 0]), int(index[0, 1]))
            feature_dim = record_to_feature(sample).shape[0]
            lazy = LazyFeatures(reader, index, feature_dim, max_workers=1)
            rows = np.arange(index.shape[0], dtype=np.int64)

            try:
                cpp_memmap = np.load(feat_npy, allow_pickle=False)
                try:
                    cpp_features = np.asarray(cpp_memmap)
                finally:
                    del cpp_memmap

                py_features = lazy[rows]
                np.testing.assert_allclose(cpp_features, py_features, rtol=1e-6, atol=1e-7)

                scaler = np.load(feat_scaler)
                try:
                    cpp_mean = np.asarray(scaler["mean"])
                    cpp_std = np.asarray(scaler["std"])
                finally:
                    scaler.close()

                py_mean, py_std = compute_feature_scaler_streaming(
                    lazy,
                    rows,
                    batch_size=2,
                    show_progress=False,
                )
                np.testing.assert_allclose(cpp_mean, py_mean, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(cpp_std, py_std, rtol=1e-6, atol=1e-7)
            finally:
                _close_reader(reader)


if __name__ == "__main__":
    unittest.main()
