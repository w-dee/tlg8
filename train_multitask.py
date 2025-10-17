#!/usr/bin/env python3
"""TLG8 マルチタスク分類モデルの学習 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from multitask_model import HEAD_ORDER, MultiTaskModel, TrainConfig, train_multitask_model

MAX_COMPONENTS = 4
BLOCK_EDGE = 8
MAX_BLOCK_PIXELS = BLOCK_EDGE * BLOCK_EDGE


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


def load_dataset(paths: Sequence[Path]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """JSONL 形式の学習データを読み込み、特徴量と各ラベルを返す。"""

    features: List[np.ndarray] = []
    best_labels: Dict[str, List[int]] = {name: [] for name in HEAD_ORDER}
    second_labels: Dict[str, List[int]] = {name: [] for name in HEAD_ORDER}

    total_lines = 0
    loaded = 0
    report_interval = 1000

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line_no, line in enumerate(fp, 1):
                    total_lines += 1
                    record = line.strip()
                    if not record:
                        continue
                    try:
                        data = json.loads(record)
                    except json.JSONDecodeError as exc:
                        print(f"警告: {path}:{line_no} の JSON 解釈に失敗しました: {exc}", file=sys.stderr)
                        continue
                    try:
                        pixels = data["pixels"]
                        block_w, block_h = data["block_size"]
                        components = data["components"]
                        best = data["best"]
                        second = data.get("second", {})
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"警告: {path}:{line_no} のレコード形式が不正です: {exc}", file=sys.stderr)
                        continue

                    if not isinstance(pixels, list):
                        print(f"警告: {path}:{line_no} の pixels が配列ではありません", file=sys.stderr)
                        continue

                    expected_len = block_w * block_h * components
                    if len(pixels) != expected_len:
                        print(
                            f"警告: {path}:{line_no} の画素数が想定({expected_len})と異なります",
                            file=sys.stderr,
                        )
                        continue

                    padded = np.zeros((MAX_COMPONENTS, MAX_BLOCK_PIXELS), dtype=np.float32)
                    offset = 0
                    for by in range(block_h):
                        for bx in range(block_w):
                            dest = by * BLOCK_EDGE + bx
                            for comp in range(components):
                                padded[comp, dest] = pixels[offset] / 255.0
                                offset += 1
                    extra = np.array([block_w / 8.0, block_h / 8.0, components / 4.0], dtype=np.float32)
                    feature = np.concatenate([padded.reshape(-1), extra])
                    features.append(feature)

                    best_labels["predictor"].append(int(best.get("predictor", 0)))
                    best_filter = int(best.get("filter", 0))
                    perm, primary, secondary = _split_filter_code(best_filter)
                    best_labels["filter_perm"].append(perm)
                    best_labels["filter_primary"].append(primary)
                    best_labels["filter_secondary"].append(secondary)
                    best_labels["reorder"].append(int(best.get("reorder", 0)))
                    best_labels["interleave"].append(int(best.get("interleave", 0)))

                    def second_value(key: str) -> int:
                        value = second.get(key, -1)
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return -1

                    sec_pred = second_value("predictor")
                    sec_filter = second_value("filter")
                    s_perm, s_primary, s_secondary = _split_filter_code(sec_filter)
                    second_labels["predictor"].append(sec_pred)
                    second_labels["filter_perm"].append(s_perm)
                    second_labels["filter_primary"].append(s_primary)
                    second_labels["filter_secondary"].append(s_secondary)
                    second_labels["reorder"].append(second_value("reorder"))
                    second_labels["interleave"].append(second_value("interleave"))

                    loaded += 1
                    if loaded % report_interval == 0:
                        print(
                            f"読み込み中: {loaded} エントリ処理済み...",
                            file=sys.stderr,
                        )
        except OSError as exc:
            print(f"警告: {path} を読み込めません: {exc}", file=sys.stderr)

    if not features:
        raise RuntimeError("学習データが読み込めませんでした")

    feature_array = np.stack(features, axis=0)
    best_array = {name: np.asarray(values, dtype=np.int32) for name, values in best_labels.items()}
    second_array = {name: np.asarray(values, dtype=np.int32) for name, values in second_labels.items()}

    print(f"読み込み完了: {loaded} エントリ / {total_lines} 行", file=sys.stderr)
    return feature_array, best_array, second_array


def compute_feature_scaler(features: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """訓練データから平均・標準偏差を算出する。"""

    subset = features[train_idx]
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def slice_labels(labels: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """辞書形式のラベル配列をインデックス指定で抽出する。"""

    return {name: values[indices] for name, values in labels.items()}


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
    args = parser.parse_args(argv)

    files = discover_input_files(args.inputs)
    if not files:
        print("エラー: 入力ファイルが見つかりません", file=sys.stderr)
        return 1

    features, best_labels, second_labels = load_dataset(files)
    total = features.shape[0]
    train_idx, val_idx = split_dataset(total, args.test_ratio, args.seed)

    mean, std = compute_feature_scaler(features, train_idx)
    train_features = features[train_idx]
    val_features = features[val_idx]
    train_best = slice_labels(best_labels, train_idx)
    train_second = slice_labels(second_labels, train_idx)
    val_best = slice_labels(best_labels, val_idx)
    val_second = slice_labels(second_labels, val_idx)

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
