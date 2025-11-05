#!/usr/bin/env python3
"""TLG8 エンコーダー用 ANN 学習データを読み込み、高精度分類モデルを学習するスクリプト。

JSONL 形式でダンプされたブロック情報を読み込み、各圧縮フェーズ
（predictor / filter / reorder / interleave / entropy）の最適解を分類するモデルを
学習し、精度を表示する。既定では標準化とミニバッチ学習を備えた多層パーセプトロンを
利用し、95% 以上の正答率を狙った高度な学習をローカルで再現できるようにしている。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - 実行環境の依存性確認
    print(
        "エラー: numpy が見つかりません。`pip install numpy` などでインストールしてください。",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from features import compute_orientation_features

# TLG8 固有の定数
K_PREDICTOR_CLASSES = 8
K_FILTER_CLASSES = 96
K_REORDER_CLASSES = 8
K_INTERLEAVE_CLASSES = 2
K_ENTROPY_CLASSES = 2
MAX_BLOCK_PIXELS = 64
MAX_COMPONENTS = 4
EXTRA_FEATURES = 16  # block_w, block_h, components + 方位統計 13 次元
BLOCK_EDGE = 8


def discover_input_files(inputs: Sequence[str]) -> List[Path]:
    """引数で与えられたファイル / ディレクトリから読み込み対象のファイル一覧を返す。"""

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


def load_dataset(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    """JSONL 形式の学習データを読み込み、特徴量とラベルを numpy 配列で返す。"""

    features: List[np.ndarray] = []
    predictor_labels: List[int] = []
    predictor_second: List[int] = []
    filter_labels: List[int] = []
    filter_second: List[int] = []
    reorder_labels: List[int] = []
    reorder_second: List[int] = []
    interleave_labels: List[int] = []
    interleave_second: List[int] = []
    entropy_labels: List[int] = []
    entropy_second: List[int] = []

    total_lines = 0
    loaded_lines = 0

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line_no, line in enumerate(fp, 1):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"警告: {path}:{line_no} の JSON 解釈に失敗しました: {exc}", file=sys.stderr)
                        continue

                    try:
                        pixels = record["pixels"]
                        block_w, block_h = record["block_size"]
                        components = record["components"]
                        best = record["best"]
                        second = record.get("second")
                    except (KeyError, ValueError, TypeError) as exc:
                        print(f"警告: {path}:{line_no} のフォーマットが不正です: {exc}", file=sys.stderr)
                        continue

                    if not isinstance(pixels, list):
                        print(f"警告: {path}:{line_no} の pixels が配列ではありません", file=sys.stderr)
                        continue

                    value_count = block_w * block_h
                    expected_len = value_count * components
                    if len(pixels) != expected_len:
                        print(
                            f"警告: {path}:{line_no} の画素数が想定({expected_len})と一致しません",
                            file=sys.stderr,
                        )
                        continue

                    padded = np.zeros((MAX_COMPONENTS, BLOCK_EDGE, BLOCK_EDGE), dtype=np.float32)
                    arr = np.asarray(pixels, dtype=np.uint8, order="C")
                    arr = arr.reshape(block_h, block_w, components).transpose(2, 0, 1)
                    normalized = arr.astype(np.float32) / 255.0
                    padded[:components, :block_h, :block_w] = normalized
                    orientation = compute_orientation_features(padded[:components, :block_h, :block_w])
                    extra = np.array([block_w / 8.0, block_h / 8.0, components / 4.0], dtype=np.float32)
                    feature = np.concatenate([padded.reshape(-1), extra, orientation])
                    features.append(feature)

                    predictor_labels.append(int(best.get("predictor", 0)))
                    filter_labels.append(int(best.get("filter", 0)))
                    reorder_labels.append(int(best.get("reorder", 0)))
                    interleave_labels.append(int(best.get("interleave", 0)))
                    entropy_labels.append(int(best.get("entropy", 0)))

                    def second_value(key: str) -> int:
                        if isinstance(second, dict) and key in second:
                            return int(second[key])
                        return -1

                    predictor_second.append(second_value("predictor"))
                    filter_second.append(second_value("filter"))
                    reorder_second.append(second_value("reorder"))
                    interleave_second.append(second_value("interleave"))
                    entropy_second.append(second_value("entropy"))

                    loaded_lines += 1
        except OSError as exc:
            print(f"警告: {path} を読み込めません: {exc}", file=sys.stderr)

    if not features:
        raise RuntimeError("学習用データが読み込めませんでした")

    data = {
        "features": np.stack(features, axis=0),
        "predictor": np.asarray(predictor_labels, dtype=np.int32),
        "predictor_second": np.asarray(predictor_second, dtype=np.int32),
        "filter": np.asarray(filter_labels, dtype=np.int32),
        "filter_second": np.asarray(filter_second, dtype=np.int32),
        "reorder": np.asarray(reorder_labels, dtype=np.int32),
        "reorder_second": np.asarray(reorder_second, dtype=np.int32),
        "interleave": np.asarray(interleave_labels, dtype=np.int32),
        "interleave_second": np.asarray(interleave_second, dtype=np.int32),
        "entropy": np.asarray(entropy_labels, dtype=np.int32),
        "entropy_second": np.asarray(entropy_second, dtype=np.int32),
    }

    print(
        f"読み込み完了: {loaded_lines} エントリ / {total_lines} 行", file=sys.stderr
    )
    return data


def split_dataset(
    count: int, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """データセットを訓練用 / 評価用に分割する。サンプルが少ない場合は全件を共有。"""

    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=np.int64)
    if count == 0:
        return indices, indices, True
    rng.shuffle(indices)
    if count < 5:
        return indices, indices, True
    test_count = max(1, int(round(count * test_ratio)))
    test_count = min(test_count, count - 1)
    train_count = count - test_count
    return indices[:train_count], indices[train_count:], False


def softmax(logits: np.ndarray) -> np.ndarray:
    """数値安定化付きのソフトマックス。"""

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    sum_exp = exp.sum(axis=1, keepdims=True)
    return exp / sum_exp


def compute_feature_scaler(
    features: np.ndarray, standardize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """特徴量の標準化に用いる平均と標準偏差を返す。"""

    if not standardize:
        return np.zeros((features.shape[1],), dtype=np.float32), np.ones(
            (features.shape[1],), dtype=np.float32
        )

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """特徴量に標準化を適用する。"""

    return (features - mean) / std


class BaseModel:
    """学習と推論を共通化するためのベースクラス。"""

    def __init__(self, feature_dim: int, class_count: int) -> None:
        self.feature_dim = feature_dim
        self.class_count = class_count

    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        lr: float,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> float:
        raise NotImplementedError

    def logits(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class LogisticRegressionModel(BaseModel):
    """単純な多クラスロジスティック回帰モデル。"""

    def __init__(self, feature_dim: int, class_count: int, seed: int) -> None:
        super().__init__(feature_dim, class_count)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=1e-3, size=(feature_dim + 1, class_count))

    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        lr: float,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> float:
        # ロジスティック回帰は全件一括更新の方が安定するためバッチサイズは無視する。
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        logits = x_aug @ self.weights
        probs = softmax(logits)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(x.shape[0]), y] = 1.0
        diff = probs - y_onehot
        grad = x_aug.T @ diff / x.shape[0]
        grad[:-1] += weight_decay * self.weights[:-1]
        self.weights -= lr * grad
        loss = -np.mean(np.log(np.clip(probs[np.arange(x.shape[0]), y], 1e-9, 1.0)))
        if not np.isfinite(self.weights).all():
            raise RuntimeError("学習が発散しました (NaN/Inf を検知)")
        return float(loss)

    def logits(self, x: np.ndarray) -> np.ndarray:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return x_aug @ self.weights

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "model_type": np.array(["logistic"], dtype="U16"),
            "weights": self.weights.astype(np.float32),
        }


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU 活性化関数。"""

    return np.maximum(x, 0.0)


def _relu_backward(grad: np.ndarray, pre_activation: np.ndarray) -> np.ndarray:
    """ReLU の逆伝播。"""

    mask = pre_activation > 0.0
    return grad * mask


class MLPModel(BaseModel):
    """多層パーセプトロンによる分類モデル。"""

    def __init__(
        self,
        feature_dim: int,
        class_count: int,
        hidden_dims: Sequence[int],
        dropout: float,
        seed: int,
    ) -> None:
        super().__init__(feature_dim, class_count)
        if not hidden_dims:
            raise ValueError("MLP モデルには 1 層以上の隠れ層が必要です")
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        self.rng = np.random.default_rng(seed)
        dims = [feature_dim] + list(hidden_dims) + [class_count]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(2.0 / in_dim)
            self.weights.append(self.rng.normal(scale=scale, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((out_dim,), dtype=np.float64))

    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        lr: float,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> float:
        if batch_size <= 0:
            raise ValueError("batch_size は 1 以上である必要があります")
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            loss = self._train_step(xb, yb, lr, weight_decay)
            total_loss += loss * len(batch_idx)
        return float(total_loss / x.shape[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        activations = x.astype(np.float64)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            activations = activations @ W + b
            activations = _relu(activations)
        logits = activations @ self.weights[-1] + self.biases[-1]
        return logits

    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {
            "model_type": np.array(["mlp"], dtype="U16"),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int32),
            "dropout": np.asarray([self.dropout], dtype=np.float32),
        }
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            state[f"W{i}"] = W.astype(np.float32)
            state[f"b{i}"] = b.astype(np.float32)
        return state

    def _forward(
        self, x: np.ndarray, train: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        activations: List[np.ndarray] = [x.astype(np.float64)]
        pre_activations: List[np.ndarray] = []
        dropout_masks: List[np.ndarray] = []
        h = activations[0]
        for idx, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = h @ W + b
            pre_activations.append(z)
            h = _relu(z)
            if train and self.dropout > 0.0:
                mask = self.rng.random(h.shape) >= self.dropout
                h = h * mask / (1.0 - self.dropout)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(np.ones_like(h))
            activations.append(h)
        logits = h @ self.weights[-1] + self.biases[-1]
        activations.append(logits)
        return activations, pre_activations, dropout_masks

    def _train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        weight_decay: float,
    ) -> float:
        activations, pre_acts, dropout_masks = self._forward(x, train=True)
        logits = activations[-1]
        probs = softmax(logits)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(x.shape[0]), y] = 1.0
        diff = (probs - y_onehot) / x.shape[0]
        grads_W: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        grad_out = activations[-2].T @ diff
        grads_W.append(grad_out + weight_decay * self.weights[-1])
        grads_b.append(diff.sum(axis=0))

        backprop = diff @ self.weights[-1].T
        for layer in reversed(range(len(self.hidden_dims))):
            backprop = _relu_backward(backprop, pre_acts[layer])
            backprop *= dropout_masks[layer]
            grad_W = activations[layer].T @ backprop
            grad_b = backprop.sum(axis=0)
            grads_W.append(grad_W + weight_decay * self.weights[layer])
            grads_b.append(grad_b)
            if layer > 0:
                backprop = backprop @ self.weights[layer].T

        grads_W.reverse()
        grads_b.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

        loss = -np.mean(np.log(np.clip(probs[np.arange(x.shape[0]), y], 1e-9, 1.0)))
        if not all(np.isfinite(arr).all() for arr in (*self.weights, *self.biases)):
            raise RuntimeError("学習が発散しました (NaN/Inf を検知)")
        return float(loss)


def evaluate_from_logits(
    logits: np.ndarray, y_best: np.ndarray, y_second: np.ndarray
) -> Dict[str, float]:
    """ロジット値から精度を計算する。"""

    if logits.size == 0:
        return {"top1": float("nan"), "top2": float("nan"), "two_choice": float("nan")}
    pred = np.argmax(logits, axis=1)
    top1 = pred == y_best
    if logits.shape[1] >= 2:
        top2_candidates = np.argpartition(logits, kth=-2, axis=1)[:, -2:]
        top2 = np.any(top2_candidates == y_best[:, None], axis=1)
    else:
        top2 = top1
    two_choice = top1.copy()
    valid_second = y_second >= 0
    two_choice |= valid_second & (pred == y_second)
    return {
        "top1": float(top1.mean()) if top1.size else float("nan"),
        "top2": float(top2.mean()) if top2.size else float("nan"),
        "two_choice": float(two_choice.mean()) if two_choice.size else float("nan"),
    }


def model_logits_from_state(state: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """保存した重み情報からロジットを再構成する。"""

    model_type_arr = state.get("model_type")
    if model_type_arr is None:
        raise RuntimeError("model_type が保存されていません")
    model_type = model_type_arr.astype(str)[0]

    if model_type == "logistic":
        weights = state["weights"].astype(np.float64)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return x_aug @ weights

    if model_type == "mlp":
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        idx = 0
        while True:
            key_w = f"W{idx}"
            key_b = f"b{idx}"
            if key_w not in state:
                break
            weights.append(state[key_w].astype(np.float64))
            biases.append(state[key_b].astype(np.float64))
            idx += 1
        activations = x.astype(np.float64)
        for W, b in zip(weights[:-1], biases[:-1]):
            activations = activations @ W + b
            activations = _relu(activations)
        return activations @ weights[-1] + biases[-1]

    raise RuntimeError(f"未知のモデル種別です: {model_type}")


def create_model(
    model_type: str,
    feature_dim: int,
    class_count: int,
    seed: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> BaseModel:
    """モデル種別に応じてインスタンスを生成する。"""

    model_type = model_type.lower()
    if model_type == "logistic":
        return LogisticRegressionModel(feature_dim, class_count, seed)
    if model_type == "mlp":
        dims = hidden_dims if hidden_dims else [512, 256]
        return MLPModel(feature_dim, class_count, dims, dropout, seed)
    raise ValueError(f"未知のモデル種別です: {model_type}")


def stage_training(
    name: str,
    data: Dict[str, np.ndarray],
    test_ratio: float,
    epochs: int,
    lr: float,
    reg: float,
    seed: int,
    class_count: int,
    model_type: str,
    hidden_dims: Sequence[int],
    dropout: float,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float], bool]:
    """各フェーズの学習と評価を行う。"""

    x = apply_feature_scaler(data["features"].astype(np.float64), mean, std)
    y = data[name]
    y_second = data[f"{name}_second"]

    train_idx, test_idx, shared = split_dataset(len(y), test_ratio, seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    y_second_train = y_second[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    y_second_test = y_second[test_idx]

    unique_classes = np.unique(y_train)
    if unique_classes.size <= 1:
        weights = np.zeros((x.shape[1] + 1, class_count), dtype=np.float64)
        default_class = int(unique_classes[0]) if unique_classes.size else 0
        weights[-1, default_class] = 1.0
        state = {
            "model_type": np.array(["logistic"], dtype="U16"),
            "weights": weights.astype(np.float32),
        }
        train_metrics = evaluate_from_logits(model_logits_from_state(state, x_train), y_train, y_second_train)
        test_metrics = evaluate_from_logits(model_logits_from_state(state, x_test), y_test, y_second_test)
        return state, train_metrics, test_metrics, True or shared

    model = create_model(
        model_type,
        feature_dim=x.shape[1],
        class_count=class_count,
        seed=seed,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    rng = np.random.default_rng(seed)
    best_state: Optional[Dict[str, np.ndarray]] = None
    best_metric = -float("inf")
    stale = 0
    patience = max(3, epochs // 10)

    for epoch in range(epochs):
        start = time.perf_counter()
        loss = model.train_epoch(x_train, y_train, batch_size, lr, reg, rng)
        logits_train = model.logits(x_train)
        logits_test = model.logits(x_test)
        train_metrics = evaluate_from_logits(logits_train, y_train, y_second_train)
        test_metrics = evaluate_from_logits(logits_test, y_test, y_second_test)
        elapsed = time.perf_counter() - start
        print(
            f"    epoch {epoch + 1:4d}/{epochs}: loss={loss:.4f} "
            f"train(top1={train_metrics['top1']*100:.2f}%, top2={train_metrics['top2']*100:.2f}%) "
            f"test(top1={test_metrics['top1']*100:.2f}%, top2={test_metrics['top2']*100:.2f}%) "
            f"elapsed={elapsed:.2f}s"
        )
        metric_score = test_metrics["two_choice"]
        if metric_score > best_metric:
            best_metric = metric_score
            best_state = model.state_dict()
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print("    ※改善が頭打ちのため早期終了します。")
            break

    if best_state is None:
        best_state = model.state_dict()

    final_train = evaluate_from_logits(model_logits_from_state(best_state, x_train), y_train, y_second_train)
    final_test = evaluate_from_logits(model_logits_from_state(best_state, x_test), y_test, y_second_test)
    return best_state, final_train, final_test, shared


def format_metrics(metrics: Dict[str, float]) -> str:
    """指標を百分率表示用の文字列に変換する。"""

    def pct(value: float) -> str:
        if math.isnan(value):
            return "---"
        return f"{value * 100:.2f}%"

    return f"top1={pct(metrics['top1'])} / top2={pct(metrics['top2'])} / top1+second={pct(metrics['two_choice'])}"


def save_model(
    export_dir: Path,
    name: str,
    state: Dict[str, np.ndarray],
    feature_dim: int,
    class_count: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """学習済みモデルを NPZ 形式で保存する。"""

    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"{name}_model.npz"
    np.savez(
        path,
        feature_dim=np.int32(feature_dim),
        class_count=np.int32(class_count),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        max_block_pixels=np.int32(MAX_BLOCK_PIXELS),
        max_components=np.int32(MAX_COMPONENTS),
        extra_features=np.int32(EXTRA_FEATURES),
        **state,
    )


def summarize_metrics(summary: Dict[str, Dict[str, float]]) -> None:
    """各フェーズの指標を表形式で表示する。"""

    header = "phase        train_top1  train_top2  train_2nd  test_top1  test_top2  test_2nd"
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in summary.items():
        print(
            f"{name:10s}  "
            f"{metrics['train_top1']*100:9.2f}%  {metrics['train_top2']*100:9.2f}%  {metrics['train_two_choice']*100:9.2f}%  "
            f"{metrics['test_top1']*100:9.2f}%  {metrics['test_top2']*100:9.2f}%  {metrics['test_two_choice']*100:9.2f}%"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TLG8 ANN 学習補助スクリプト")
    parser.add_argument("inputs", nargs="+", help="JSONL 形式の学習データファイルまたはディレクトリ")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="評価用データの割合 (0-0.5 程度)")
    parser.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正則化係数")
    parser.add_argument("--batch-size", type=int, default=512, help="ミニバッチサイズ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--export-dir", type=Path, help="学習済みモデルを保存するディレクトリ")
    parser.add_argument(
        "--model",
        choices=["mlp", "logistic"],
        default="mlp",
        help="学習に利用するモデル種別",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[1024, 512, 256],
        help="MLP の隠れ層ユニット数 (mlp 選択時)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="MLP のドロップアウト率 (0-0.9)",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="特徴量の標準化を無効化する",
    )
    args = parser.parse_args(argv)

    files = discover_input_files(args.inputs)
    if not files:
        print("エラー: 入力ファイルが見つかりません", file=sys.stderr)
        return 1

    data = load_dataset(files)
    feature_dim = data["features"].shape[1]
    mean, std = compute_feature_scaler(data["features"], not args.no_standardize)

    stages = [
        ("predictor", K_PREDICTOR_CLASSES),
        ("filter", K_FILTER_CLASSES),
        ("reorder", K_REORDER_CLASSES),
        ("interleave", K_INTERLEAVE_CLASSES),
        ("entropy", K_ENTROPY_CLASSES),
    ]

    summary: Dict[str, Dict[str, float]] = {}

    for name, classes in stages:
        print(f"\n=== {name} ===")
        try:
            state, train_metrics, test_metrics, note = stage_training(
                name,
                data,
                args.test_ratio,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.seed,
                classes,
                args.model,
                args.hidden_dims,
                args.dropout,
                args.batch_size,
                mean,
                std,
            )
        except RuntimeError as exc:
            print(f"  学習失敗: {exc}")
            continue

        print(f"  特徴量次元: {feature_dim}, クラス数: {classes}")
        print(f"  訓練データ: {format_metrics(train_metrics)}")
        print(f"  評価データ: {format_metrics(test_metrics)}")
        if note:
            print("  ※サンプルが少ない、あるいは単一クラスのため参考値です。")

        summary[name] = {
            "train_top1": train_metrics["top1"],
            "train_top2": train_metrics["top2"],
            "train_two_choice": train_metrics["two_choice"],
            "test_top1": test_metrics["top1"],
            "test_top2": test_metrics["top2"],
            "test_two_choice": test_metrics["two_choice"],
        }

        if args.export_dir:
            save_model(args.export_dir, name, state, feature_dim, classes, mean, std)

    summarize_metrics(summary)

    if args.export_dir:
        metrics_path = Path(args.export_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        print(f"\nメトリクスを {metrics_path} に保存しました。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
