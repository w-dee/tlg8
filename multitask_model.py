"""TLG8 用マルチタスク学習モデルの実装モジュール。

本モジュールは 8x8 ブロック特徴量を入力とし、各圧縮フェーズ
（predictor / filter_perm / filter_primary / filter_secondary / reorder / interleave）の
分類を同時に行う多層パーセプトロンを提供する。学習・評価・推論および
NPZ 形式での保存 / 復元機能を一括して実装する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# 各ヘッドのクラス数定義
HEAD_SPECS: Dict[str, int] = {
    "predictor": 8,
    "filter_perm": 6,
    "filter_primary": 4,
    "filter_secondary": 4,
    "reorder": 8,
    "interleave": 2,
}

HEAD_ORDER: Tuple[str, ...] = (
    "predictor",
    "filter_perm",
    "filter_primary",
    "filter_secondary",
    "reorder",
    "interleave",
)


@dataclass
class TrainConfig:
    """学習時に使用する各種ハイパーパラメータをまとめた設定。"""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    epsilon_soft: float
    patience: int
    seed: int


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU 活性化関数。"""

    return np.maximum(x, 0.0)


def _relu_backward(grad: np.ndarray, pre_activation: np.ndarray) -> np.ndarray:
    """ReLU の逆伝播を計算する。"""

    mask = pre_activation > 0.0
    return grad * mask


def _logsumexp(logits: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """数値安定化付き log-sum-exp。"""

    max_vals = np.max(logits, axis=axis, keepdims=True)
    stabilized = logits - max_vals
    sum_exp = np.sum(np.exp(stabilized), axis=axis, keepdims=True)
    return max_vals + np.log(sum_exp)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """log-softmax を返す。"""

    return logits - _logsumexp(logits)


def softmax(logits: np.ndarray) -> np.ndarray:
    """softmax を返す。"""

    return np.exp(log_softmax(logits))


def build_soft_targets(
    best: np.ndarray, second: np.ndarray, class_count: int, epsilon: float
) -> np.ndarray:
    """最良候補と第 2 候補からソフトターゲット分布を構築する。"""

    epsilon = float(np.clip(epsilon, 0.0, 0.5))
    targets = np.zeros((best.shape[0], class_count), dtype=np.float64)
    for idx in range(best.shape[0]):
        b = int(best[idx])
        if b < 0 or b >= class_count:
            raise ValueError("best ラベルがクラス数の範囲外です")
        s = int(second[idx]) if idx < second.shape[0] else -1
        if s >= 0 and s < class_count and s != b:
            targets[idx, b] = 1.0 - epsilon
            targets[idx, s] = epsilon
        else:
            targets[idx, b] = 1.0
    return targets


def compute_metrics(
    logits: np.ndarray, best: np.ndarray, second: np.ndarray
) -> Dict[str, float]:
    """ロジットから top1/top2/two_choice 指標を算出する。"""

    if logits.ndim != 2:
        raise ValueError("logits は (N, C) 形状である必要があります")
    if logits.shape[0] == 0:
        return {"top1": float("nan"), "top2": float("nan"), "two_choice": float("nan")}

    pred = np.argmax(logits, axis=1)
    top1 = pred == best

    if logits.shape[1] >= 2:
        top2_idx = np.argpartition(logits, kth=-2, axis=1)[:, -2:]
        top2 = np.any(top2_idx == best[:, None], axis=1)
    else:
        top2 = top1

    two_choice = top1.copy()
    valid_second = second >= 0
    two_choice |= valid_second & (pred == second)

    return {
        "top1": float(np.mean(top1)),
        "top2": float(np.mean(top2)),
        "two_choice": float(np.mean(two_choice)),
    }


class MultiTaskModel:
    """TLG8 ブロック構成推定用のマルチタスク MLP モデル。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim は正の整数である必要があります")
        if not hidden_dims:
            raise ValueError("少なくとも 1 層の隠れ層が必要です")
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        self.feature_mean = feature_mean.astype(np.float32)
        safe_std = feature_std.astype(np.float32).copy()
        safe_std[safe_std < 1e-6] = 1.0
        self.feature_std = safe_std
        self.trunk_weights: List[np.ndarray] = []
        self.trunk_biases: List[np.ndarray] = []
        self.head_weights: Dict[str, np.ndarray] = {}
        self.head_biases: Dict[str, np.ndarray] = {}

    def initialize(self, rng: np.random.Generator) -> None:
        """重みを乱数初期化する。"""

        dims = [self.input_dim] + self.hidden_dims
        self.trunk_weights = []
        self.trunk_biases = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / in_dim)
            W = rng.normal(scale=scale, size=(in_dim, out_dim))
            b = np.zeros((out_dim,), dtype=np.float64)
            self.trunk_weights.append(W.astype(np.float64))
            self.trunk_biases.append(b)

        self.head_weights = {}
        self.head_biases = {}
        last_dim = self.hidden_dims[-1]
        for name, classes in HEAD_SPECS.items():
            scale = np.sqrt(2.0 / last_dim)
            self.head_weights[name] = rng.normal(scale=scale, size=(last_dim, classes)).astype(
                np.float64
            )
            self.head_biases[name] = np.zeros((classes,), dtype=np.float64)

    def copy_state(self) -> Dict[str, np.ndarray]:
        """現在の重み情報をコピーして返す。"""

        state: Dict[str, np.ndarray] = {
            "input_dim": np.asarray([self.input_dim], dtype=np.int32),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int32),
            "dropout": np.asarray([self.dropout], dtype=np.float32),
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_std": self.feature_std.astype(np.float32),
        }
        for idx, (W, b) in enumerate(zip(self.trunk_weights, self.trunk_biases)):
            state[f"trunk_W{idx}"] = W.astype(np.float32)
            state[f"trunk_b{idx}"] = b.astype(np.float32)
        for name in HEAD_ORDER:
            state[f"head_{name}_W"] = self.head_weights[name].astype(np.float32)
            state[f"head_{name}_b"] = self.head_biases[name].astype(np.float32)
        return state

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        """保存済み状態を現在のインスタンスへ読み込む。"""

        self.input_dim = int(state["input_dim"][0])
        self.hidden_dims = [int(v) for v in state["hidden_dims"]]
        self.dropout = float(state["dropout"][0])
        self.feature_mean = state["feature_mean"].astype(np.float32)
        safe_std = state["feature_std"].astype(np.float32)
        safe_std[safe_std < 1e-6] = 1.0
        self.feature_std = safe_std
        self.trunk_weights = []
        self.trunk_biases = []
        idx = 0
        while True:
            key_w = f"trunk_W{idx}"
            key_b = f"trunk_b{idx}"
            if key_w not in state:
                break
            self.trunk_weights.append(state[key_w].astype(np.float64))
            self.trunk_biases.append(state[key_b].astype(np.float64))
            idx += 1
        self.head_weights = {}
        self.head_biases = {}
        for name in HEAD_ORDER:
            self.head_weights[name] = state[f"head_{name}_W"].astype(np.float64)
            self.head_biases[name] = state[f"head_{name}_b"].astype(np.float64)

    def save(self, path: str) -> None:
        """モデル状態を NPZ 形式で保存する。"""

        state = self.copy_state()
        np.savez(path, **state)

    @staticmethod
    def load(path: str) -> "MultiTaskModel":
        """NPZ ファイルからモデルを復元する。"""

        with np.load(path) as data:
            state = {key: data[key] for key in data.files}
        input_dim = int(state["input_dim"][0])
        hidden_dims = [int(v) for v in state["hidden_dims"]]
        dropout = float(state["dropout"][0])
        feature_mean = state["feature_mean"].astype(np.float32)
        feature_std = state["feature_std"].astype(np.float32)
        model = MultiTaskModel(input_dim, hidden_dims, dropout, feature_mean, feature_std)
        model.load_state(state)
        return model

    def _forward_trunk(
        self, x: np.ndarray, training: bool, rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """干渉しない順伝播結果を返す内部ヘルパー。"""

        activations: List[np.ndarray] = [x.astype(np.float64)]
        pre_activations: List[np.ndarray] = []
        dropout_masks: List[np.ndarray] = []
        h = activations[0]
        for idx, (W, b) in enumerate(zip(self.trunk_weights, self.trunk_biases)):
            z = h @ W + b
            pre_activations.append(z)
            h = _relu(z)
            if training and self.dropout > 0.0:
                raw_mask = rng.random(h.shape) >= self.dropout
                scale = 1.0 / (1.0 - self.dropout)
                mask = raw_mask.astype(np.float64) * scale
            else:
                mask = np.ones_like(h)
            h = h * mask
            dropout_masks.append(mask)
            activations.append(h)
        return activations, pre_activations, dropout_masks

    def _forward_hidden(
        self, x: np.ndarray, rng: np.random.Generator, training: bool
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """隠れ層出力と中間結果を返す。"""

        activations, pre_acts, dropout_masks = self._forward_trunk(x, training, rng)
        hidden = activations[-1]
        return hidden, activations, pre_acts, dropout_masks

    def train_epoch(
        self,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        batch_size: int,
        lr: float,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> float:
        """1 エポック分の学習を実行する。"""

        if batch_size <= 0:
            raise ValueError("batch_size は正の整数である必要があります")
        sample_count = features.shape[0]
        indices = rng.permutation(sample_count)
        total_loss = 0.0
        for start in range(0, sample_count, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = features[batch_idx]
            hidden, activations, pre_acts, dropout_masks = self._forward_hidden(
                xb, rng, training=True
            )
            batch_loss, grad_hidden = self._update_heads(
                hidden, targets, batch_idx, lr, weight_decay
            )
            self._update_trunk(
                activations,
                pre_acts,
                dropout_masks,
                grad_hidden,
                lr,
                weight_decay,
            )
            total_loss += batch_loss * len(batch_idx)
        return float(total_loss / sample_count)

    def _update_heads(
        self,
        hidden: np.ndarray,
        targets: Dict[str, np.ndarray],
        batch_idx: np.ndarray,
        lr: float,
        weight_decay: float,
    ) -> Tuple[float, np.ndarray]:
        """ヘッド層の更新と損失計算を行う。"""

        grad_hidden = np.zeros_like(hidden)
        total_loss = 0.0
        grads_w: Dict[str, np.ndarray] = {}
        grads_b: Dict[str, np.ndarray] = {}
        for name in HEAD_ORDER:
            W = self.head_weights[name]
            b = self.head_biases[name]
            logits = hidden @ W + b
            log_probs = log_softmax(logits)
            probs = np.exp(log_probs)
            target = targets[name][batch_idx]
            loss = -np.sum(target * log_probs) / hidden.shape[0]
            diff = (probs - target) / hidden.shape[0]
            grad_hidden += diff @ W.T
            grads_w[name] = hidden.T @ diff + weight_decay * W
            grads_b[name] = diff.sum(axis=0)
            total_loss += float(loss)
        for name in HEAD_ORDER:
            self.head_weights[name] -= lr * grads_w[name]
            self.head_biases[name] -= lr * grads_b[name]
        return total_loss, grad_hidden

    def _update_trunk(
        self,
        activations: List[np.ndarray],
        pre_acts: List[np.ndarray],
        dropout_masks: List[np.ndarray],
        grad_hidden: np.ndarray,
        lr: float,
        weight_decay: float,
    ) -> None:
        """隠れ層の逆伝播を実施して重みを更新する。"""

        backprop = grad_hidden
        for layer in reversed(range(len(self.trunk_weights))):
            W = self.trunk_weights[layer]
            mask = dropout_masks[layer]
            grad = backprop * mask
            grad = _relu_backward(grad, pre_acts[layer])
            grad_W = activations[layer].T @ grad + weight_decay * W
            grad_b = grad.sum(axis=0)
            backprop = grad @ W.T
            self.trunk_weights[layer] -= lr * grad_W
            self.trunk_biases[layer] -= lr * grad_b

    def predict_logits(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """入力特徴量から各ヘッドのロジットを返す。"""

        feats = np.asarray(features, dtype=np.float32)
        squeeze = False
        if feats.ndim == 1:
            feats = feats[None, :]
            squeeze = True
        standardized = (feats - self.feature_mean) / self.feature_std
        rng = np.random.default_rng(0)
        activations, _, _ = self._forward_trunk(standardized, training=False, rng=rng)
        hidden = activations[-1]
        logits: Dict[str, np.ndarray] = {}
        for name in HEAD_ORDER:
            out = hidden @ self.head_weights[name] + self.head_biases[name]
            if squeeze:
                logits[name] = out[0]
            else:
                logits[name] = out
        return logits

    def predict_topk(
        self, features: np.ndarray, k: int = 2
    ) -> Dict[str, List[Tuple[int, float]]]:
        """各ヘッドについて上位 k 件の (class_id, logprob) を返す。"""

        logits = self.predict_logits(features)
        result: Dict[str, List[Tuple[int, float]]] = {}
        for name, arr in logits.items():
            vec = np.asarray(arr)
            if vec.ndim == 1:
                data = vec
            elif vec.ndim == 2 and vec.shape[0] == 1:
                data = vec[0]
            else:
                raise ValueError("predict_topk は単一サンプルの入力にのみ対応します")
            log_probs = log_softmax(data[None, :])[0]
            k_eff = min(k, data.shape[0])
            order = np.argsort(data)[::-1][:k_eff]
            result[name] = [(int(idx), float(log_probs[idx])) for idx in order]
        return result


def train_multitask_model(
    model: MultiTaskModel,
    train_features: np.ndarray,
    train_labels: Dict[str, np.ndarray],
    train_second: Dict[str, np.ndarray],
    val_features: np.ndarray,
    val_labels: Dict[str, np.ndarray],
    val_second: Dict[str, np.ndarray],
    config: TrainConfig,
) -> Tuple[MultiTaskModel, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """マルチタスクモデルを学習し、最良エポックのモデルを返す。"""

    rng = np.random.default_rng(config.seed)
    model.dropout = float(np.clip(config.dropout, 0.0, 0.9))
    model.initialize(rng)

    train_targets: Dict[str, np.ndarray] = {}
    for name in HEAD_ORDER:
        train_targets[name] = build_soft_targets(
            train_labels[name], train_second[name], HEAD_SPECS[name], config.epsilon_soft
        )

    best_state = model.copy_state()
    best_metric = -np.inf
    best_train_metrics: Dict[str, Dict[str, float]] = {}
    best_val_metrics: Dict[str, Dict[str, float]] = {}
    patience_counter = 0

    for epoch in range(config.epochs):
        loss = model.train_epoch(
            train_features,
            train_targets,
            config.batch_size,
            config.learning_rate,
            config.weight_decay,
            rng,
        )
        train_logits = model.predict_logits(train_features)
        val_logits = model.predict_logits(val_features)
        train_metrics: Dict[str, Dict[str, float]] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for name in HEAD_ORDER:
            train_metrics[name] = compute_metrics(
                train_logits[name], train_labels[name], train_second[name]
            )
            val_metrics[name] = compute_metrics(
                val_logits[name], val_labels[name], val_second[name]
            )
        mean_two_choice = float(
            np.mean([val_metrics[name]["two_choice"] for name in HEAD_ORDER])
        )
        print(
            f"epoch {epoch + 1:03d}: loss={loss:.4f} val_two_choice={mean_two_choice*100:.2f}%"
        )
        if mean_two_choice > best_metric + 1e-6:
            best_metric = mean_two_choice
            best_state = model.copy_state()
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("早期終了: 改善が見られませんでした")
                break

    model.load_state(best_state)
    return model, best_train_metrics, best_val_metrics


def build_filter_candidates(
    top_perm: List[Tuple[int, float]],
    top_primary: List[Tuple[int, float]],
    top_secondary: List[Tuple[int, float]],
    max_candidates: int = 4,
) -> List[Tuple[int, float]]:
    """フィルター候補を組み合わせた上位リストを構築する。"""

    if not top_perm:
        return []
    perm_id, perm_score = top_perm[0]
    primaries = top_primary[:2] if top_primary else []
    secondaries = top_secondary[:2] if top_secondary else []
    if not primaries or not secondaries:
        return []
    combos: List[Tuple[int, float]] = []
    for primary_id, primary_score in primaries:
        for secondary_id, secondary_score in secondaries:
            code = ((perm_id % 6) << 4) | ((primary_id % 4) << 2) | (secondary_id % 4)
            score = perm_score + primary_score + secondary_score
            combos.append((code, score))
    combos.sort(key=lambda item: item[1], reverse=True)
    return combos[:max_candidates]


def evaluate_multitask(
    model: MultiTaskModel,
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """学習済みモデルの指標を計算する。"""

    logits = model.predict_logits(features)
    metrics: Dict[str, Dict[str, float]] = {}
    for name in HEAD_ORDER:
        metrics[name] = compute_metrics(logits[name], labels[name], second[name])
    return metrics
