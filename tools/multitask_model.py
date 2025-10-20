"""TLG8 用マルチタスク学習モデルの実装モジュール。

本モジュールは 8x8 ブロック特徴量を入力とし、各圧縮フェーズ
（predictor / filter_perm / filter_primary / filter_secondary / reorder / interleave）の
分類を同時に行う多層パーセプトロンを提供する。学習・評価・推論および
NPZ 形式での保存 / 復元機能を一括して実装する。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, TYPE_CHECKING

import sys

import numpy as np

try:  # PyTorch が未導入環境でも numpy バックエンドを維持するための遅延インポート
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - PyTorch 非導入環境でのフォールバック
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

if TYPE_CHECKING:  # 型チェック時のみ使用
    import torch  # noqa: F401

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


def _ensure_torch() -> "torch":
    """PyTorch 利用前に導入状態を検証する。"""

    if torch is None:
        raise ImportError("PyTorch がインストールされていません。'pip install torch' を実行してください")
    return torch


def pick_device(prefer: str = "xpu") -> "torch.device":
    """利用可能なデバイスから優先候補を選択するヘルパー。"""

    torch_mod = _ensure_torch()
    prefer_norm = (prefer or "").lower()
    if prefer_norm == "xpu" and hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available():
        return torch_mod.device("xpu")
    return torch_mod.device("cpu")


def _autocast_cm(device: "torch.device", amp: str) -> object:
    """AMP 設定に応じた自動混合精度コンテキストを返す。"""

    if torch is None or amp.lower() != "bf16":
        return nullcontext()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.amp.autocast(dtype=torch.bfloat16)  # type: ignore[attr-defined]
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class TorchMultiTask(nn.Module if nn is not None else object):
    """PyTorch 実装のマルチタスク MLP。"""

    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        dropout: float = 0.1,
        disabled_heads: Sequence[str] = (),
    ) -> None:
        torch_mod = _ensure_torch()
        super().__init__()
        if feature_mean.ndim != 1:
            raise ValueError("feature_mean は 1 次元ベクトルである必要があります")
        if feature_std.ndim != 1 or feature_std.shape != feature_mean.shape:
            raise ValueError("feature_std は feature_mean と同じ形状である必要があります")
        input_dim = int(feature_mean.shape[0])
        if input_dim <= 0:
            raise ValueError("入力特徴量次元が正の値ではありません")
        self.input_dim = input_dim
        self.hidden_dim = 384
        self.dropout_rate = float(np.clip(dropout, 0.0, 0.9))
        safe_std = feature_std.astype(np.float32).copy()
        safe_std[safe_std < 1e-6] = 1.0
        mean32 = feature_mean.astype(np.float32)
        self.register_buffer("feature_mean", torch_mod.from_numpy(mean32))
        self.register_buffer("feature_std", torch_mod.from_numpy(safe_std))
        self.fc1 = nn.Linear(self.input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1536)
        self.fc3 = nn.Linear(1536, self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        disabled_set = {name for name in disabled_heads}
        unknown = disabled_set - set(HEAD_ORDER)
        if unknown:
            raise ValueError(f"無効化対象のヘッド名が不正です: {sorted(unknown)}")
        self.active_heads: Tuple[str, ...] = tuple(
            name for name in HEAD_ORDER if name not in disabled_set
        )
        heads: Dict[str, nn.Linear] = {}
        for name, classes in HEAD_SPECS.items():
            if name in self.active_heads:
                heads[name] = nn.Linear(self.hidden_dim, classes)
        self.heads = nn.ModuleDict(heads)
        self.inference_temperature: float = 1.0

    def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """標準化後にトランクとヘッドを順伝播する。"""

        torch_mod = _ensure_torch()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.dtype not in (torch_mod.float16, torch_mod.float32, torch_mod.bfloat16):
            x = x.to(dtype=torch_mod.float32)
        standardized = (x - self.feature_mean) / self.feature_std
        h = F.gelu(self.fc1(standardized))
        h = self.dropout1(h)
        h = F.gelu(self.fc2(h))
        h = self.dropout2(h)
        h = F.gelu(self.fc3(h))
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.active_heads:
            outputs[name] = self.heads[name](h)
        return outputs

    def load_from_npz(self, npz_path: str) -> None:
        """既存 NPZ 形式の重みを PyTorch モデルへ読み込む。"""

        torch_mod = _ensure_torch()
        with torch_mod.no_grad():
            with np.load(npz_path) as data:
                state = {key: data[key] for key in data.files}
            mean = state.get("feature_mean")
            std = state.get("feature_std")
            if mean is None or std is None:
                raise KeyError("NPZ に feature_mean/std が含まれていません")
            safe_std = np.asarray(std, dtype=np.float32).copy()
            safe_std[safe_std < 1e-6] = 1.0
            self.feature_mean.copy_(torch_mod.from_numpy(np.asarray(mean, dtype=np.float32)))
            self.feature_std.copy_(torch_mod.from_numpy(safe_std))
            dropout_arr = state.get("dropout")
            if dropout_arr is not None:
                self.dropout_rate = float(np.asarray(dropout_arr, dtype=np.float32)[0])
                self.dropout1.p = self.dropout_rate
                self.dropout2.p = self.dropout_rate
            temp_arr = state.get("inference_temperature")
            if temp_arr is not None:
                self.inference_temperature = float(np.asarray(temp_arr, dtype=np.float32)[0])
            for idx, layer in enumerate((self.fc1, self.fc2, self.fc3)):
                weight_key = f"trunk_W{idx}"
                bias_key = f"trunk_b{idx}"
                if weight_key not in state or bias_key not in state:
                    raise KeyError(f"NPZ に {weight_key}/{bias_key} が見つかりません")
                weight = np.asarray(state[weight_key], dtype=np.float32)
                bias = np.asarray(state[bias_key], dtype=np.float32)
                layer.weight.data.copy_(torch_mod.from_numpy(weight.T))
                layer.bias.data.copy_(torch_mod.from_numpy(bias))
            available = state.get("active_heads")
            if available is not None:
                active = [str(name) for name in np.asarray(available).tolist()]
            else:
                active = list(HEAD_ORDER)
            active_loaded: List[str] = []
            for name in active:
                if name not in self.heads:
                    continue
                weight_key = f"head_{name}_W"
                bias_key = f"head_{name}_b"
                if weight_key not in state or bias_key not in state:
                    raise KeyError(f"NPZ に {weight_key}/{bias_key} が見つかりません")
                weight = np.asarray(state[weight_key], dtype=np.float32)
                bias = np.asarray(state[bias_key], dtype=np.float32)
                head = self.heads[name]
                head.weight.data.copy_(torch_mod.from_numpy(weight.T))
                head.bias.data.copy_(torch_mod.from_numpy(bias))
                active_loaded.append(name)
            self.active_heads = tuple(active_loaded)

    def save_torch(self, path: str) -> None:
        """state_dict とスケーラ情報を保存する。"""

        _ensure_torch()
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "mean": self.feature_mean.detach().cpu().numpy(),
            "std": self.feature_std.detach().cpu().numpy(),
            "dropout": self.dropout_rate,
            "inference_temperature": self.inference_temperature,
        }
        torch.save(payload, path)  # type: ignore[union-attr]

    @staticmethod
    def load_torch(path: str, *, map_location: str | "torch.device" | None = None) -> "TorchMultiTask":
        """保存済み PyTorch モデルを復元する。"""

        torch_mod = _ensure_torch()
        payload = torch_mod.load(path, map_location=map_location)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        dropout = float(payload.get("dropout", 0.1))
        model = TorchMultiTask(mean, std, dropout=dropout)
        model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
        model.inference_temperature = float(payload.get("inference_temperature", 1.0))
        return model


def predict_logits_batched(
    model: TorchMultiTask,
    features: np.ndarray,
    device: "torch.device",
    *,
    batch_size: int = 8192,
    amp: str = "bf16",
) -> Dict[str, np.ndarray]:
    """PyTorch モデルでバッチ推論を行い NumPy 配列を返す。"""

    torch_mod = _ensure_torch()
    if batch_size <= 0:
        raise ValueError("batch_size は正の整数である必要があります")
    if isinstance(features, np.ndarray):
        arr = np.asarray(features, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        total = arr.shape[0]
        fetch = lambda start, end: arr[start:end]
    else:
        squeeze = False
        total = int(getattr(features, "shape")[0])  # type: ignore[index]
        fetch = lambda start, end: np.asarray(features[start:end], dtype=np.float32)  # type: ignore[index]
    active_heads = tuple(getattr(model, "active_heads", HEAD_ORDER))
    if total == 0:
        return {
            name: np.empty((0, HEAD_SPECS[name]), dtype=np.float32)
            if not squeeze
            else np.empty((HEAD_SPECS[name],), dtype=np.float32)
            for name in active_heads
        }
    outputs: Dict[str, np.ndarray] = {
        name: np.empty((total, HEAD_SPECS[name]), dtype=np.float32)
        for name in active_heads
    }
    was_training = model.training  # type: ignore[attr-defined]
    model.eval()
    with torch_mod.no_grad():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch_np = fetch(start, end)
            xb = torch_mod.from_numpy(batch_np).to(device=device, dtype=torch_mod.float32)
            with _autocast_cm(device, amp):
                logits = model(xb)
            for name in active_heads:
                outputs[name][start:end] = logits[name].detach().cpu().numpy()
    if was_training:
        model.train()
    if squeeze:
        return {name: outputs[name][0] for name in active_heads}
    return outputs


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


def _deterministic_topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    """(-logit, class_id) の順でソートした上位 k インデックスを返す。"""

    if logits.ndim != 2:
        raise ValueError("logits は (N, C) 形状である必要があります")
    if k <= 0:
        raise ValueError("k は正の整数である必要があります")
    n_samples, n_classes = logits.shape
    k_eff = min(k, n_classes)
    result = np.zeros((n_samples, k_eff), dtype=np.int64)
    class_ids = np.arange(n_classes, dtype=np.int64)
    for idx in range(n_samples):
        row = logits[idx]
        order = np.lexsort((class_ids, -row))
        result[idx] = order[:k_eff]
    return result


def _deterministic_topk_row(logits_row: np.ndarray, k: int) -> np.ndarray:
    """1 次元ロジット配列に対して決定的な top-k インデックスを返す。"""

    if logits_row.ndim != 1:
        raise ValueError("logits_row は 1 次元である必要があります")
    class_ids = np.arange(logits_row.shape[0], dtype=np.int64)
    order = np.lexsort((class_ids, -logits_row))
    return order[: min(k, logits_row.shape[0])]


def compute_metrics(
    logits: np.ndarray, best: np.ndarray, second: np.ndarray
) -> Dict[str, float]:
    """ロジットから top1/top2/top3/three_choice 指標を算出する。"""

    if logits.ndim != 2:
        raise ValueError("logits は (N, C) 形状である必要があります")
    sample_count = logits.shape[0]
    if sample_count == 0:
        return {
            "top1": float("nan"),
            "top2": float("nan"),
            "top3": float("nan"),
            "three_choice": float("nan"),
        }

    best = np.asarray(best, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    top1_idx = _deterministic_topk_indices(logits, 1)
    pred = top1_idx[:, 0]
    top1 = pred == best

    if logits.shape[1] >= 2:
        top2_idx = _deterministic_topk_indices(logits, 2)
        top2 = np.any(top2_idx == best[:, None], axis=1)
    else:
        top2 = top1

    top3_idx = _deterministic_topk_indices(logits, 3)
    top3 = np.any(top3_idx == best[:, None], axis=1)

    three_choice = top1.copy()
    valid_second = second >= 0
    three_choice |= valid_second & (pred == second)

    return {
        "top1": float(np.mean(top1)),
        "top2": float(np.mean(top2)),
        "top3": float(np.mean(top3)),
        "three_choice": float(np.mean(three_choice)),
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
        *,
        disabled_heads: Sequence[str] = (),
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
        self.inference_temperature: float = 1.0
        disabled_set = {name for name in disabled_heads}
        unknown = disabled_set - set(HEAD_ORDER)
        if unknown:
            raise ValueError(f"無効化対象のヘッド名が不正です: {sorted(unknown)}")
        self.active_heads: Tuple[str, ...] = tuple(
            name for name in HEAD_ORDER if name not in disabled_set
        )
        self.disabled_heads: Tuple[str, ...] = tuple(disabled_set)

    def _standardize_batch(self, features: np.ndarray) -> np.ndarray:
        """特徴量を平均・標準偏差で正規化する。"""

        arr = np.asarray(features, dtype=np.float32)
        standardized = (arr - self.feature_mean) / self.feature_std
        return standardized.astype(np.float64)

    def _prepare_standardized(self, features: np.ndarray) -> Tuple[np.ndarray, bool]:
        """推論向けに 1 サンプル入力を正規化し、元次元情報を返す。"""

        arr = np.asarray(features, dtype=np.float32)
        squeeze = False
        if arr.ndim == 1:
            arr = arr[None, :]
            squeeze = True
        standardized = self._standardize_batch(arr)
        return standardized, squeeze

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
            if name not in self.active_heads:
                continue
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
            "inference_temperature": np.asarray([self.inference_temperature], dtype=np.float32),
        }
        for idx, (W, b) in enumerate(zip(self.trunk_weights, self.trunk_biases)):
            state[f"trunk_W{idx}"] = W.astype(np.float32)
            state[f"trunk_b{idx}"] = b.astype(np.float32)
        state["active_heads"] = np.asarray(self.active_heads, dtype="U16")
        for name in self.active_heads:
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
        temp_arr = state.get("inference_temperature")
        self.inference_temperature = float(temp_arr[0]) if temp_arr is not None else 1.0
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
        active_arr = state.get("active_heads")
        if active_arr is not None:
            active_names = [str(name) for name in np.asarray(active_arr).tolist()]
        else:
            active_names = list(HEAD_ORDER)
        self.active_heads = tuple(active_names)
        self.disabled_heads = tuple(name for name in HEAD_ORDER if name not in self.active_heads)
        self.head_weights = {}
        self.head_biases = {}
        for name in self.active_heads:
            weight_key = f"head_{name}_W"
            bias_key = f"head_{name}_b"
            if weight_key not in state or bias_key not in state:
                raise KeyError(f"状態に {weight_key}/{bias_key} が見つかりません")
            self.head_weights[name] = state[weight_key].astype(np.float64)
            self.head_biases[name] = state[bias_key].astype(np.float64)

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

        standardized = self._standardize_batch(x)
        activations, pre_acts, dropout_masks = self._forward_trunk(
            standardized, training, rng
        )
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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> float:
        """1 エポック分の学習を実行する。"""

        if batch_size <= 0:
            raise ValueError("batch_size は正の整数である必要があります")
        sample_count = features.shape[0]
        indices = rng.permutation(sample_count)
        total_loss = 0.0
        total_batches = max(1, (sample_count + batch_size - 1) // batch_size)
        for batch_no, start in enumerate(range(0, sample_count, batch_size), start=1):
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
            if progress_callback is not None:
                progress_callback(batch_no, total_batches)
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
        for name in self.active_heads:
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
        for name in self.active_heads:
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

        if isinstance(features, np.ndarray):
            standardized, squeeze = self._prepare_standardized(features)
            rng = np.random.default_rng(0)
            activations, _, _ = self._forward_trunk(standardized, training=False, rng=rng)
            hidden = activations[-1]
            logits: Dict[str, np.ndarray] = {}
            for name in self.active_heads:
                out = hidden @ self.head_weights[name] + self.head_biases[name]
                if squeeze:
                    logits[name] = out[0]
                else:
                    logits[name] = out
            return logits

        sample_count = int(getattr(features, "shape")[0])  # type: ignore[index]
        if sample_count == 0:
            return {name: np.empty((0, HEAD_SPECS[name]), dtype=np.float64) for name in self.active_heads}

        rng = np.random.default_rng(0)
        batch_size = min(8192, sample_count)
        outputs: Dict[str, np.ndarray] = {
            name: np.empty((sample_count, HEAD_SPECS[name]), dtype=np.float64)
            for name in self.active_heads
        }
        position = 0
        while position < sample_count:
            end = min(sample_count, position + batch_size)
            batch_idx = np.arange(position, end, dtype=np.int64)
            batch = features[batch_idx]
            hidden, _, _, _ = self._forward_hidden(batch, rng, training=False)
            for name in self.active_heads:
                outputs[name][position:end] = hidden @ self.head_weights[name] + self.head_biases[name]
            position = end
        return outputs

    def predict_topk(
        self,
        features: np.ndarray,
        k: int = 3,
        per_head_k: Dict[str, int] | None = None,
        temperature: float | None = None,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """各ヘッドについて上位候補 (class_id, logprob) を返す。"""

        logits = self.predict_logits(features)
        base_temp = self.inference_temperature if temperature is None else float(temperature)
        temp = max(base_temp, 1e-6)
        result: Dict[str, List[Tuple[int, float]]] = {}
        for name, arr in logits.items():
            vec = np.asarray(arr)
            if vec.ndim == 1:
                data = vec
            elif vec.ndim == 2 and vec.shape[0] == 1:
                data = vec[0]
            else:
                raise ValueError("predict_topk は単一サンプルの入力にのみ対応します")
            head_k = per_head_k.get(name, k) if per_head_k else k
            head_k = max(1, min(int(head_k), data.shape[0]))
            scaled = data / temp
            log_probs = log_softmax(scaled[None, :])[0]
            order = _deterministic_topk_row(scaled, head_k)
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

    active_heads = tuple(getattr(model, "active_heads", HEAD_ORDER))
    if not active_heads:
        raise ValueError("学習対象ヘッドが存在しません")

    head_list = list(active_heads)

    train_targets: Dict[str, np.ndarray] = {}
    for name in active_heads:
        train_targets[name] = build_soft_targets(
            train_labels[name], train_second[name], HEAD_SPECS[name], config.epsilon_soft
        )

    best_state = model.copy_state()
    best_metric = -np.inf
    best_train_metrics: Dict[str, Dict[str, float]] = {}
    best_val_metrics: Dict[str, Dict[str, float]] = {}
    patience_counter = 0

    for epoch in range(config.epochs):
        total_batches = max(1, (train_features.shape[0] + config.batch_size - 1) // config.batch_size)
        digits = len(str(total_batches))
        bar_width = 30
        progress_rendered = False

        def report_progress(batch_idx: int, total: int) -> None:
            """バッチ処理の進捗を表示する。"""

            nonlocal progress_rendered
            progress_rendered = True
            ratio = min(max(batch_idx / total, 0.0), 1.0)
            filled = min(bar_width, max(0, int(round(ratio * bar_width))))
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\rエポック {epoch + 1:03d} {batch_idx:>{digits}}/{total} "
                f"[{bar}] {ratio * 100:6.2f}%"
            )
            sys.stdout.flush()

        loss = model.train_epoch(
            train_features,
            train_targets,
            config.batch_size,
            config.learning_rate,
            config.weight_decay,
            rng,
            progress_callback=report_progress,
        )
        if progress_rendered:
            sys.stdout.write("\n")
        train_logits = model.predict_logits(train_features)
        val_logits = model.predict_logits(val_features)
        train_metrics: Dict[str, Dict[str, float]] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for name in active_heads:
            train_metrics[name] = compute_metrics(
                train_logits[name], train_labels[name], train_second[name]
            )
            val_metrics[name] = compute_metrics(
                val_logits[name], val_labels[name], val_second[name]
            )
        mean_top3 = float(np.mean([val_metrics[name]["top3"] for name in head_list]))
        mean_three_choice = float(
            np.mean([val_metrics[name]["three_choice"] for name in head_list])
        )
        mean_top1 = float(np.mean([val_metrics[name]["top1"] for name in head_list]))
        mean_top2 = float(np.mean([val_metrics[name]["top2"] for name in head_list]))
        print(
            "epoch {epoch:03d}: loss={loss:.4f} val_top3={top3:.2f}% "
            "val_three={three:.2f}% val_top1={top1:.2f}% val_top2={top2:.2f}%".format(
                epoch=epoch + 1,
                loss=loss,
                top3=mean_top3 * 100.0,
                three=mean_three_choice * 100.0,
                top1=mean_top1 * 100.0,
                top2=mean_top2 * 100.0,
            )
        )
        if mean_top3 > best_metric + 1e-6:
            best_metric = mean_top3
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

    if not top_perm or not top_primary or not top_secondary:
        return []
    combos: List[Tuple[int, float]] = []
    for perm_id, perm_score in top_perm:
        perm_mod = perm_id % 6
        for primary_id, primary_score in top_primary:
            primary_mod = primary_id % 4
            for secondary_id, secondary_score in top_secondary:
                secondary_mod = secondary_id % 4
                code = (perm_mod << 4) | (primary_mod << 2) | secondary_mod
                score = perm_score + primary_score + secondary_score
                combos.append((code, score))
    combos.sort(key=lambda item: (-item[1], item[0]))
    if max_candidates <= 0:
        return combos
    return combos[: max_candidates]


def evaluate_multitask(
    model: MultiTaskModel,
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """学習済みモデルの指標を計算する。"""

    logits = model.predict_logits(features)
    metrics: Dict[str, Dict[str, float]] = {}
    for name in getattr(model, "active_heads", HEAD_ORDER):
        metrics[name] = compute_metrics(logits[name], labels[name], second[name])
    return metrics
