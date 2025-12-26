#!/usr/bin/env python3
"""
TLG8 のマルチタスク分類器（reorder/predictor/color_filter/interleave）学習スクリプト。

目的:
  - 圧縮率や top-1 の最大化ではなく、エンコード時の試行回数削減に繋がる top-K 精度を確認する
  - 推論を重くしない（シンプルな MLP、入出力を追いやすい）

入力（--data-dir）:
  - features.npy            float32 [N, D]  （推奨: np.load(..., mmap_mode="r") で参照）
  - labels_reorder.npy      int64   [N]     （reorder クラス 0..7）
  - labels_predictor.npy    int64   [N]     （predictor クラス 0..7）
  - labels_cf_perm.npy      int64   [N]     （color filter perm 0..5）
  - labels_cf_primary.npy   int64   [N]     （color filter primary 0..3）
  - labels_cf_secondary.npy int64   [N]     （color filter secondary 0..3）
  - labels_interleave.npy   int64   [N]     （interleave 0/1）
  - meta.json               特徴量仕様（任意。存在すれば保存する）
  - feature_mean.npy / feature_std.npy （任意。存在すれば z-score 正規化に使用）

出力（--out-dir）:
  - config.json
  - metrics.json
  - train_idx.npy / val_idx.npy
  - checkpoint_last.pt
  - model_best.pt （--save-best が有効で、val acc@3 が改善した時）

依存:
  - numpy, torch, tqdm

動作確認例（best 重みから fine-tune）:
  python ml/train.py \\
    --data-dir ml/runs/dataset \\
    --out-dir ml/runs/reorder_baseline_ft1 \\
    --resume-weights ml/runs/reorder_baseline/model_best.pt \\
    --epochs 100 \\
    --lr 2e-5 \\
    --amp \\
    --early-stop-patience 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

TASK_LABEL_FILES = {
    "predictor": "labels_predictor.npy",
    "cf_perm": "labels_cf_perm.npy",
    "cf_primary": "labels_cf_primary.npy",
    "cf_secondary": "labels_cf_secondary.npy",
    "reorder": "labels_reorder.npy",
    "interleave": "labels_interleave.npy",
}
TASK_CLASS_COUNTS = {
    "predictor": 8,
    "cf_perm": 6,
    "cf_primary": 4,
    "cf_secondary": 4,
    "reorder": 8,
    "interleave": 2,
}
TASK_ACC_KS = {
    "predictor": (1, 3, 5),
    "cf_perm": (1, 3),
    "cf_primary": (1,),
    "cf_secondary": (1,),
    "reorder": (1, 3, 5),
    "interleave": (1,),
}
TASK_BEST_K = {
    "predictor": 3,
    "cf_perm": 3,
    "cf_primary": 1,
    "cf_secondary": 1,
    "reorder": 3,
    "interleave": 1,
}


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class _Logger:
    def __init__(self, out_path: Path) -> None:
        self._fp = out_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fp.close()

    def log(self, msg: str) -> None:
        line = f"[{_now_ts()}] {msg}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()


def _is_rank0() -> bool:
    """
    分散実行時にログが多重出力されるのを避けるための簡易判定。
    このスクリプトは基本的に単一プロセス想定だが、環境変数や torch.distributed 初期化済み
    の場合は rank0 のみがログ/CSV を書く。
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        pass
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _fmt_float6(value: Any) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.6f}"


def _fmt_lr(value: Any) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    # lr は 0.000 になりやすいので科学表記にする（小数点以下 3 桁）。
    return f"{x:.3e}"


def _parse_tasks(arg: Optional[str]) -> list[str]:
    if arg is None:
        return ["reorder"]
    raw = [s.strip() for s in arg.split(",") if s.strip()]
    if not raw:
        return ["reorder"]

    tasks: list[str] = []
    for name in raw:
        if name == "cf":
            for t in ("cf_perm", "cf_primary", "cf_secondary"):
                if t not in tasks:
                    tasks.append(t)
            continue
        if name not in TASK_LABEL_FILES:
            raise ValueError(f"未対応のタスク名です: {name}")
        if name not in tasks:
            tasks.append(name)

    if not tasks:
        tasks = ["reorder"]
    return tasks


def _parse_loss_weights(arg: Optional[str], tasks: list[str]) -> dict[str, float]:
    weights = {t: 1.0 for t in tasks}
    if arg is None:
        return weights
    raw = [s.strip() for s in arg.split(",") if s.strip()]
    for token in raw:
        if "=" not in token:
            raise ValueError(f"--loss-weights の形式が不正です: {token} (name=value 形式)")
        name, value = token.split("=", 1)
        name = name.strip()
        value = float(value.strip())
        if name == "cf":
            for t in ("cf_perm", "cf_primary", "cf_secondary"):
                if t in weights:
                    weights[t] = value
            continue
        if name not in weights:
            raise ValueError(f"--loss-weights に未対応のタスク名があります: {name}")
        weights[name] = value
    return weights


def _primary_task(tasks: list[str]) -> str:
    if "reorder" in tasks:
        return "reorder"
    return tasks[0]


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self._path = path
        self._fieldnames = fieldnames
        self._path.parent.mkdir(parents=True, exist_ok=True)
        need_header = (not self._path.exists()) or self._path.stat().st_size == 0
        self._fp = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames, extrasaction="ignore")
        if need_header:
            self._writer.writeheader()
            self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def write_row(self, row: dict[str, Any]) -> None:
        normalized: dict[str, Any] = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(normalized)
        self._fp.flush()


def _try_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
    except Exception:
        return None
    s = out.decode("utf-8", errors="replace").strip()
    return s or None


def _save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        # RNG 復元は補助的なものなので、壊れていても学習自体は継続できるよう WARN に留める。
        print(f"WARN: RNG state の復元に失敗しました: {e}", flush=True)


@dataclass(frozen=True)
class SplitPaths:
    train_idx: Path
    val_idx: Path


def _validate_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    total: int,
    paths: SplitPaths,
) -> None:
    if train_idx.ndim != 1 or val_idx.ndim != 1:
        raise ValueError(
            "既存 split の次元が不正です: "
            f"train_idx.shape={train_idx.shape} val_idx.shape={val_idx.shape} "
            f"(files: {paths.train_idx}, {paths.val_idx})"
        )
    if not np.issubdtype(train_idx.dtype, np.integer) or not np.issubdtype(val_idx.dtype, np.integer):
        raise ValueError(
            "既存 split の dtype が不正です: "
            f"train_idx.dtype={train_idx.dtype} val_idx.dtype={val_idx.dtype} "
            f"(files: {paths.train_idx}, {paths.val_idx})"
        )
    n_train = int(train_idx.size)
    n_val = int(val_idx.size)
    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            "既存 split が空です（壊れている可能性があります）: "
            f"train={n_train} val={n_val} (files: {paths.train_idx}, {paths.val_idx})"
        )
    if n_train + n_val > int(total):
        raise ValueError(
            "既存 split がデータセットサイズを超えています: "
            f"train+val={n_train+n_val} total={total} (files: {paths.train_idx}, {paths.val_idx})"
        )

    train_min = int(np.min(train_idx))
    train_max = int(np.max(train_idx))
    val_min = int(np.min(val_idx))
    val_max = int(np.max(val_idx))
    max_idx = max(train_max, val_max)
    min_idx = min(train_min, val_min)
    if min_idx < 0 or max_idx >= int(total):
        raise ValueError(
            "既存 split が現在のデータセットサイズと不整合です: "
            f"total={total} min_idx={min_idx} max_idx={max_idx} "
            f"(files: {paths.train_idx}, {paths.val_idx})"
        )


def _prepare_split(
    out_dir: Path,
    total: int,
    val_ratio: float,
    seed: int,
    logger: _Logger,
) -> SplitPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = SplitPaths(train_idx=out_dir / "train_idx.npy", val_idx=out_dir / "val_idx.npy")
    if paths.train_idx.is_file() and paths.val_idx.is_file():
        train_idx = np.load(paths.train_idx, mmap_mode="r")
        val_idx = np.load(paths.val_idx, mmap_mode="r")
        _validate_split_indices(train_idx=train_idx, val_idx=val_idx, total=int(total), paths=paths)
        logger.log(
            "既存 split を使用: "
            f"train={paths.train_idx}({int(train_idx.size)}) "
            f"val={paths.val_idx}({int(val_idx.size)})"
        )
        return paths

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"--val-ratio は (0,1) である必要があります: {val_ratio}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(total).astype(np.int64, copy=False)
    n_val = int(round(total * val_ratio))
    n_val = max(1, min(total - 1, n_val))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    np.save(paths.train_idx, train_idx)
    np.save(paths.val_idx, val_idx)
    logger.log(f"split を作成: total={total} train={len(train_idx)} val={len(val_idx)} val_ratio={val_ratio}")
    return paths


class MultiTaskBatchDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """
    1アイテム = 1バッチ（まとめて memmap を index する）として返す Dataset。

    9.2M サンプルを通常の map-style Dataset（1件ずつ __getitem__）で回すと Python 呼び出しが支配的になるため、
    バッチ単位の取り出しで Python オーバーヘッドを削減する。
    """

    def __init__(
        self,
        x: np.ndarray,
        labels: dict[str, np.ndarray],
        batches: list[np.ndarray],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        eps: float = 1e-6,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"features.npy は 2次元である必要があります: shape={x.shape}")
        for name, y in labels.items():
            if y.ndim != 1:
                raise ValueError(f"labels_{name} は 1次元である必要があります: shape={y.shape}")
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"N が一致しません: X={x.shape} labels_{name}={y.shape}")
        self.x = x
        self.labels = labels
        self.batches = batches

        # 正規化は Dataset 内（CPU）では行わない。
        # エポック毎の前処理や DataLoader の重い処理と重なると GPU が待たされやすいので、
        # train/eval ループ側で device 転送後（GPU 側）にまとめて行う。
        self.eps = float(eps)
        self.mean = None if mean is None else mean.astype(np.float32, copy=False)
        self.std = None if std is None else std.astype(np.float32, copy=False)

        d = x.shape[1]
        if self.mean is not None and self.mean.shape != (d,):
            raise ValueError(f"feature_mean の shape が不一致です: mean={self.mean.shape} expected=({d},)")
        if self.std is not None and self.std.shape != (d,):
            raise ValueError(f"feature_std の shape が不一致です: std={self.std.shape} expected=({d},)")

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_idx = self.batches[i]
        feats_np = np.asarray(self.x[base_idx], dtype=np.float32)
        feats = torch.from_numpy(feats_np)

        labels: dict[str, torch.Tensor] = {}
        for name, arr in self.labels.items():
            labs_np = np.asarray(arr[base_idx], dtype=np.int64)
            labels[name] = torch.from_numpy(labs_np)
        return feats, labels


class MLPReorderClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, head_dims: dict[str, int]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(p=dropout)

        reorder_dim = head_dims.get("reorder", 8)
        self.fc3 = nn.Linear(hidden, reorder_dim)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden, dim) for name, dim in head_dims.items() if name != "reorder"}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(F.gelu(self.fc1(x)))
        h = self.drop(F.gelu(self.fc2(h)))
        out = {"reorder": self.fc3(h)}
        for name, head in self.heads.items():
            out[name] = head(h)
        return out


class MLPReorderClassifierTvHeads(nn.Module):
    """
    特徴量末尾の tv1(8) / tv2(8) を分離して扱う版。

    前提:
      - 入力 x: [B, D]
      - 末尾 16 次元が tv1(8) + tv2(8)（この順）
      - それ以外を base としてまとめて扱う
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dropout: float,
        head_dims: dict[str, int],
        tv_dim: int = 8,
        base_head: int = 256,
        tv_head: int = 64,
    ) -> None:
        super().__init__()
        if in_dim < tv_dim * 2:
            raise ValueError(f"in_dim が小さすぎます: in_dim={in_dim} tv_dim={tv_dim}")
        self.tv_dim = int(tv_dim)
        self.base_dim = int(in_dim - tv_dim * 2)

        self.base_fc = nn.Linear(self.base_dim, int(base_head))
        self.tv1_fc = nn.Linear(self.tv_dim, int(tv_head))
        self.tv2_fc = nn.Linear(self.tv_dim, int(tv_head))
        self.drop = nn.Dropout(p=float(dropout))

        trunk_in = int(base_head) + int(tv_head) * 2
        self.fc1 = nn.Linear(trunk_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        reorder_dim = head_dims.get("reorder", 8)
        self.fc3 = nn.Linear(hidden, reorder_dim)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden, dim) for name, dim in head_dims.items() if name != "reorder"}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        d0 = self.base_dim
        d1 = d0 + self.tv_dim

        x_base = x[:, :d0]
        x_tv1 = x[:, d0:d1]
        x_tv2 = x[:, d1 : d1 + self.tv_dim]

        h_base = self.drop(F.gelu(self.base_fc(x_base)))
        h_tv1 = self.drop(F.gelu(self.tv1_fc(x_tv1)))
        h_tv2 = self.drop(F.gelu(self.tv2_fc(x_tv2)))

        h = torch.cat([h_base, h_tv1, h_tv2], dim=1)
        h = self.drop(F.gelu(self.fc1(h)))
        h = self.drop(F.gelu(self.fc2(h)))
        out = {"reorder": self.fc3(h)}
        for name, head in self.heads.items():
            out[name] = head(h)
        return out


@torch.no_grad()
def _acc_topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    if k <= 0:
        raise ValueError(f"k は正である必要があります: {k}")
    k = min(k, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    correct = topk.eq(y.view(-1, 1)).any(dim=1)
    return int(correct.sum().item())


def _get_task_metric(stats: dict[str, Any], task: str, k: int) -> float:
    task_stats = stats.get("tasks", {}).get(task, {})
    value = task_stats.get(f"acc{k}")
    try:
        return float(value)
    except Exception:
        return float("nan")


class TaskStats:
    def __init__(self, ks: tuple[int, ...]) -> None:
        self.loss_sum = 0.0
        self.n = 0
        self.ks = ks
        self.correct = {k: 0 for k in ks}

    def update(self, loss: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        bs = int(y.shape[0])
        self.loss_sum += float(loss.item()) * bs
        self.n += bs
        for k in self.ks:
            self.correct[k] += _acc_topk(logits, y, k)

    def as_dict(self) -> dict[str, float]:
        if self.n == 0:
            out = {"loss": float("nan")}
            for k in self.ks:
                out[f"acc{k}"] = float("nan")
            return out
        out = {"loss": self.loss_sum / self.n}
        for k in self.ks:
            out[f"acc{k}"] = self.correct[k] / self.n
        return out


class MultiTaskStats:
    def __init__(self, tasks: list[str]) -> None:
        self.loss_sum = 0.0
        self.n = 0
        self.task_stats = {t: TaskStats(TASK_ACC_KS[t]) for t in tasks}

    def update(self, total_loss: torch.Tensor, per_task: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        if not per_task:
            return
        first_task = next(iter(per_task.values()))
        bs = int(first_task[2].shape[0])
        self.loss_sum += float(total_loss.item()) * bs
        self.n += bs
        for task, (loss, logits, y) in per_task.items():
            self.task_stats[task].update(loss, logits, y)

    def as_dict(self, primary_task: str) -> dict[str, Any]:
        if self.n == 0:
            base = {"loss": float("nan"), "acc1": float("nan"), "acc3": float("nan"), "acc5": float("nan")}
        else:
            base = {"loss": self.loss_sum / self.n, "acc1": float("nan"), "acc3": float("nan"), "acc5": float("nan")}

        if primary_task in self.task_stats:
            p = self.task_stats[primary_task].as_dict()
            base["acc1"] = float(p.get("acc1", float("nan")))
            base["acc3"] = float(p.get("acc3", float("nan")))
            base["acc5"] = float(p.get("acc5", float("nan")))
        base["tasks"] = {t: self.task_stats[t].as_dict() for t in self.task_stats}
        return base


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    scaler: Optional[torch.amp.GradScaler],
    norm_mean: Optional[torch.Tensor],
    norm_inv_std: Optional[torch.Tensor],
    log_every: int,
    max_batches: Optional[int],
    tasks: list[str],
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    model.train()
    stats = MultiTaskStats(tasks)
    primary = _primary_task(tasks)

    use_cuda_amp = amp and device.type == "cuda"

    pbar = tqdm(loader, desc="train", unit="step")
    for step, (x, y_dict) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break
        x = x.to(device, non_blocking=True)
        if norm_mean is not None and norm_inv_std is not None:
            x.sub_(norm_mean).mul_(norm_inv_std)
        y = {t: y_dict[t].to(device, non_blocking=True) for t in tasks}

        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp and scaler is not None:
            # autocast は forward（=モデル推論）に限定して、挙動差分を最小化する。
            with torch.amp.autocast("cuda"):
                logits = model(x)
                total_loss = 0.0
                per_task: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                for task in tasks:
                    task_logits = logits[task]
                    task_y = y[task]
                    task_loss = F.cross_entropy(task_logits, task_y)
                    total_loss = total_loss + task_loss * float(loss_weights[task])
                    per_task[task] = (task_loss, task_logits, task_y)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            total_loss = 0.0
            per_task = {}
            for task in tasks:
                task_logits = logits[task]
                task_y = y[task]
                task_loss = F.cross_entropy(task_logits, task_y)
                total_loss = total_loss + task_loss * float(loss_weights[task])
                per_task[task] = (task_loss, task_logits, task_y)
            total_loss.backward()
            optimizer.step()

        stats.update(total_loss.detach(), {k: (v[0].detach(), v[1].detach(), v[2].detach()) for k, v in per_task.items()})
        if log_every > 0 and (step % log_every == 0 or step == 1):
            cur = stats.as_dict(primary)
            pbar.set_postfix(
                loss=f"{cur['loss']:.4f}",
                acc1=f"{cur['acc1']*100:.2f}%",
                acc3=f"{cur['acc3']*100:.2f}%",
                acc5=f"{cur['acc5']*100:.2f}%",
            )
    pbar.close()
    return stats.as_dict(primary)


@torch.no_grad()
def _eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_mean: Optional[torch.Tensor],
    norm_inv_std: Optional[torch.Tensor],
    max_batches: Optional[int],
    tasks: list[str],
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    stats = MultiTaskStats(tasks)
    primary = _primary_task(tasks)

    pbar = tqdm(loader, desc="val", unit="step")
    for step, (x, y_dict) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break
        x = x.to(device, non_blocking=True)
        if norm_mean is not None and norm_inv_std is not None:
            x.sub_(norm_mean).mul_(norm_inv_std)
        y = {t: y_dict[t].to(device, non_blocking=True) for t in tasks}
        logits = model(x)
        total_loss = 0.0
        per_task: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for task in tasks:
            task_logits = logits[task]
            task_y = y[task]
            task_loss = F.cross_entropy(task_logits, task_y)
            total_loss = total_loss + task_loss * float(loss_weights[task])
            per_task[task] = (task_loss, task_logits, task_y)
        stats.update(total_loss, per_task)

        cur = stats.as_dict(primary)
        pbar.set_postfix(
            loss=f"{cur['loss']:.4f}",
            acc1=f"{cur['acc1']*100:.2f}%",
            acc3=f"{cur['acc3']*100:.2f}%",
            acc5=f"{cur['acc5']*100:.2f}%",
        )
    pbar.close()
    return stats.as_dict(primary)


class _EpochBatchOrderSampler(Sampler[int]):
    """
    Dataset の「バッチ番号」だけをエポック毎に並び替える Sampler。

    N=9.2M のような巨大データで「インデックス全体を毎エポックシャッフル」すると CPU が支配的になり、
    GPU が待つ（starvation）問題が起きやすい。ここではバッチ数（だいたい N/batch_size）だけを
    シャッフルすることで、十分なランダム性を保ちつつコストを大幅に下げる。
    """

    def __init__(self, n_batches: int, seed: int) -> None:
        if n_batches <= 0:
            raise ValueError(f"n_batches は正である必要があります: {n_batches}")
        self._n = int(n_batches)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        order = rng.permutation(self._n)
        return iter(order.tolist())

    def __len__(self) -> int:
        return self._n


def _make_loader(
    dataset: Dataset,
    num_workers: int,
    device: torch.device,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    pin = device.type == "cuda"
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
    )


def _make_batches(indices: np.ndarray, batch_size: int, drop_last: bool) -> list[np.ndarray]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size は正である必要があります: {batch_size}")
    n = int(indices.shape[0])
    n_full = n // batch_size
    batches: list[np.ndarray] = []
    end_full = n_full * batch_size
    for i in range(0, end_full, batch_size):
        batches.append(indices[i : i + batch_size])
    if not drop_last and end_full < n:
        batches.append(indices[end_full:])
    if drop_last and not batches:
        raise ValueError("train split が小さすぎて 1バッチも作れません（--batch-size を下げてください）")
    return batches


def _validate_labels_range(name: str, y: np.ndarray, n_classes: int) -> tuple[int, int]:
    # int64 [N] を全走査して min/max を確認（大きいが 9.2M 程度なので許容）
    y_min = int(np.min(y))
    y_max = int(np.max(y))
    if y_min < 0 or y_max >= n_classes:
        raise ValueError(
            f"labels_{name} の範囲が不正です: min={y_min} max={y_max} expected=[0..{n_classes-1}]"
        )
    return y_min, y_max


def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"未対応の --device: {arg}")


def _extract_model_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """
    チェックポイント形式の揺れに対応して state_dict を取り出す。

    想定:
      - {"model_state": state_dict, ...}  (本スクリプトの形式)
      - {"model": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - state_dict そのもの（{"layer.weight": tensor, ...}）
    """
    if isinstance(ckpt, dict):
        for k in ("model_state", "model", "state_dict"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v  # type: ignore[return-value]
        # state_dict そのものっぽい場合（値が全部 Tensor）
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]
    raise ValueError("チェックポイントから model state_dict を取り出せません（形式が未対応です）")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TLG8 reorder(8-class) MLP trainer")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log-csv", type=Path, default=None, help="学習過程のCSVログを追記保存（未指定なら無効）")
    p.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="学習タスク（例: reorder,predictor,cf,interleave / default: reorder）",
    )
    p.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help="タスク別 loss 重み（例: reorder=1,predictor=1,cf=1,interleave=1）",
    )

    p.add_argument("--model", choices=["baseline", "tvheads"], default="baseline")
    p.add_argument("--tvhead-base", type=int, default=256)
    p.add_argument("--tvhead-tv", type=int, default=64)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--tv2-weight-decay",
        type=float,
        default=None,
        help="tvheads 時の tv2 ブランチ(tv2_fc.weight)用 weight decay（未指定なら --weight-decay と同じ）",
    )
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", choices=["cuda", "cpu"], default=default_device)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=False)

    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path, default=None, help="checkpoint_last.pt 等から完全再開（optimizer/scaler/epoch/best/RNG）")
    resume_group.add_argument(
        "--resume-weights",
        "--resume-best",
        dest="resume_weights",
        type=Path,
        default=None,
        help="model_best.pt 等の重みのみロードして fine-tune（optimizer/scaler/epoch は新規）",
    )
    p.add_argument(
        "--reset-best-metric",
        action="store_true",
        default=False,
        help="--resume-weights 時に best 指標を引き継がず、0 から開始する",
    )
    p.add_argument("--early-stop-patience", type=int, default=0)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tv2_wd = args.weight_decay if args.tv2_weight_decay is None else float(args.tv2_weight_decay)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _Logger(out_dir / "train.log")
    csv_logger: Optional[CsvLogger] = None
    try:
        _seed_everything(int(args.seed))
        device = _resolve_device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        data_dir: Path = args.data_dir
        features_path = data_dir / "features.npy"
        if not features_path.is_file():
            raise FileNotFoundError(f"features.npy が見つかりません: {features_path}")

        tasks = _parse_tasks(args.tasks)
        loss_weights = _parse_loss_weights(args.loss_weights, tasks)
        primary_task = _primary_task(tasks)

        label_paths: dict[str, Path] = {}
        for task in tasks:
            label_path = data_dir / TASK_LABEL_FILES[task]
            if not label_path.is_file():
                raise FileNotFoundError(f"labels_{task}.npy が見つかりません: {label_path}")
            label_paths[task] = label_path

        meta = _load_json_if_exists(data_dir / "meta.json")
        mean_path = data_dir / "feature_mean.npy"
        std_path = data_dir / "feature_std.npy"
        mean = np.load(mean_path) if mean_path.is_file() else None
        std = np.load(std_path) if std_path.is_file() else None
        if (mean is None) != (std is None):
            logger.log("WARN: feature_mean / feature_std の片方しかありません（正規化は無効化します）")
            mean = None
            std = None

        logger.log(f"features: {features_path}")
        for task in tasks:
            logger.log(f"labels_{task}: {label_paths[task]}")
        logger.log(f"tasks: {','.join(tasks)}")
        logger.log(
            "loss_weights: "
            + ",".join(f"{name}={loss_weights[name]:.3f}" for name in tasks)
        )
        if mean is not None:
            logger.log("normalization: z-score (feature_mean.npy / feature_std.npy)")
        else:
            logger.log("normalization: none")

        x = np.load(features_path, mmap_mode="r")
        labels = {task: np.load(path, mmap_mode="r") for task, path in label_paths.items()}

        for task, y in labels.items():
            if int(y.shape[0]) != int(x.shape[0]):
                raise ValueError(f"N が一致しません: features={x.shape} labels_{task}={y.shape}")

        label_ranges: dict[str, tuple[int, int]] = {}
        label_hist: dict[str, dict[str, int]] = {}
        for task, y in labels.items():
            y_min, y_max = _validate_labels_range(task, y, TASK_CLASS_COUNTS[task])
            label_ranges[task] = (y_min, y_max)
            hist_np = np.bincount(np.asarray(y, dtype=np.int64), minlength=TASK_CLASS_COUNTS[task])
            label_hist[task] = {str(i): int(hist_np[i]) for i in range(TASK_CLASS_COUNTS[task])}
        n, d = int(x.shape[0]), int(x.shape[1])
        logger.log(f"dataset: N={n} D={d}")

        split_paths = _prepare_split(out_dir, total=n, val_ratio=float(args.val_ratio), seed=int(args.seed), logger=logger)
        train_idx = np.load(split_paths.train_idx)
        val_idx = np.load(split_paths.val_idx)
        logger.log(f"train blocks: {len(train_idx)}")
        logger.log(f"val blocks:   {len(val_idx)}")

        norm_mean_t: Optional[torch.Tensor] = None
        norm_inv_std_t: Optional[torch.Tensor] = None
        if mean is not None and std is not None:
            eps = np.float32(1e-6)
            inv_std = (1.0 / (std.astype(np.float32, copy=False) + eps)).astype(np.float32, copy=False)
            norm_mean_t = torch.from_numpy(mean.astype(np.float32, copy=False)).to(device)
            norm_inv_std_t = torch.from_numpy(inv_std).to(device)

        # 学習は 1バッチ=1getitem で Python オーバーヘッドを削減する。
        # train は split の順序（作成時点で乱択済み）をそのままバッチ化し、
        # エポック毎は「バッチ順序」だけをシャッフルする。
        train_batches = _make_batches(train_idx.astype(np.int64, copy=False), int(args.batch_size), drop_last=True)
        train_ds_batch = MultiTaskBatchDataset(x=x, labels=labels, batches=train_batches, mean=mean, std=std)
        train_sampler = _EpochBatchOrderSampler(n_batches=len(train_batches), seed=int(args.seed))
        train_loader = _make_loader(
            train_ds_batch,
            num_workers=int(args.num_workers),
            device=device,
            sampler=train_sampler,
        )

        # validation は順序固定でよいので、一度だけ batch 化する。
        val_batches = _make_batches(val_idx.astype(np.int64, copy=False), int(args.batch_size), drop_last=False)
        val_ds = MultiTaskBatchDataset(x=x, labels=labels, batches=val_batches, mean=mean, std=std)
        val_loader = _make_loader(
            val_ds,
            num_workers=int(args.num_workers),
            device=device,
        )

        head_dims = {task: TASK_CLASS_COUNTS[task] for task in tasks}
        if args.model == "baseline":
            model = MLPReorderClassifier(
                in_dim=d,
                hidden=int(args.hidden),
                dropout=float(args.dropout),
                head_dims=head_dims,
            ).to(device)
        elif args.model == "tvheads":
            model = MLPReorderClassifierTvHeads(
                in_dim=d,
                hidden=int(args.hidden),
                dropout=float(args.dropout),
                head_dims=head_dims,
                tv_dim=8,
                base_head=int(args.tvhead_base),
                tv_head=int(args.tvhead_tv),
            ).to(device)
        else:
            raise ValueError(f"unknown --model: {args.model}")
        if args.model == "tvheads":
            tv2_params = []
            other_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.endswith("tv2_fc.weight"):
                    tv2_params.append(p)
                else:
                    other_params.append(p)
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "weight_decay": float(args.weight_decay)},
                    {"params": tv2_params, "weight_decay": float(tv2_wd)},
                ],
                lr=float(args.lr),
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            )

        logger.log(f"optimizer: AdamW lr={float(args.lr):.8g} wd={float(args.weight_decay):.8g}")
        if args.model == "tvheads":
            logger.log(f"tv2_weight_decay: {float(tv2_wd):.8g}")
        if int(args.early_stop_patience) > 0:
            logger.log(f"early_stop_patience: {int(args.early_stop_patience)}")
        else:
            logger.log("early_stop_patience: disabled")

        scaler: Optional[torch.amp.GradScaler] = None
        if bool(args.amp) and device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        start_epoch = 1
        best_val_acc3 = -1.0
        best_metric_name = f"val_{primary_task}_acc{TASK_BEST_K[primary_task]}"

        if args.resume_weights is not None:
            resume_weights_path = Path(args.resume_weights)
            if resume_weights_path.is_dir():
                resume_weights_path = resume_weights_path / "model_best.pt"
            if not resume_weights_path.is_file():
                raise FileNotFoundError(f"--resume-weights で指定されたファイルが見つかりません: {resume_weights_path}")
            ckpt = torch.load(resume_weights_path, map_location="cpu")
            model_state = _extract_model_state_dict(ckpt)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                logger.log(f"resume_weights: missing_keys={len(missing)}")
            if unexpected:
                logger.log(f"resume_weights: unexpected_keys={len(unexpected)}")

            # fine-tune は「新規 run 扱い」なので epoch/optimizer/scaler はロードしない。
            # best 指標はデフォルトで「ロード元の best」を引き継ぐ（初回エポックでの不自然な best 更新を避けるため）。
            loaded_val_acc3: Optional[float] = None
            if isinstance(ckpt, dict):
                v = ckpt.get("val_acc3", ckpt.get("best_val_acc3", ckpt.get("best_metric")))
                if v is not None:
                    loaded_val_acc3 = float(v)
            if bool(args.reset_best_metric) or loaded_val_acc3 is None:
                best_val_acc3 = -1.0
                if loaded_val_acc3 is None:
                    logger.log(f"resume_weights: {resume_weights_path} (weights only)")
                else:
                    logger.log(
                        f"resume_weights: {resume_weights_path} (weights only) loaded_val_acc3={loaded_val_acc3:.6f} (best_metric reset)"
                    )
            else:
                best_val_acc3 = float(loaded_val_acc3)
                logger.log(
                    f"resume_weights: {resume_weights_path} (weights only) loaded_val_acc3={loaded_val_acc3:.6f} (best_metric inherited)"
                )

        elif args.resume is not None:
            ckpt = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc3 = float(ckpt.get("best_val_acc3", best_val_acc3))
            best_metric_name = str(ckpt.get("best_metric_name", best_metric_name))
            rng_state = ckpt.get("rng_state")
            if isinstance(rng_state, dict):
                _set_rng_state(rng_state)
            logger.log(
                f"resume: {args.resume} start_epoch={start_epoch} "
                f"{best_metric_name}={best_val_acc3:.6f}"
            )

        args_json: dict[str, Any] = {}
        for k, v in vars(args).items():
            if isinstance(v, Path):
                args_json[k] = str(v)
            else:
                args_json[k] = v

        repo_commit = _try_git_commit(Path(__file__).resolve().parents[1])
        config = {
            "created_at": _now_ts(),
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "device": str(device),
            "git_commit": repo_commit,
            "meta": meta,
            "N": int(n),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "feature_dim": d,
            "label_ranges": {k: {"min": int(v[0]), "max": int(v[1])} for k, v in label_ranges.items()},
            "label_hist": label_hist,
            "tasks": tasks,
            "loss_weights": {k: float(v) for k, v in loss_weights.items()},
            "label_classes": {k: int(TASK_CLASS_COUNTS[k]) for k in tasks},
            "normalization": {
                "kind": "zscore" if mean is not None else "none",
                "mean_file": "feature_mean.npy" if mean is not None else None,
                "std_file": "feature_std.npy" if std is not None else None,
            },
            "args": args_json,
        }
        _save_json(out_dir / "config.json", config)

        metrics_path = out_dir / "metrics.json"
        all_metrics: dict[str, Any] = {"epochs": []}
        if metrics_path.is_file():
            try:
                all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                logger.log("WARN: metrics.json の読み込みに失敗（新規作成します）")
                all_metrics = {"epochs": []}

        if args.log_csv is not None and _is_rank0():
            csv_logger = CsvLogger(
                Path(args.log_csv),
                fieldnames=[
                    "epoch",
                    "lr",
                    "time_sec",
                    "train_loss",
                    "train_acc1",
                    "train_acc3",
                    "train_acc5",
                    "val_loss",
                    "val_acc1",
                    "val_acc3",
                    "val_acc5",
                    "best_metric_name",
                    "best_metric",
                    "is_best",
                ],
            )
            logger.log(f"csv_log: {Path(args.log_csv)}")

        early_stop_patience = int(args.early_stop_patience)
        no_improve_epochs = 0

        for epoch in range(start_epoch, int(args.epochs) + 1):
            logger.log(f"epoch {epoch}/{int(args.epochs)}")
            epoch_t0 = time.perf_counter()
            # 毎エポックの巨大インデックスシャッフルは避け、バッチ順序だけを入れ替える。
            train_sampler.set_epoch(int(epoch))
            train_stats = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                amp=bool(args.amp),
                scaler=scaler,
                norm_mean=norm_mean_t,
                norm_inv_std=norm_inv_std_t,
                log_every=int(args.log_every),
                max_batches=args.max_train_batches,
                tasks=tasks,
                loss_weights=loss_weights,
            )
            val_stats = _eval(
                model=model,
                loader=val_loader,
                device=device,
                norm_mean=norm_mean_t,
                norm_inv_std=norm_inv_std_t,
                max_batches=args.max_val_batches,
                tasks=tasks,
                loss_weights=loss_weights,
            )
            logger.log(
                "train: "
                f"loss={train_stats['loss']:.6f} acc1={train_stats['acc1']*100:.2f}% "
                f"acc3={train_stats['acc3']*100:.2f}% acc5={train_stats['acc5']*100:.2f}%"
            )
            logger.log(
                "val:   "
                f"loss={val_stats['loss']:.6f} acc1={val_stats['acc1']*100:.2f}% "
                f"acc3={val_stats['acc3']*100:.2f}% acc5={val_stats['acc5']*100:.2f}%"
            )

            epoch_time_sec = time.perf_counter() - epoch_t0

            all_metrics["epochs"].append(
                {
                    "epoch": int(epoch),
                    "train": train_stats,
                    "val": val_stats,
                }
            )
            _save_json(metrics_path, all_metrics)

            best_k = TASK_BEST_K[primary_task]
            val_metric = _get_task_metric(val_stats, primary_task, best_k)
            improved = val_metric > best_val_acc3
            if improved:
                best_val_acc3 = val_metric
                no_improve_epochs = 0
                if bool(args.save_best):
                    best = {
                        "epoch": int(epoch),
                        "feature_dim": int(d),
                        "val_acc3": float(val_metric),
                        "best_metric_name": best_metric_name,
                        "best_metric": float(val_metric),
                        "model_state": model.state_dict(),
                        "args": args_json,
                        "meta": meta,
                    }
                    torch.save(best, out_dir / "model_best.pt")
                    logger.log(f"best updated: {best_metric_name}={best_val_acc3:.6f} -> model_best.pt")
            else:
                no_improve_epochs += 1

            if csv_logger is not None:
                lr = float(optimizer.param_groups[0].get("lr", float(args.lr)))
                csv_logger.write_row(
                    {
                        "epoch": int(epoch),
                        "lr": _fmt_lr(lr),
                        "time_sec": _fmt_float6(epoch_time_sec),
                        "train_loss": _fmt_float6(train_stats.get("loss")),
                        "train_acc1": _fmt_float6(train_stats.get("acc1")),
                        "train_acc3": _fmt_float6(train_stats.get("acc3")),
                        "train_acc5": _fmt_float6(train_stats.get("acc5")),
                        "val_loss": _fmt_float6(val_stats.get("loss")),
                        "val_acc1": _fmt_float6(val_stats.get("acc1")),
                        "val_acc3": _fmt_float6(val_stats.get("acc3")),
                        "val_acc5": _fmt_float6(val_stats.get("acc5")),
                        "best_metric_name": best_metric_name,
                        "best_metric": _fmt_float6(val_metric),
                        "is_best": 1 if improved else 0,
                    }
                )

            ckpt_last = {
                "epoch": int(epoch),
                "feature_dim": int(d),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_acc3": float(best_val_acc3),
                "best_metric_name": best_metric_name,
                "best_metric": float(best_val_acc3),
                "rng_state": _get_rng_state(),
                "args": args_json,
            }
            torch.save(ckpt_last, out_dir / "checkpoint_last.pt")

            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                logger.log(
                    "early stop triggered: "
                    f"patience={early_stop_patience} best_val_acc3={best_val_acc3:.6f}"
                )
                break

        logger.log("done")
        return 0
    finally:
        if csv_logger is not None:
            csv_logger.close()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
