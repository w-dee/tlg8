#!/usr/bin/env python3
"""
TLG8 reorder（8クラス）分類器のベースライン学習スクリプト（最小・素直な MLP）。

目的:
  - 圧縮率や top-1 の最大化ではなく、エンコード時の試行回数削減に繋がる top-K 精度を確認する
  - 推論を重くしない（シンプルな MLP、入出力を追いやすい）

入力（--data-dir）:
  - features.npy            float32 [N, D]  （推奨: np.load(..., mmap_mode="r") で参照）
  - labels_reorder.npy      int64   [N]     （reorder クラス 0..7）
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
    --resume-best ml/runs/reorder_baseline/model_best.pt \\
    --epochs 100 \\
    --lr 2e-5 \\
    --amp \\
    --early-stop-patience 20
"""

from __future__ import annotations

import argparse
import json
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


class ReorderBatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    1アイテム = 1バッチ（まとめて memmap を index する）として返す Dataset。

    9.2M サンプルを通常の map-style Dataset（1件ずつ __getitem__）で回すと Python 呼び出しが支配的になるため、
    バッチ単位の取り出しで Python オーバーヘッドを削減する。
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batches: list[np.ndarray],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        eps: float = 1e-6,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"features.npy は 2次元である必要があります: shape={x.shape}")
        if y.ndim != 1:
            raise ValueError(f"labels_reorder.npy は 1次元である必要があります: shape={y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"N が一致しません: X={x.shape} y={y.shape}")
        self.x = x
        self.y = y
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.batches[i]
        feats_np = np.asarray(self.x[base_idx], dtype=np.float32)
        feats = torch.from_numpy(feats_np)

        labs_np = np.asarray(self.y[base_idx], dtype=np.int64)
        labs = torch.from_numpy(labs_np)
        return feats, labs


class MLPReorderClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 8)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(F.gelu(self.fc2(x)))
        return self.fc3(x)


@torch.no_grad()
def _acc_topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    if k <= 0:
        raise ValueError(f"k は正である必要があります: {k}")
    k = min(k, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    correct = topk.eq(y.view(-1, 1)).any(dim=1)
    return int(correct.sum().item())


@dataclass
class PhaseStats:
    loss_sum: float = 0.0
    n: int = 0
    correct1: int = 0
    correct3: int = 0
    correct5: int = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        bs = int(y.shape[0])
        self.loss_sum += float(loss.item()) * bs
        self.n += bs
        self.correct1 += _acc_topk(logits, y, 1)
        self.correct3 += _acc_topk(logits, y, 3)
        self.correct5 += _acc_topk(logits, y, 5)

    def as_dict(self) -> dict[str, float]:
        if self.n == 0:
            return {"loss": float("nan"), "acc1": float("nan"), "acc3": float("nan"), "acc5": float("nan")}
        return {
            "loss": self.loss_sum / self.n,
            "acc1": self.correct1 / self.n,
            "acc3": self.correct3 / self.n,
            "acc5": self.correct5 / self.n,
        }


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
) -> dict[str, float]:
    model.train()
    stats = PhaseStats()

    use_cuda_amp = amp and device.type == "cuda"

    pbar = tqdm(loader, desc="train", unit="step")
    for step, (x, y) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break
        x = x.to(device, non_blocking=True)
        if norm_mean is not None and norm_inv_std is not None:
            x.sub_(norm_mean).mul_(norm_inv_std)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp and scaler is not None:
            # autocast は forward（=モデル推論）に限定して、挙動差分を最小化する。
            with torch.amp.autocast("cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        stats.update(loss.detach(), logits.detach(), y.detach())
        if log_every > 0 and (step % log_every == 0 or step == 1):
            cur = stats.as_dict()
            pbar.set_postfix(
                loss=f"{cur['loss']:.4f}",
                acc1=f"{cur['acc1']*100:.2f}%",
                acc3=f"{cur['acc3']*100:.2f}%",
                acc5=f"{cur['acc5']*100:.2f}%",
            )
    pbar.close()
    return stats.as_dict()


@torch.no_grad()
def _eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_mean: Optional[torch.Tensor],
    norm_inv_std: Optional[torch.Tensor],
    max_batches: Optional[int],
) -> dict[str, float]:
    model.eval()
    stats = PhaseStats()

    pbar = tqdm(loader, desc="val", unit="step")
    for step, (x, y) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break
        x = x.to(device, non_blocking=True)
        if norm_mean is not None and norm_inv_std is not None:
            x.sub_(norm_mean).mul_(norm_inv_std)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        stats.update(loss, logits, y)

        cur = stats.as_dict()
        pbar.set_postfix(
            loss=f"{cur['loss']:.4f}",
            acc1=f"{cur['acc1']*100:.2f}%",
            acc3=f"{cur['acc3']*100:.2f}%",
            acc5=f"{cur['acc5']*100:.2f}%",
        )
    pbar.close()
    return stats.as_dict()


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


def _validate_labels_range(y: np.ndarray) -> tuple[int, int]:
    # int64 [N] を全走査して min/max を確認（大きいが 9.2M 程度なので許容）
    y_min = int(np.min(y))
    y_max = int(np.max(y))
    if y_min < 0 or y_max > 7:
        raise ValueError(f"labels_reorder の範囲が不正です: min={y_min} max={y_max} expected=[0..7]")
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

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
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
    resume_group.add_argument("--resume", type=Path, default=None)
    resume_group.add_argument("--resume-best", type=Path, default=None)
    p.add_argument("--early-stop-patience", type=int, default=0)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _Logger(out_dir / "train.log")
    try:
        _seed_everything(int(args.seed))
        device = _resolve_device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        data_dir: Path = args.data_dir
        features_path = data_dir / "features.npy"
        labels_path = data_dir / "labels_reorder.npy"
        if not features_path.is_file():
            raise FileNotFoundError(f"features.npy が見つかりません: {features_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"labels_reorder.npy が見つかりません: {labels_path}")

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
        logger.log(f"labels:   {labels_path}")
        if mean is not None:
            logger.log("normalization: z-score (feature_mean.npy / feature_std.npy)")
        else:
            logger.log("normalization: none")

        x = np.load(features_path, mmap_mode="r")
        y = np.load(labels_path, mmap_mode="r")

        y_min, y_max = _validate_labels_range(y)
        label_hist_np = np.bincount(np.asarray(y, dtype=np.int64), minlength=8)
        label_hist = {str(i): int(label_hist_np[i]) for i in range(8)}
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
        train_ds_batch = ReorderBatchDataset(x=x, y=y, batches=train_batches, mean=mean, std=std)
        train_sampler = _EpochBatchOrderSampler(n_batches=len(train_batches), seed=int(args.seed))
        train_loader = _make_loader(
            train_ds_batch,
            num_workers=int(args.num_workers),
            device=device,
            sampler=train_sampler,
        )

        # validation は順序固定でよいので、一度だけ batch 化する。
        val_batches = _make_batches(val_idx.astype(np.int64, copy=False), int(args.batch_size), drop_last=False)
        val_ds = ReorderBatchDataset(x=x, y=y, batches=val_batches, mean=mean, std=std)
        val_loader = _make_loader(
            val_ds,
            num_workers=int(args.num_workers),
            device=device,
        )

        model = MLPReorderClassifier(in_dim=d, hidden=int(args.hidden), dropout=float(args.dropout)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        logger.log(f"optimizer: AdamW lr={float(args.lr):.8g} wd={float(args.weight_decay):.8g}")
        if int(args.early_stop_patience) > 0:
            logger.log(f"early_stop_patience: {int(args.early_stop_patience)}")
        else:
            logger.log("early_stop_patience: disabled")

        scaler: Optional[torch.amp.GradScaler] = None
        if bool(args.amp) and device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        start_epoch = 1
        best_val_acc3 = -1.0

        if args.resume_best is not None:
            resume_best_path = Path(args.resume_best)
            if resume_best_path.is_dir():
                resume_best_path = resume_best_path / "model_best.pt"
            if not resume_best_path.is_file():
                raise FileNotFoundError(f"--resume-best で指定されたファイルが見つかりません: {resume_best_path}")
            ckpt = torch.load(resume_best_path, map_location="cpu")
            model_state = _extract_model_state_dict(ckpt)
            model.load_state_dict(model_state)

            # fine-tune は「新規 run 扱い」なので epoch/optimizer/scaler はロードしない。
            # best 指標は「新規 run 扱い」でリセットする（早期停止や best 保存をこの run 内で完結させるため）。
            loaded_val_acc3: Optional[float] = None
            if isinstance(ckpt, dict):
                v = ckpt.get("val_acc3", ckpt.get("best_val_acc3"))
                if v is not None:
                    loaded_val_acc3 = float(v)
            best_val_acc3 = -1.0
            if loaded_val_acc3 is None:
                logger.log(f"resume_best: {resume_best_path} (weights only)")
            else:
                logger.log(f"resume_best: {resume_best_path} (weights only) loaded_val_acc3={loaded_val_acc3:.6f}")

        elif args.resume is not None:
            ckpt = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc3 = float(ckpt.get("best_val_acc3", best_val_acc3))
            logger.log(f"resume: {args.resume} start_epoch={start_epoch} best_val_acc3={best_val_acc3:.6f}")

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
            "label_min": int(y_min),
            "label_max": int(y_max),
            "label_hist": label_hist,
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

        early_stop_patience = int(args.early_stop_patience)
        no_improve_epochs = 0

        for epoch in range(start_epoch, int(args.epochs) + 1):
            logger.log(f"epoch {epoch}/{int(args.epochs)}")
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
            )
            val_stats = _eval(
                model=model,
                loader=val_loader,
                device=device,
                norm_mean=norm_mean_t,
                norm_inv_std=norm_inv_std_t,
                max_batches=args.max_val_batches,
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

            all_metrics["epochs"].append(
                {
                    "epoch": int(epoch),
                    "train": train_stats,
                    "val": val_stats,
                }
            )
            _save_json(metrics_path, all_metrics)

            val_acc3 = float(val_stats["acc3"])
            improved = val_acc3 > best_val_acc3
            if improved:
                best_val_acc3 = val_acc3
                no_improve_epochs = 0
                if bool(args.save_best):
                    best = {
                        "epoch": int(epoch),
                        "feature_dim": int(d),
                        "val_acc3": float(val_acc3),
                        "model_state": model.state_dict(),
                        "args": args_json,
                        "meta": meta,
                    }
                    torch.save(best, out_dir / "model_best.pt")
                    logger.log(f"best updated: val_acc3={best_val_acc3:.6f} -> model_best.pt")
            else:
                no_improve_epochs += 1

            ckpt_last = {
                "epoch": int(epoch),
                "feature_dim": int(d),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_acc3": float(best_val_acc3),
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
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
