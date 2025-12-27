#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover
    raise SystemExit("tqdm is required: pip install tqdm") from exc

from dataset_builder import DatasetCache, build_dataset
from tlg_ml_utils import (
    HEADS,
    RAW_RGB_DIMS,
    apply_transform,
    ensure_dir,
    now_iso,
    save_json,
)

HEAD_ORDER = ["predictor", "cf_perm", "cf_primary", "cf_secondary", "reorder", "interleave"]
BEAM_WIDTHS = {
    "predictor": 4,
    "cf_perm": 4,
    "cf_primary": 3,
    "cf_secondary": 2,
    "reorder": 4,
    "interleave": 2,
}

PROJ_OFFSETS: dict[str, int] = {}
_proj_off = 0
for _head in HEAD_ORDER:
    PROJ_OFFSETS[_head] = _proj_off
    _proj_off += int(HEADS[_head])


@dataclass
class FeatureSpec:
    name: str
    raw_indices: list[int]
    transforms: list[dict[str, Any]]
    include_raw: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TLG8 ML ranker heads with trial logging")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="Directory with training.all.jsonl/labels.all.bin (defaults depend on --smoke)",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under ml/runs (auto if omitted)")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--max-epochs", type=int, default=40, help="Max epochs per head")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--neg-k", type=int, default=32, help="Negatives per sample for tuple ranking loss")
    parser.add_argument(
        "--loss",
        type=str,
        default="tuple_nce",
        choices=["tuple_nce", "tuple_rank", "tuple_nce_proj", "per_head_ce", "per_head_proj"],
        help="Training loss (default: tuple_nce)",
    )
    parser.add_argument(
        "--proj-alpha",
        type=float,
        default=0.1,
        help="Weight for projection supervision when --loss=tuple_nce_proj (default: 0.1)",
    )
    parser.add_argument(
        "--proj-temp",
        type=float,
        default=1.0,
        help="Temperature for projection softmax when --loss=tuple_nce_proj (default: 1.0)",
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=10.0,
        help="Temperature for bits-weighted soft targets (smaller => prefer best more)",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Max trials to run")
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Cap eligible blocks for training via reservoir sampling (deterministic by --seed)",
    )
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild dataset cache")
    parser.add_argument("--smoke", action="store_true", help="Run a small, fast smoke trial")
    parser.add_argument(
        "--eval-topn",
        type=str,
        default="1,3,10",
        help="Comma-separated top-N values for valid hit@N reporting (default: 1,3,10)",
    )
    return parser.parse_args()


def resolve_in_dir(path: Path | None, *, smoke: bool) -> Path:
    if path is not None:
        return path
    if smoke:
        fallback = Path("ml/runs/packed_smoke_entropy0_filterbits_c")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0_filter_top2")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0_filter_top4")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0_scores")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0_ctx_proj")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0_ctx")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke_entropy0")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
        fallback = Path("ml/runs/packed_smoke")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
    packed_out = Path("ml/source/packed_out_entropy0_ctx")
    if (packed_out / "training.all.jsonl").is_file() and (packed_out / "labels.all.bin").is_file():
        return packed_out
    packed_out = Path("ml/source/packed_out_entropy0_filterbits2")
    if (packed_out / "training.all.jsonl").is_file() and (packed_out / "labels.all.bin").is_file():
        return packed_out
    packed_out = Path("ml/source/packed_out_entropy0_filterbits")
    if (packed_out / "training.all.jsonl").is_file() and (packed_out / "labels.all.bin").is_file():
        return packed_out
    packed_out = Path("ml/source/packed_out_entropy0")
    if (packed_out / "training.all.jsonl").is_file() and (packed_out / "labels.all.bin").is_file():
        return packed_out
    packed_out = Path("ml/source/packed_out")
    if (packed_out / "training.all.jsonl").is_file() and (packed_out / "labels.all.bin").is_file():
        return packed_out
    fallback = Path("ml/runs/packed_smoke")
    if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
        return fallback
    raise SystemExit("Missing --in-dir and no suitable default dataset found")


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ML tasks, but torch.cuda.is_available() is False")
    return torch.device("cuda")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def apply_feature_pipeline(raw_numeric: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    if raw_numeric.shape[1] == 0 or not spec.raw_indices:
        raw_sel = np.empty((raw_numeric.shape[0], 0), dtype=np.float32)
    else:
        raw_sel = raw_numeric[:, spec.raw_indices]
    features: list[np.ndarray] = []
    if spec.include_raw:
        features.append(raw_sel)
    for transform in spec.transforms:
        kind = transform["kind"]
        if transform.get("apply_to") == "subset":
            cols = transform["columns"]
            base = raw_sel[:, cols]
        else:
            base = raw_sel
        derived = apply_transform(
            base,
            kind,
            clip_min=transform.get("clip_min"),
            clip_max=transform.get("clip_max"),
        )
        features.append(derived.astype(np.float32))
    if features:
        return np.concatenate(features, axis=1)
    return np.empty((raw_numeric.shape[0], 0), dtype=np.float32)


def build_feature_specs(dataset: DatasetCache) -> list[FeatureSpec]:
    raw_numeric = dataset.raw_numeric
    raw_dim = raw_numeric.shape[1]
    if raw_dim == 0:
        return [FeatureSpec(name="pixels_only", raw_indices=[], transforms=[])]

    # Handcrafted, budget-aligned filter feature set (<=256 dims) to stabilize search.
    # Prefer these over variance-only selection when present.
    def _find_prefix(prefix: str) -> list[int]:
        return [i for i, name in enumerate(dataset.raw_names) if name.startswith(prefix)]

    def _find_exact(name: str) -> list[int]:
        out: list[int] = []
        for i, n in enumerate(dataset.raw_names):
            if n == name:
                out.append(i)
        return out

    bits_filter_none = _find_prefix("score_bits_plain_hilbert_none_min_by_filter[")  # 96
    bits_filter_interleave = _find_prefix("score_bits_plain_hilbert_interleave_min_by_filter[")  # 96
    bits_filter_none_top2pred = _find_prefix("score_bits_plain_hilbert_none_min_by_filter_top2pred[")  # 96
    bits_filter_interleave_top2pred = _find_prefix("score_bits_plain_hilbert_interleave_min_by_filter_top2pred[")  # 96
    bits_perm_none = _find_prefix("score_bits_plain_hilbert_none_min_by_perm[")  # 6 (if present)
    bits_primary_none = _find_prefix("score_bits_plain_hilbert_none_min_by_primary[")  # 4 (if present)
    bits_secondary_none = _find_prefix("score_bits_plain_hilbert_none_min_by_secondary[")  # 4 (if present)
    bits_perm_none_top2pred = _find_prefix("score_bits_plain_hilbert_none_min_by_perm_top2pred[")  # 6 (if present)
    bits_primary_none_top2pred = _find_prefix("score_bits_plain_hilbert_none_min_by_primary_top2pred[")  # 4 (if present)
    bits_secondary_none_top2pred = _find_prefix("score_bits_plain_hilbert_none_min_by_secondary_top2pred[")  # 4 (if present)
    bits_perm_interleave = _find_prefix("score_bits_plain_hilbert_interleave_min_by_perm[")  # 6 (if present)
    bits_primary_interleave = _find_prefix("score_bits_plain_hilbert_interleave_min_by_primary[")  # 4 (if present)
    bits_secondary_interleave = _find_prefix("score_bits_plain_hilbert_interleave_min_by_secondary[")  # 4 (if present)
    bits_perm_interleave_top2pred = _find_prefix("score_bits_plain_hilbert_interleave_min_by_perm_top2pred[")  # 6
    bits_primary_interleave_top2pred = _find_prefix("score_bits_plain_hilbert_interleave_min_by_primary_top2pred[")  # 4
    bits_secondary_interleave_top2pred = _find_prefix("score_bits_plain_hilbert_interleave_min_by_secondary_top2pred[")  # 4
    energy_filter = _find_prefix("score_filtered_energy_min_by_filter[")  # 96
    energy_perm = _find_prefix("score_filtered_energy_min_by_perm[")  # 6
    energy_primary = _find_prefix("score_filtered_energy_min_by_primary[")  # 4
    energy_secondary = _find_prefix("score_filtered_energy_min_by_secondary[")  # 4
    pred_energy = _find_prefix("score_residual_energy_by_predictor[")  # 8
    reorder_tv = _find_prefix("reorder_tv[")  # 8
    reorder_tv_mean = _find_prefix("reorder_tv_mean[")  # 8
    reorder_tv2_mean = _find_prefix("reorder_tv2_mean[")  # 8
    pixel_derived = (
        _find_prefix("luma_pool4x4[")
        + _find_prefix("luma_dct4x4[")
        + _find_exact("luma_mean")
        + _find_exact("luma_var")
        + _find_exact("mean_r")
        + _find_exact("mean_g")
        + _find_exact("mean_b")
        + _find_exact("var_r")
        + _find_exact("var_g")
        + _find_exact("var_b")
        + _find_exact("grad_abs_dx_mean")
        + _find_exact("grad_abs_dy_mean")
        + _find_exact("grad_abs_mean")
        + _find_exact("edge_density")
    )
    best_energy = _find_exact("score_best_filtered_energy") + _find_exact("score_best_residual_energy")

    # Keep within the non-pixel cap.
    filter_budget_indices: list[int] = []
    # Prioritize filter-bits signals (both none + interleave). If top2pred-conditioned variants exist,
    # prefer those to provide cheap predictor↔filter interaction signal.
    if len(bits_filter_none_top2pred) == 96 and len(bits_filter_interleave_top2pred) == 96:
        filter_budget_indices += bits_filter_none_top2pred + bits_filter_interleave_top2pred
        filter_budget_indices += (
            bits_perm_none_top2pred
            + bits_primary_none_top2pred
            + bits_secondary_none_top2pred
            + bits_perm_interleave_top2pred
            + bits_primary_interleave_top2pred
            + bits_secondary_interleave_top2pred
        )
    else:
        filter_budget_indices += bits_filter_none + bits_filter_interleave
        filter_budget_indices += (
            bits_perm_none
            + bits_primary_none
            + bits_secondary_none
            + bits_perm_interleave
            + bits_primary_interleave
            + bits_secondary_interleave
        )
    # Add lightweight energy predictors if space remains.
    filter_budget_indices += energy_filter + pred_energy
    filter_budget_indices = filter_budget_indices[:256]

    # Filter + reorder signals (still within <=256 non-pixel dims).
    filter_reorder_indices: list[int] = []
    filter_reorder_indices += bits_filter_none + bits_filter_interleave
    filter_reorder_indices += energy_filter + pred_energy
    filter_reorder_indices += reorder_tv + reorder_tv_mean + reorder_tv2_mean
    filter_reorder_indices += best_energy
    filter_reorder_indices = filter_reorder_indices[:256]

    # Filter bits + reorder + pixel-derived summary (no full 96-dim filtered energy).
    filter_bits_reorder_pixel_indices: list[int] = []
    filter_bits_reorder_pixel_indices += bits_filter_none + bits_filter_interleave
    filter_bits_reorder_pixel_indices += (
        bits_perm_none + bits_primary_none + bits_secondary_none + bits_perm_interleave + bits_primary_interleave + bits_secondary_interleave
    )
    filter_bits_reorder_pixel_indices += pred_energy + best_energy
    filter_bits_reorder_pixel_indices += reorder_tv + reorder_tv_mean + reorder_tv2_mean
    filter_bits_reorder_pixel_indices += pixel_derived
    filter_bits_reorder_pixel_indices = filter_bits_reorder_pixel_indices[:256]

    train_raw = raw_numeric[dataset.train_indices]
    variances = np.var(train_raw, axis=0)
    order = np.argsort(-variances)

    specs: list[FeatureSpec] = []
    if len(filter_budget_indices) >= 32:
        specs.append(
            FeatureSpec(
                name="filter_budget_scores_raw",
                raw_indices=filter_budget_indices,
                transforms=[],
                include_raw=True,
            )
        )
        specs.append(
            FeatureSpec(
                name="filter_budget_scores_log1p",
                raw_indices=filter_budget_indices,
                transforms=[{"kind": "log1p", "apply_to": "all", "clip_max": 1_000_000.0}],
                include_raw=False,
            )
        )
    if len(filter_reorder_indices) >= 32:
        specs.append(
            FeatureSpec(
                name="filter_reorder_budget_log1p",
                raw_indices=filter_reorder_indices,
                transforms=[{"kind": "log1p", "apply_to": "all", "clip_max": 1_000_000.0}],
                include_raw=False,
            )
        )
    if len(filter_bits_reorder_pixel_indices) >= 32:
        specs.append(
            FeatureSpec(
                name="filter_bits_reorder_pixel_log1p",
                raw_indices=filter_bits_reorder_pixel_indices,
                transforms=[{"kind": "log1p", "apply_to": "all", "clip_max": 1_000_000.0}],
                include_raw=False,
            )
        )
    for k in (16, 32, 64, 128, 192, 256):
        if k > raw_dim:
            continue
        idx = order[:k].tolist()
        specs.append(FeatureSpec(name=f"raw_top{k}", raw_indices=idx, transforms=[]))
        if k * 2 <= 256:
            specs.append(
                FeatureSpec(
                    name=f"raw_top{k}_log1p",
                    raw_indices=idx,
                    transforms=[{"kind": "log1p", "apply_to": "all"}],
                )
            )
        if k * 2 <= 256:
            specs.append(
                FeatureSpec(
                    name=f"raw_top{k}_sqrt",
                    raw_indices=idx,
                    transforms=[{"kind": "sqrt", "apply_to": "all"}],
                )
            )
        if k * 2 <= 256:
            specs.append(
                FeatureSpec(
                    name=f"raw_top{k}_square",
                    raw_indices=idx,
                    transforms=[{"kind": "square", "apply_to": "all"}],
                )
            )
        if k * 2 <= 256:
            specs.append(
                FeatureSpec(
                    name=f"raw_top{k}_abs",
                    raw_indices=idx,
                    transforms=[{"kind": "abs", "apply_to": "all"}],
                )
            )
    specs.append(FeatureSpec(name="raw_top0", raw_indices=[], transforms=[]))
    return specs


def build_arch_specs() -> list[dict[str, list[int]]]:
    candidates = [
        [512, 256],
        [768, 256],
        [1024, 512],
        [256, 128],
        [128, 64],
        [64, 32],
        [1024],
        [768],
        [512],
        [256],
        [128],
        [96],
        [64],
        [32],
        [],
    ]
    specs: list[dict[str, list[int]]] = []
    for hidden in candidates:
        specs.append({head: list(hidden) for head in HEAD_ORDER})
    return specs


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

def top2_soft_ce_loss(logits: torch.Tensor, top2: torch.Tensor) -> torch.Tensor:
    # Soft-target CE with exactly two positive classes per sample: 0.5/0.5 (or 1.0 if the same).
    if top2.ndim != 2 or top2.shape[1] != 2:
        raise ValueError("top2 must have shape [N,2]")
    top2 = top2.to(torch.int64)
    log_probs = torch.log_softmax(logits, dim=1)
    a = top2[:, 0]
    b = top2[:, 1]
    log_a = log_probs.gather(1, a[:, None]).squeeze(1)
    log_b = log_probs.gather(1, b[:, None]).squeeze(1)
    same = a == b
    loss = torch.where(same, -log_a, -0.5 * (log_a + log_b))
    return loss.mean()


def build_soft_targets(
    labels_best: np.ndarray,
    labels_second: np.ndarray,
    bits_best: np.ndarray,
    bits_second: np.ndarray,
    *,
    target_temp: float,
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    n = int(labels_best.shape[0])
    if labels_second.shape[0] != n:
        raise ValueError("labels_best/labels_second length mismatch")
    for head_idx, head in enumerate(HEAD_ORDER):
        classes = HEADS[head]
        target = np.zeros((n, classes), dtype=np.float32)
        best_vals = labels_best[:, head_idx].astype(np.int64, copy=False)
        second_vals = labels_second[:, head_idx].astype(np.int64, copy=False)
        # Soft targets:
        # - Always include best + second head values.
        # - Weight them by bits margin so the model prefers the true best tuple when they differ.
        for i in range(n):
            b = int(best_vals[i])
            s = int(second_vals[i])
            if b == s:
                target[i, b] = 1.0
                continue
            b_bits = float(bits_best[i])
            s_bits = float(bits_second[i])
            delta = max(0.0, s_bits - b_bits)
            # Temperature is tuned to bits scale (a few bits difference should matter).
            temp = float(target_temp)
            w_best = 1.0
            w_second = float(np.exp(-delta / temp))
            z = w_best + w_second
            target[i, b] = float(w_best / z)
            target[i, s] = float(w_second / z)
        targets[head] = target
    return targets


def standardize_features(
    x_train: np.ndarray, x_valid: np.ndarray, *, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x_train.shape[1] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return x_train.astype(np.float32, copy=False), x_valid.astype(np.float32, copy=False), empty, empty
    mean = x_train.mean(axis=0, dtype=np.float64)
    var = x_train.var(axis=0, dtype=np.float64)
    std = np.sqrt(var + eps)
    x_train_z = ((x_train - mean) / std).astype(np.float32, copy=False)
    x_valid_z = ((x_valid - mean) / std).astype(np.float32, copy=False)
    return x_train_z, x_valid_z, mean.astype(np.float32), std.astype(np.float32)


def tuple_score_from_logits(logits: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.int64)
    score = torch.zeros((labels.shape[0],), device=labels.device, dtype=logits["predictor"].dtype)
    for head_idx, head in enumerate(HEAD_ORDER):
        ids = labels[:, head_idx]
        score = score + logits[head].gather(1, ids[:, None]).squeeze(1)
    return score


def sample_negative_labels(
    device: torch.device, batch_size: int, neg_k: int, *, generator: torch.Generator | None = None
) -> dict[str, torch.Tensor]:
    if neg_k <= 0:
        raise ValueError("--neg-k must be positive")
    neg: dict[str, torch.Tensor] = {}
    for head in HEAD_ORDER:
        classes = int(HEADS[head])
        neg[head] = torch.randint(0, classes, (batch_size, neg_k), device=device, generator=generator, dtype=torch.int64)
    return neg


def tuple_scores_for_negatives(logits: dict[str, torch.Tensor], neg: dict[str, torch.Tensor]) -> torch.Tensor:
    score = torch.zeros_like(neg["predictor"], dtype=logits["predictor"].dtype)
    for head in HEAD_ORDER:
        score = score + logits[head].gather(1, neg[head])
    return score

def sample_negative_tuples_from_batch(
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    *,
    neg_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Returns shape [B, neg_k, 6], sampled from {best, second} of this batch (more realistic than iid class sampling).
    if neg_k <= 0:
        raise ValueError("--neg-k must be positive")
    labels_best = labels_best.to(torch.int64)
    labels_second = labels_second.to(torch.int64)
    bsz = int(labels_best.shape[0])
    if bsz == 0:
        return torch.empty((0, neg_k, labels_best.shape[1]), device=labels_best.device, dtype=torch.int64)
    if bsz < 2:
        # No meaningful in-batch negatives possible; fall back to iid class sampling.
        parts: list[torch.Tensor] = []
        for head in HEAD_ORDER:
            classes = int(HEADS[head])
            parts.append(
                torch.randint(0, classes, (bsz, neg_k), device=labels_best.device, generator=generator, dtype=torch.int64)
            )
        return torch.stack(parts, dim=2)
    pool = torch.cat([labels_best, labels_second], dim=0)  # [2B,6]
    pool_size = int(pool.shape[0])
    idx = torch.randint(0, pool_size, (bsz, neg_k), device=labels_best.device, generator=generator, dtype=torch.int64)
    # Avoid sampling the current sample's own positives as negatives.
    base = torch.arange(bsz, device=labels_best.device, dtype=torch.int64)[:, None]
    forbidden0 = base
    forbidden1 = base + bsz
    mask = (idx == forbidden0) | (idx == forbidden1)
    idx = torch.where(mask, (idx + 1) % pool_size, idx)
    return pool[idx]


def tuple_scores_for_neg_tuples(logits: dict[str, torch.Tensor], neg_labels: torch.Tensor) -> torch.Tensor:
    # neg_labels: [B, K, 6]
    neg_labels = neg_labels.to(torch.int64)
    score = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]), device=neg_labels.device, dtype=logits["predictor"].dtype)
    for head_idx, head in enumerate(HEAD_ORDER):
        ids = neg_labels[:, :, head_idx]
        score = score + logits[head].gather(1, ids)
    return score


def tuple_ranking_loss(
    logits: dict[str, torch.Tensor],
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    *,
    neg_k: int,
    neg_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz = int(labels_best.shape[0])
    s_best = tuple_score_from_logits(logits, labels_best)
    s_second = tuple_score_from_logits(logits, labels_second)
    if neg_labels is None:
        neg_labels = sample_negative_tuples_from_batch(labels_best, labels_second, neg_k=neg_k)
    s_neg = tuple_scores_for_neg_tuples(logits, neg_labels)
    loss_best_neg = F.softplus(-(s_best[:, None] - s_neg)).mean()
    loss_second_neg = F.softplus(-(s_second[:, None] - s_neg)).mean()
    loss_best_second = F.softplus(-(s_best - s_second)).mean()
    return loss_best_neg + 0.5 * loss_second_neg + 0.5 * loss_best_second


def tuple_nce_loss(
    logits: dict[str, torch.Tensor],
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    *,
    neg_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    labels_best = labels_best.to(torch.int64)
    labels_second = labels_second.to(torch.int64)
    if neg_k <= 0:
        raise ValueError("--neg-k must be positive")

    log_probs = {head: torch.log_softmax(head_logits, dim=1) for head, head_logits in logits.items()}
    s_best = tuple_score_from_logits(log_probs, labels_best)
    s_second = tuple_score_from_logits(log_probs, labels_second)
    neg_labels = sample_negative_tuples_from_batch(labels_best, labels_second, neg_k=neg_k, generator=generator)
    s_neg = tuple_scores_for_neg_tuples(log_probs, neg_labels)

    scores = torch.cat([s_best[:, None], s_second[:, None], s_neg], dim=1)
    logz = torch.logsumexp(scores, dim=1, keepdim=True)
    logp = scores - logz
    return -(0.5 * logp[:, 0] + 0.5 * logp[:, 1]).mean()


def train_all_heads(
    device: torch.device,
    models: dict[str, nn.Module],
    x_train: torch.Tensor,
    labels_best_train: torch.Tensor,
    labels_second_train: torch.Tensor,
    x_valid: torch.Tensor,
    labels_best_valid: torch.Tensor,
    labels_second_valid: torch.Tensor,
    *,
    proj_scores_train: torch.Tensor | None = None,
    proj_scores_valid: torch.Tensor | None = None,
    proj_alpha: float = 0.0,
    proj_temp: float = 1.0,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    neg_k: int,
    loss_kind: str,
    eval_topn: list[int],
    trial_id: int,
) -> tuple[float, int, dict[int, float]]:
    params: list[torch.Tensor] = []
    for m in models.values():
        params.extend(list(m.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    best_hit_rate = -1.0
    best_hit_epoch = 0
    best_hit_rates_at: dict[int, float] = {n: 0.0 for n in eval_topn}
    best_hit_state: dict[str, dict[str, torch.Tensor]] | None = None

    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    train_slices = _batch_slices(n_train, batch_size)
    valid_slices = _batch_slices(n_valid, batch_size)

    pbar = tqdm(total=max_epochs, desc=f"trial {trial_id:04d} joint", leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        for m in models.values():
            m.train()
        order = torch.randperm(n_train, device=device, dtype=torch.int64)
        gen = torch.Generator(device=device)
        gen.manual_seed(0xA11CE + int(trial_id) * 1000 + int(epoch))
        for s, e in train_slices:
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            best_b = labels_best_train.index_select(0, idx)
            second_b = labels_second_train.index_select(0, idx)
            optimizer.zero_grad()
            logits = {name: model(xb) for name, model in models.items()}
            if loss_kind in ("tuple_nce", "tuple_nce_proj"):
                loss = tuple_nce_loss(logits, best_b, second_b, neg_k=neg_k, generator=gen)
            else:
                neg_labels = sample_negative_tuples_from_batch(best_b, second_b, neg_k=neg_k, generator=gen)
                loss = tuple_ranking_loss(logits, best_b, second_b, neg_k=neg_k, neg_labels=neg_labels)

            if loss_kind == "tuple_nce_proj":
                if proj_scores_train is None:
                    raise RuntimeError("tuple_nce_proj requires proj_scores_train")
                proj_b = proj_scores_train.index_select(0, idx)
                if proj_alpha <= 0.0:
                    raise RuntimeError("proj_alpha must be > 0 for tuple_nce_proj")
                if proj_temp <= 0.0:
                    raise RuntimeError("proj_temp must be > 0 for tuple_nce_proj")
                proj_loss = 0.0
                for head in HEAD_ORDER:
                    classes = int(HEADS[head])
                    off = int(PROJ_OFFSETS[head])
                    t_scores = proj_b[:, off : off + classes]
                    t_probs = torch.softmax(t_scores / float(proj_temp), dim=1)
                    logp = torch.log_softmax(logits[head], dim=1)
                    proj_loss = proj_loss + (-(t_probs * logp).sum(dim=1).mean())
                loss = loss + float(proj_alpha) * proj_loss
            loss.backward()
            optimizer.step()

        for m in models.values():
            m.eval()
        gen_v = torch.Generator(device=device)
        gen_v.manual_seed(0xBEE7 + int(trial_id) * 1000 + int(epoch))
        valid_losses: list[float] = []
        with torch.no_grad():
            for s, e in valid_slices:
                xb = x_valid[s:e]
                best_b = labels_best_valid[s:e]
                second_b = labels_second_valid[s:e]
                logits = {name: model(xb) for name, model in models.items()}
                if loss_kind in ("tuple_nce", "tuple_nce_proj"):
                    loss = tuple_nce_loss(logits, best_b, second_b, neg_k=neg_k, generator=gen_v)
                else:
                    neg_labels = sample_negative_tuples_from_batch(best_b, second_b, neg_k=neg_k, generator=gen_v)
                    loss = tuple_ranking_loss(logits, best_b, second_b, neg_k=neg_k, neg_labels=neg_labels)
                if loss_kind == "tuple_nce_proj":
                    if proj_scores_valid is None:
                        raise RuntimeError("tuple_nce_proj requires proj_scores_valid")
                    proj_b = proj_scores_valid[s:e]
                    proj_loss = 0.0
                    for head in HEAD_ORDER:
                        classes = int(HEADS[head])
                        off = int(PROJ_OFFSETS[head])
                        t_scores = proj_b[:, off : off + classes]
                        t_probs = torch.softmax(t_scores / float(proj_temp), dim=1)
                        logp = torch.log_softmax(logits[head], dim=1)
                        proj_loss = proj_loss + (-(t_probs * logp).sum(dim=1).mean())
                    loss = loss + float(proj_alpha) * proj_loss
                valid_losses.append(float(loss.item()))
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")

        # Best model selection by hit-rate.
        valid_hit_rates_at = evaluate_hit_rates_at(
            device,
            models,
            x_valid,
            labels_best_valid.detach().cpu().numpy(),
            labels_second_valid.detach().cpu().numpy(),
            batch_size,
            topn=eval_topn,
            score_mode="log_softmax",
            trial_id=trial_id,
        )
        valid_hit_rate = float(valid_hit_rates_at.get(3, 0.0))

        if valid_hit_rate >= best_hit_rate + 1e-9:
            best_hit_rate = valid_hit_rate
            best_hit_epoch = epoch
            best_hit_rates_at = valid_hit_rates_at
            best_hit_state = {k: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()} for k, m in models.items()}

        if valid_loss < best_valid_loss - 1e-6:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        pbar.set_postfix_str(
            f"loss={valid_loss:.5f} best_loss={best_valid_loss:.5f} hit@3={valid_hit_rate*100:.2f}% no_imp={epochs_no_improve}"
        )
        pbar.update(1)
        if epochs_no_improve >= patience:
            break
    pbar.close()

    if best_hit_state is None:
        raise RuntimeError("failed to capture best hit-rate state")
    for head, state in best_hit_state.items():
        models[head].load_state_dict(state)
    return best_valid_loss, best_hit_epoch, best_hit_rates_at


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _batch_slices(n: int, batch_size: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    bs = max(1, int(batch_size))
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]


def train_head(
    head: str,
    model: nn.Module,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    *,
    batch_size: int,
    trial_id: int,
    head_index: int,
    total_heads: int,
    max_epochs: int,
    patience: int,
    lr: float,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    total_epochs = max_epochs * total_heads
    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    desc = f"trial {trial_id:04d} {head}"
    pbar = tqdm(total=max_epochs, desc=desc, leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=device, dtype=torch.int64)
        for s, e in _batch_slices(n_train, batch_size):
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            optimizer.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for s, e in _batch_slices(n_valid, batch_size):
                xb = x_valid[s:e]
                yb = y_valid[s:e]
                logits = model(xb)
                loss = soft_ce_loss(logits, yb)
                losses.append(float(loss.item()))
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        pbar.set_postfix_str(f"valid_loss={valid_loss:.5f} best={best_loss:.5f} no_improve={epochs_no_improve}")
        pbar.update(1)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, best_epoch

def train_head_proj(
    head: str,
    model: nn.Module,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    *,
    batch_size: int,
    trial_id: int,
    head_index: int,
    total_heads: int,
    max_epochs: int,
    patience: int,
    lr: float,
) -> tuple[float, int]:
    # Regression from features -> per-class score vector (higher is better).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    desc = f"trial {trial_id:04d} {head}"
    pbar = tqdm(total=max_epochs, desc=desc, leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=device, dtype=torch.int64)
        for s, e in _batch_slices(n_train, batch_size):
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for s, e in _batch_slices(n_valid, batch_size):
                xb = x_valid[s:e]
                yb = y_valid[s:e]
                pred = model(xb)
                loss = F.smooth_l1_loss(pred, yb)
                losses.append(float(loss.item()))
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        pbar.set_postfix_str(f"valid_loss={valid_loss:.5f} best={best_loss:.5f} no_improve={epochs_no_improve}")
        pbar.update(1)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, best_epoch

def train_head_top2(
    head: str,
    model: nn.Module,
    device: torch.device,
    x_train: torch.Tensor,
    top2_train: torch.Tensor,
    x_valid: torch.Tensor,
    top2_valid: torch.Tensor,
    *,
    batch_size: int,
    trial_id: int,
    head_index: int,
    total_heads: int,
    max_epochs: int,
    patience: int,
    lr: float,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    desc = f"trial {trial_id:04d} {head}"
    pbar = tqdm(total=max_epochs, desc=desc, leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=device, dtype=torch.int64)
        for s, e in _batch_slices(n_train, batch_size):
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            yb = top2_train.index_select(0, idx)
            optimizer.zero_grad()
            logits = model(xb)
            loss = top2_soft_ce_loss(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for s, e in _batch_slices(n_valid, batch_size):
                xb = x_valid[s:e]
                yb = top2_valid[s:e]
                logits = model(xb)
                loss = top2_soft_ce_loss(logits, yb)
                losses.append(float(loss.item()))
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        pbar.set_postfix_str(f"valid_loss={valid_loss:.5f} best={best_loss:.5f} no_improve={epochs_no_improve}")
        pbar.update(1)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, best_epoch


def _topk_with_tiebreak_torch(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (topk_scores, topk_ids) with deterministic tiebreak by smaller class id."""
    if k <= 0:
        empty = torch.empty((scores.shape[0], 0), device=scores.device, dtype=scores.dtype)
        empty_i = torch.empty((scores.shape[0], 0), device=scores.device, dtype=torch.int64)
        return empty, empty_i
    classes = int(scores.shape[1])
    ids = torch.arange(classes, device=scores.device, dtype=torch.float32).unsqueeze(0)
    eps = 1e-6
    # Higher is better; for ties, prefer smaller id -> larger bonus for smaller id.
    adjusted = scores + (float(classes) - ids) * eps
    _, top_idx = torch.topk(adjusted, k, dim=1)
    top_scores = scores.gather(1, top_idx)
    return top_scores, top_idx


def _parse_eval_topn(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("--eval-topn must not be empty")
    topn: list[int] = []
    for p in parts:
        n = int(p)
        if n <= 0:
            raise ValueError(f"--eval-topn contains non-positive N: {p}")
        topn.append(n)
    topn = sorted(set(topn))
    if topn[-1] > 768:
        raise ValueError(f"--eval-topn max N must be <= 768, got {topn[-1]}")
    return topn


def evaluate_hit_rates_at(
    device: torch.device,
    models: dict[str, nn.Module],
    x_valid: torch.Tensor,
    labels_best_valid: np.ndarray,
    labels_second_valid: np.ndarray,
    batch_size: int,
    *,
    topn: list[int],
    score_mode: str = "log_softmax",
    trial_id: int | None = None,
) -> dict[int, float]:
    total = int(x_valid.shape[0])
    if total <= 0:
        return {n: 0.0 for n in topn}

    for head in models.values():
        head.eval()

    labels_best_t = torch.as_tensor(labels_best_valid, dtype=torch.int64, device=device)
    labels_second_t = torch.as_tensor(labels_second_valid, dtype=torch.int64, device=device)

    best_pred = labels_best_t[:, 0]
    best_filter = (labels_best_t[:, 1] << 4) | (labels_best_t[:, 2] << 2) | labels_best_t[:, 3]
    best_reorder = labels_best_t[:, 4]
    best_interleave = labels_best_t[:, 5]

    second_pred = labels_second_t[:, 0]
    second_filter = (labels_second_t[:, 1] << 4) | (labels_second_t[:, 2] << 2) | labels_second_t[:, 3]
    second_reorder = labels_second_t[:, 4]
    second_interleave = labels_second_t[:, 5]

    best_rank = (((best_pred * 96 + best_filter) * 8 + best_reorder) * 2 + best_interleave).to(torch.int64)
    second_rank = (((second_pred * 96 + second_filter) * 8 + second_reorder) * 2 + second_interleave).to(torch.int64)

    max_n = int(max(topn))
    hits_at = {n: 0 for n in topn}
    batch_slices = _batch_slices(total, batch_size)
    eval_desc = "eval" if trial_id is None else f"eval trial {trial_id:04d}"
    pbar = tqdm(total=len(batch_slices), desc=eval_desc, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for s, e in batch_slices:
            xb = x_valid[s:e]
            if score_mode == "raw":
                logits = {name: model(xb) for name, model in models.items()}
            else:
                logits = {name: torch.log_softmax(model(xb), dim=1) for name, model in models.items()}

            pred_scores, pred_ids = _topk_with_tiebreak_torch(logits["predictor"], BEAM_WIDTHS["predictor"])
            perm_scores, perm_ids = _topk_with_tiebreak_torch(logits["cf_perm"], BEAM_WIDTHS["cf_perm"])
            prim_scores, prim_ids = _topk_with_tiebreak_torch(logits["cf_primary"], BEAM_WIDTHS["cf_primary"])
            sec_scores, sec_ids = _topk_with_tiebreak_torch(logits["cf_secondary"], BEAM_WIDTHS["cf_secondary"])
            re_scores, re_ids = _topk_with_tiebreak_torch(logits["reorder"], BEAM_WIDTHS["reorder"])
            inter_scores, inter_ids = _topk_with_tiebreak_torch(logits["interleave"], BEAM_WIDTHS["interleave"])

            bsz = int(xb.shape[0])
            total_scores = (
                pred_scores[:, :, None, None, None, None, None]
                + perm_scores[:, None, :, None, None, None, None]
                + prim_scores[:, None, None, :, None, None, None]
                + sec_scores[:, None, None, None, :, None, None]
                + re_scores[:, None, None, None, None, :, None]
                + inter_scores[:, None, None, None, None, None, :]
            )

            pred_id_b = pred_ids[:, :, None, None, None, None, None]
            perm_id_b = perm_ids[:, None, :, None, None, None, None]
            prim_id_b = prim_ids[:, None, None, :, None, None, None]
            sec_id_b = sec_ids[:, None, None, None, :, None, None]
            re_id_b = re_ids[:, None, None, None, None, :, None]
            inter_id_b = inter_ids[:, None, None, None, None, None, :]
            filter_code = (perm_id_b << 4) | (prim_id_b << 2) | sec_id_b
            tuple_rank = (((pred_id_b * 96 + filter_code) * 8 + re_id_b) * 2 + inter_id_b).to(torch.int64)

            score_flat = total_scores.reshape(bsz, -1)
            rank_flat = tuple_rank.reshape(bsz, -1)
            # Deterministic tiebreak for identical combined scores: prefer smaller tuple_rank.
            eps = 1e-6
            score_adj = score_flat + (rank_flat.max(dim=1, keepdim=True).values - rank_flat).to(score_flat.dtype) * eps
            _, top_pos = torch.topk(score_adj, max_n, dim=1)
            top_rank = rank_flat.gather(1, top_pos)
            match = (top_rank == best_rank[s:e, None]) | (top_rank == second_rank[s:e, None])
            for n in topn:
                hits_at[n] += int(match[:, :n].any(dim=1).sum().item())
            pbar.update(1)
    pbar.close()
    return {n: float(hits_at[n]) / float(total) for n in topn}


def load_progress(progress_path: Path) -> tuple[int, float, int, dict[str, Any] | None]:
    if not progress_path.exists():
        return 0, -1.0, 0, None
    best_rate = -1.0
    best_params_total = 1 << 60
    best_entry: dict[str, Any] | None = None
    no_improve = 0
    last_best = -1.0
    max_trial = -1
    with progress_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            entry = json.loads(line)
            max_trial = max(max_trial, int(entry.get("trial_id", -1)))
            rate = float(entry.get("valid_hit_rate", -1.0))
            params_total = int(entry.get("params", {}).get("total", 1 << 60))
            if rate >= best_rate + 0.001:
                best_rate = rate
                best_params_total = params_total
                best_entry = entry
                last_best = rate
                no_improve = 0
            else:
                if abs(rate - best_rate) <= 1e-9 and params_total < best_params_total:
                    best_params_total = params_total
                    best_entry = entry
                if last_best >= 0:
                    no_improve += 1
    return max_trial + 1, best_rate, no_improve, best_entry


def get_trial_config(
    trial_id: int, feature_specs: list[FeatureSpec], arch_specs: list[dict[str, list[int]]], seed: int
) -> tuple[FeatureSpec, dict[str, list[int]]]:
    if trial_id < len(feature_specs) * len(arch_specs):
        # Enumerate with architecture varying fastest so early trials cover capacity quickly.
        arch_idx = trial_id % len(arch_specs)
        feat_idx = (trial_id // len(arch_specs)) % len(feature_specs)
        return feature_specs[feat_idx], arch_specs[arch_idx]
    rng = np.random.default_rng(seed)
    total = len(feature_specs) * len(arch_specs)
    picks = trial_id - total + 1
    feat = feature_specs[0]
    arch = arch_specs[0]
    for _ in range(picks):
        feat = rng.choice(feature_specs)
        arch = rng.choice(arch_specs)
    return feat, arch


def main() -> None:
    args = parse_args()
    eval_topn = _parse_eval_topn(args.eval_topn)
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 64)
        args.max_trials = 1 if args.max_trials is None else min(args.max_trials, 1)
    args.in_dir = resolve_in_dir(args.in_dir, smoke=args.smoke)
    device = ensure_cuda()
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(f"Using CUDA device {device_index}: {device_name}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or time_run_id()
    run_dir = args.run_root / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "dataset")
    ensure_dir(run_dir / "splits")

    max_blocks = args.max_blocks
    if (
        not args.smoke
        and max_blocks is None
        and args.in_dir.resolve()
        in {
            Path("ml/source/packed_out").resolve(),
            Path("ml/source/packed_out_entropy0").resolve(),
            Path("ml/source/packed_out_entropy0_filterbits").resolve(),
            Path("ml/source/packed_out_entropy0_filterbits2").resolve(),
        }
    ):
        max_blocks = 500_000
    dataset = build_dataset(
        run_dir,
        args.in_dir,
        args.seed,
        rebuild=args.rebuild_dataset,
        max_blocks=max_blocks,
    )

    feature_specs = build_feature_specs(dataset)
    arch_specs = build_arch_specs()
    config_path = run_dir / "config.json"
    config = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "neg_k": args.neg_k,
        "loss": args.loss,
        "target_temp": args.target_temp,
        "proj_alpha": args.proj_alpha,
        "proj_temp": args.proj_temp,
        "eval_topn": eval_topn,
        "run_id": run_id,
        "input_dir": str(args.in_dir),
        "max_blocks": max_blocks,
        "feature_specs": [
            {
                "name": spec.name,
                "raw_indices": spec.raw_indices,
                "transforms": spec.transforms,
                "include_raw": spec.include_raw,
            }
            for spec in feature_specs
        ],
        "arch_specs": arch_specs,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if "eval_topn" not in existing:
            existing["eval_topn"] = eval_topn
        if "neg_k" not in existing:
            existing["neg_k"] = args.neg_k
        if "loss" not in existing:
            existing["loss"] = args.loss
        if "target_temp" not in existing:
            existing["target_temp"] = args.target_temp
        if "proj_alpha" not in existing:
            existing["proj_alpha"] = args.proj_alpha
        if "proj_temp" not in existing:
            existing["proj_temp"] = args.proj_temp
        if existing != config:
            raise SystemExit("config.json exists and differs; use a new run_id for material changes")
    else:
        save_json(config_path, config)

    start_trial, best_rate, no_improve, best_from_progress = load_progress(run_dir / "progress.jsonl")
    max_trials = args.max_trials or len(feature_specs) * len(arch_specs)

    progress_path = run_dir / "progress.jsonl"
    best_path = run_dir / "best.json"

    best_trial_id = -1
    best_bundle_path = ""
    best_params_total = 1 << 60
    best_params_per_head: dict[str, int] = {}
    best_valid_loss_mean = float("inf")
    if best_path.exists():
        try:
            prev = json.loads(best_path.read_text(encoding="utf-8"))
            best_trial_id = int(prev.get("best_trial_id", prev.get("trial_id", -1)))
            best_bundle_path = str(prev.get("best_bundle_path", prev.get("bundle_path", "")))
            best_valid_loss_mean = float(prev.get("best_valid_loss_mean", prev.get("valid_loss_mean", best_valid_loss_mean)))
            params = prev.get("best_params", prev.get("params", {})) or {}
            if isinstance(params, dict):
                best_params_total = int(params.get("total", best_params_total))
                per_head = params.get("per_head", {})
                if isinstance(per_head, dict):
                    best_params_per_head = {str(k): int(v) for k, v in per_head.items()}
        except Exception:
            pass
    if best_from_progress is not None and (best_trial_id < 0 or not best_bundle_path):
        try:
            best_trial_id = int(best_from_progress.get("trial_id", best_trial_id))
            best_bundle_path = str(best_from_progress.get("bundle_path", best_bundle_path))
            best_valid_loss_mean = float(best_from_progress.get("valid_loss_mean", best_valid_loss_mean))
            params = best_from_progress.get("params", {}) or {}
            if isinstance(params, dict):
                best_params_total = int(params.get("total", best_params_total))
                per_head = params.get("per_head", {}) or {}
                if isinstance(per_head, dict):
                    best_params_per_head = {str(k): int(v) for k, v in per_head.items()}
        except Exception:
            pass

    targets: dict[str, np.ndarray] | None = None
    target_source = "none"
    if args.loss == "per_head_ce":
        if dataset.proj_top2 is not None:
            target_source = "proj_top2"
        else:
            target_source = "best_second_bits"
            bits_best = dataset.bits[:, 0]
            bits_second = dataset.bits[:, 1]
            targets = build_soft_targets(
                dataset.labels_best,
                dataset.labels_second,
                bits_best,
                bits_second,
                target_temp=args.target_temp,
            )
    elif args.loss == "per_head_proj":
        if dataset.proj_scores is None:
            raise SystemExit("per_head_proj requires dataset.proj_scores (rebuild dataset or use a dataset with proj_*_bits)")
        target_source = "proj_scores"
    elif args.loss == "tuple_nce_proj":
        if dataset.proj_scores is None:
            raise SystemExit("tuple_nce_proj requires dataset.proj_scores (rebuild dataset or use a dataset with proj_*_bits)")
        target_source = "proj_scores"

    if args.smoke:
        train_idx = dataset.train_indices[: min(256, dataset.train_indices.shape[0])]
        valid_idx = dataset.valid_indices[: min(128, dataset.valid_indices.shape[0])]
    else:
        train_idx = dataset.train_indices
        valid_idx = dataset.valid_indices

    def write_best_state(payload: dict[str, Any]) -> None:
        payload["timestamp"] = now_iso()
        save_json(best_path, payload)

    for trial_id in range(start_trial, max_trials):
        trial_t0 = time.perf_counter()
        feature_spec, arch_spec = get_trial_config(trial_id, feature_specs, arch_specs, args.seed)
        # Build features only for train/valid to avoid per-trial full-dataset concatenation and copies.
        # Sort for better memory locality; membership is unchanged.
        train_idx_sorted = np.sort(train_idx)
        valid_idx_sorted = np.sort(valid_idx)

        feat_t0 = time.perf_counter()
        non_pixel_train = apply_feature_pipeline(dataset.raw_numeric[train_idx_sorted], feature_spec)
        if non_pixel_train.shape[1] > 256:
            continue
        non_pixel_valid = apply_feature_pipeline(dataset.raw_numeric[valid_idx_sorted], feature_spec)

        non_pixel_train_z, non_pixel_valid_z, non_pixel_mean, non_pixel_std = standardize_features(
            non_pixel_train, non_pixel_valid
        )

        x_train_np = np.concatenate([dataset.pixels[train_idx_sorted], non_pixel_train_z], axis=1).astype(
            np.float32, copy=False
        )
        x_valid_np = np.concatenate([dataset.pixels[valid_idx_sorted], non_pixel_valid_z], axis=1).astype(
            np.float32, copy=False
        )
        input_dim = int(x_train_np.shape[1])

        # Move feature matrices to GPU once per trial; keep them there for all heads.
        torch.cuda.synchronize(device)
        to_dev_t0 = time.perf_counter()
        x_train = torch.from_numpy(x_train_np).to(device)
        x_valid = torch.from_numpy(x_valid_np).to(device)
        torch.cuda.synchronize(device)
        to_dev_t1 = time.perf_counter()

        trial_dir = run_dir / "artifacts" / f"trial_{trial_id:04d}"
        ensure_dir(trial_dir)

        model_bundle = {
            "run_id": run_id,
            "trial_id": trial_id,
            "input_dim": input_dim,
            "pixel_dim": RAW_RGB_DIMS,
            "raw_feature_names": dataset.raw_names,
            "feature_spec": {
                "name": feature_spec.name,
                "raw_indices": feature_spec.raw_indices,
                "transforms": feature_spec.transforms,
                "include_raw": feature_spec.include_raw,
            },
            "target_source": target_source,
            "score_mode": "raw" if args.loss == "per_head_proj" else "log_softmax",
            "proj_alpha": float(args.proj_alpha),
            "proj_temp": float(args.proj_temp),
            "feature_norm": {
                "non_pixel_mean": non_pixel_mean.tolist(),
                "non_pixel_std": non_pixel_std.tolist(),
            },
            "heads": {},
        }

        models: dict[str, nn.Module] = {}
        head_losses: dict[str, float] = {}
        head_epochs: dict[str, int] = {}
        head_params: dict[str, int] = {}
        status = "ok"
        error_summary = ""
        timings = {
            "feature_build_sec": float(to_dev_t0 - feat_t0),
            "feature_to_device_sec": float(to_dev_t1 - to_dev_t0),
        }

        try:
            for head in HEAD_ORDER:
                hidden_sizes = arch_spec[head]
                model = MLP(input_dim, hidden_sizes, HEADS[head]).to(device)
                models[head] = model
                head_params[head] = count_params(model)

            if args.loss == "per_head_ce":
                for head_idx, head in enumerate(HEAD_ORDER):
                    head_t0 = time.perf_counter()
                    model = models[head]
                    if dataset.proj_top2 is not None:
                        top2_train = torch.from_numpy(dataset.proj_top2[train_idx_sorted, head_idx]).to(device)
                        top2_valid = torch.from_numpy(dataset.proj_top2[valid_idx_sorted, head_idx]).to(device)
                        valid_loss, best_epoch = train_head_top2(
                            head,
                            model,
                            device,
                            x_train,
                            top2_train,
                            x_valid,
                            top2_valid,
                            batch_size=args.batch_size,
                            trial_id=trial_id,
                            head_index=head_idx,
                            total_heads=len(HEAD_ORDER),
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            lr=args.lr,
                        )
                    else:
                        assert targets is not None
                        y_train = torch.from_numpy(targets[head][train_idx_sorted]).to(device)
                        y_valid = torch.from_numpy(targets[head][valid_idx_sorted]).to(device)
                        valid_loss, best_epoch = train_head(
                            head,
                            model,
                            device,
                            x_train,
                            y_train,
                            x_valid,
                            y_valid,
                            batch_size=args.batch_size,
                            trial_id=trial_id,
                            head_index=head_idx,
                            total_heads=len(HEAD_ORDER),
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            lr=args.lr,
                        )
                    torch.cuda.synchronize(device)
                    timings[f"train_{head}_sec"] = float(time.perf_counter() - head_t0)
                    head_losses[head] = valid_loss
                    head_epochs[head] = best_epoch
            elif args.loss == "per_head_proj":
                # Regress per-class projected scores (higher is better).
                # Important: keep absolute scale comparable across heads, since inference sums head scores.
                proj = dataset.proj_scores
                assert proj is not None
                for head_idx, head in enumerate(HEAD_ORDER):
                    head_t0 = time.perf_counter()
                    model = models[head]
                    classes = int(HEADS[head])
                    off = int(PROJ_OFFSETS[head])
                    y_train_np = proj[train_idx_sorted, off : off + classes].astype(np.float32, copy=False)
                    y_valid_np = proj[valid_idx_sorted, off : off + classes].astype(np.float32, copy=False)
                    # Targets are only defined up to an additive per-sample constant (adding the same
                    # value to all classes for a head doesn't change head argmax nor tuple ranking).
                    # Normalize to make regression well-conditioned, especially when many entries are
                    # "unreachable" (very negative).
                    y_train_np = y_train_np - y_train_np.max(axis=1, keepdims=True)
                    y_valid_np = y_valid_np - y_valid_np.max(axis=1, keepdims=True)
                    # Clip extreme unreachable-derived values to limit gradient spikes.
                    y_train_np = np.clip(y_train_np, -20.0, 0.0, dtype=np.float32)
                    y_valid_np = np.clip(y_valid_np, -20.0, 0.0, dtype=np.float32)
                    y_train = torch.from_numpy(y_train_np).to(device)
                    y_valid = torch.from_numpy(y_valid_np).to(device)
                    valid_loss, best_epoch = train_head_proj(
                        head,
                        model,
                        device,
                        x_train,
                        y_train,
                        x_valid,
                        y_valid,
                        batch_size=args.batch_size,
                        trial_id=trial_id,
                        head_index=head_idx,
                        total_heads=len(HEAD_ORDER),
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        lr=args.lr,
                    )
                    torch.cuda.synchronize(device)
                    timings[f"train_{head}_sec"] = float(time.perf_counter() - head_t0)
                    head_losses[head] = valid_loss
                    head_epochs[head] = best_epoch
            else:
                train_t0 = time.perf_counter()
                labels_best_train = torch.from_numpy(dataset.labels_best[train_idx_sorted].astype(np.int64)).to(device)
                labels_second_train = torch.from_numpy(dataset.labels_second[train_idx_sorted].astype(np.int64)).to(device)
                labels_best_valid = torch.from_numpy(dataset.labels_best[valid_idx_sorted].astype(np.int64)).to(device)
                labels_second_valid = torch.from_numpy(dataset.labels_second[valid_idx_sorted].astype(np.int64)).to(device)
                proj_scores_train = None
                proj_scores_valid = None
                if args.loss == "tuple_nce_proj":
                    assert dataset.proj_scores is not None
                    proj_scores_train = torch.from_numpy(dataset.proj_scores[train_idx_sorted].astype(np.float32)).to(device)
                    proj_scores_valid = torch.from_numpy(dataset.proj_scores[valid_idx_sorted].astype(np.float32)).to(device)
                valid_loss, best_epoch, valid_hit_rates_at = train_all_heads(
                    device,
                    models,
                    x_train,
                    labels_best_train,
                    labels_second_train,
                    x_valid,
                    labels_best_valid,
                    labels_second_valid,
                    proj_scores_train=proj_scores_train,
                    proj_scores_valid=proj_scores_valid,
                    proj_alpha=float(args.proj_alpha) if args.loss == "tuple_nce_proj" else 0.0,
                    proj_temp=float(args.proj_temp),
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    lr=args.lr,
                    neg_k=args.neg_k,
                    loss_kind=args.loss,
                    eval_topn=eval_topn,
                    trial_id=trial_id,
                )
                torch.cuda.synchronize(device)
                timings["train_joint_sec"] = float(time.perf_counter() - train_t0)
                head_losses = {"joint": float(valid_loss)}
                head_epochs = {"joint": int(best_epoch)}

            eval_t0 = time.perf_counter()
            valid_hit_rates_at = evaluate_hit_rates_at(
                device,
                models,
                x_valid,
                dataset.labels_best[valid_idx_sorted],
                dataset.labels_second[valid_idx_sorted],
                args.batch_size,
                topn=eval_topn,
                score_mode=model_bundle["score_mode"],
                trial_id=trial_id,
            )
            valid_hit_rate = float(valid_hit_rates_at.get(3, 0.0))
            torch.cuda.synchronize(device)
            timings["eval_sec"] = float(time.perf_counter() - eval_t0)

            for head in HEAD_ORDER:
                hidden_sizes = arch_spec[head]
                model_path = trial_dir / f"{head}.pt"
                torch.save(models[head].state_dict(), model_path)
                model_bundle["heads"][head] = {
                    "path": str(model_path),
                    "hidden_sizes": hidden_sizes,
                    "classes": HEADS[head],
                }
        except Exception as exc:
            valid_hit_rate = 0.0
            valid_hit_rates_at = {n: 0.0 for n in eval_topn}
            status = "failed"
            error_summary = str(exc)
        timings["trial_total_sec"] = float(time.perf_counter() - trial_t0)

        total_params = int(sum(head_params.values()))
        valid_loss_mean = float(np.mean(list(head_losses.values()))) if head_losses else float("inf")
        hit_rate_str = " ".join(f"hit@{n}={valid_hit_rates_at.get(n, 0.0) * 100:.2f}%" for n in eval_topn)
        print(f"[trial {trial_id:04d}] valid {hit_rate_str}", flush=True)

        save_json(trial_dir / "feature_spec.json", model_bundle["feature_spec"])
        save_json(
            trial_dir / "eval.json",
            {
                "valid_hit_rate": valid_hit_rate,
                "valid_hit_rates_at": {str(k): float(v) for k, v in sorted(valid_hit_rates_at.items())},
                "valid_loss_mean": valid_loss_mean,
                "head_losses": head_losses,
            },
        )
        bundle_path = trial_dir / "bundle.json"
        save_json(bundle_path, model_bundle)

        progress_entry = {
            "trial_id": trial_id,
            "timestamp": now_iso(),
            "git_commit": git_commit_hash(),
            "dataset": {
                "training_all_jsonl": str(args.in_dir / "training.all.jsonl"),
                "labels_all_bin": str(args.in_dir / "labels.all.bin"),
                "count_filtered": int(dataset.indices.shape[0]),
                "count_train": int(train_idx.shape[0]),
                "count_valid": int(valid_idx.shape[0]),
                "max_blocks": max_blocks,
            },
            "feature_spec": model_bundle["feature_spec"],
            "arch_spec": arch_spec,
            "hyperparams": {
                "seed": args.seed,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "neg_k": args.neg_k,
                "loss": args.loss,
                "target_temp": args.target_temp,
                "smoke": args.smoke,
            },
            "valid_loss_mean": valid_loss_mean,
            "valid_hit_rate": valid_hit_rate,
            "valid_hit_rates_at": {str(k): float(v) for k, v in sorted(valid_hit_rates_at.items())},
            "params": {
                "per_head": head_params,
                "total": total_params,
            },
            "head_losses": head_losses,
            "head_best_epoch": head_epochs,
            "bundle_path": str(bundle_path),
            "status": status,
            "error": error_summary,
            "timings": timings,
        }
        with progress_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(progress_entry) + "\n")

        updated = False
        if valid_hit_rate >= best_rate + 0.001:
            updated = True
        elif abs(valid_hit_rate - best_rate) <= 1e-9 and total_params > 0:
            if total_params < best_params_total:
                updated = True

        if updated:
            best_rate = valid_hit_rate
            no_improve = 0
            best_trial_id = trial_id
            best_bundle_path = str(bundle_path)
            best_valid_loss_mean = float(valid_loss_mean)
            best_params_total = int(total_params)
            best_params_per_head = dict(head_params)
        else:
            no_improve += 1

        best_payload = {
            "status": "ok" if best_rate >= 0.9 else "below_threshold",
            "best_trial_id": best_trial_id,
            "best_valid_hit_rate": best_rate,
            "best_valid_loss_mean": best_valid_loss_mean,
            "best_params": {"per_head": best_params_per_head, "total": best_params_total},
            "best_bundle_path": best_bundle_path,
            "latest_trial_id": trial_id,
            "latest_valid_hit_rate": valid_hit_rate,
            "latest_valid_loss_mean": valid_loss_mean,
            "latest_params": {"per_head": head_params, "total": total_params},
            "latest_bundle_path": str(bundle_path),
            "no_improve": no_improve,
        }
        write_best_state(best_payload)

        if no_improve >= 10:
            failure_payload = {
                "status": "failed_no_improve",
                "trial_id": trial_id,
                "best_trial_id": best_trial_id,
                "best_valid_hit_rate": best_rate,
                "best_bundle_path": best_bundle_path,
                "no_improve": no_improve,
            }
            write_best_state(failure_payload)
            break


def time_run_id() -> str:
    return now_iso().replace(":", "").replace("-", "").replace("T", "_")


if __name__ == "__main__":
    main()
