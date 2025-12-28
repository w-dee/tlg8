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
PAIRWISE_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("predictor", "cf_perm"),
    ("predictor", "cf_primary"),
    ("predictor", "cf_secondary"),
    ("reorder", "cf_perm"),
    ("reorder", "cf_primary"),
    ("reorder", "cf_secondary"),
]
DEFAULT_BEAM_WIDTHS = {
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


def _pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


class PairwiseInteractions(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim: int,
        *,
        pairs: list[tuple[str, str]] | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.scale = float(scale)
        self.pairs = list(pairs) if pairs is not None else list(PAIRWISE_DEFAULT_PAIRS)
        if self.dim <= 0:
            raise ValueError("PairwiseInteractions dim must be positive")
        for a, b in self.pairs:
            if a not in HEADS or b not in HEADS:
                raise ValueError(f"unknown head in pairwise pair: {(a, b)}")

        used_heads: set[str] = set()
        for a, b in self.pairs:
            used_heads.add(a)
            used_heads.add(b)
        self.emb = nn.ModuleDict({h: nn.Embedding(int(HEADS[h]), self.dim) for h in sorted(used_heads)})
        self.gates = nn.ModuleDict({_pair_key(a, b): nn.Linear(self.input_dim, self.dim) for a, b in self.pairs})

    def _gate(self, x: torch.Tensor, a: str, b: str) -> torch.Tensor:
        gate = self.gates[_pair_key(a, b)](x)
        return torch.tanh(gate)

    def score_pos(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(torch.int64)
        score = torch.zeros((labels.shape[0],), device=labels.device, dtype=x.dtype)
        for a, b in self.pairs:
            ia = int(HEAD_ORDER.index(a))
            ib = int(HEAD_ORDER.index(b))
            ea = self.emb[a](labels[:, ia])
            eb = self.emb[b](labels[:, ib])
            g = self._gate(x, a, b)
            score = score + (ea * g * eb).sum(dim=1)
        return score * float(self.scale)

    def score_neg(self, x: torch.Tensor, neg_labels: torch.Tensor) -> torch.Tensor:
        # neg_labels: [B, K, 6]
        neg_labels = neg_labels.to(torch.int64)
        score = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]), device=neg_labels.device, dtype=x.dtype)
        for a, b in self.pairs:
            ia = int(HEAD_ORDER.index(a))
            ib = int(HEAD_ORDER.index(b))
            ea = self.emb[a](neg_labels[:, :, ia])
            eb = self.emb[b](neg_labels[:, :, ib])
            g = self._gate(x, a, b)[:, None, :]
            score = score + (ea * g * eb).sum(dim=2)
        return score * float(self.scale)


class TupleReranker(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        if self.embed_dim <= 0:
            raise ValueError("TupleReranker embed_dim must be positive")
        if self.hidden <= 0:
            raise ValueError("TupleReranker hidden must be positive")
        self.x_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.embed_dim),
            nn.ReLU(),
        )
        self.emb = nn.ModuleDict({h: nn.Embedding(int(HEADS[h]), self.embed_dim) for h in HEAD_ORDER})
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def _tuple_embed_pos(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(torch.int64)
        out = torch.zeros((labels.shape[0], self.embed_dim), device=labels.device, dtype=torch.float32)
        for head_idx, head in enumerate(HEAD_ORDER):
            out = out + self.emb[head](labels[:, head_idx])
        return out

    def _tuple_embed_many(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B,K,6] -> [B,K,D]
        labels = labels.to(torch.int64)
        out = torch.zeros((labels.shape[0], labels.shape[1], self.embed_dim), device=labels.device, dtype=torch.float32)
        for head_idx, head in enumerate(HEAD_ORDER):
            out = out + self.emb[head](labels[:, :, head_idx])
        return out

    def score_pos(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_emb = self.x_net(x)
        t_emb = self._tuple_embed_pos(labels)
        feat = torch.cat([x_emb, t_emb], dim=1)
        return self.mlp(feat).squeeze(1)

    def score_many(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B,K,6]
        x_emb = self.x_net(x)  # [B,D]
        t_emb = self._tuple_embed_many(labels)  # [B,K,D]
        x_exp = x_emb[:, None, :].expand(-1, t_emb.shape[1], -1)
        feat = torch.cat([x_exp, t_emb], dim=2).reshape(-1, self.embed_dim * 2)
        out = self.mlp(feat).reshape(t_emb.shape[0], t_emb.shape[1])
        return out


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
        "--per-head-target",
        type=str,
        default="auto",
        choices=["auto", "proj_top2", "best_second_bits"],
        help="Target source for --loss=per_head_ce (default: auto; prefers proj_top2 if present)",
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
        "--hard-neg-filterbits",
        action="store_true",
        help="Use filterbits-derived hard negatives for tuple losses",
    )
    parser.add_argument(
        "--hard-neg-topk",
        type=int,
        default=8,
        help="Top-K filter candidates per interleave for hard negatives (default: 8)",
    )
    parser.add_argument(
        "--hard-neg-proj",
        action="store_true",
        help="Use proj_*_bits-derived hard negatives for tuple losses",
    )
    parser.add_argument(
        "--hard-neg-proj-topk",
        type=int,
        default=4,
        help="Top-K per head from proj scores for hard negatives (default: 4)",
    )
    parser.add_argument(
        "--hard-neg-proj-near",
        action="store_true",
        help="Restrict proj hard negatives to candidates near the best proj score",
    )
    parser.add_argument(
        "--hard-neg-proj-margin",
        type=float,
        default=0.5,
        help="Score margin for near-best proj candidates (default: 0.5)",
    )
    parser.add_argument(
        "--hard-neg-proj-fix-pr",
        action="store_true",
        help="Fix predictor/reorder to best/second when sampling proj hard negatives",
    )
    parser.add_argument(
        "--pretrain-filterbits-epochs",
        type=int,
        default=0,
        help="Pretrain filter-related heads on filterbits heuristics for N epochs (default: 0)",
    )
    parser.add_argument(
        "--pretrain-filterbits-lr",
        type=float,
        default=3e-4,
        help="Learning rate for filterbits pretrain (default: 3e-4)",
    )
    parser.add_argument(
        "--pairwise-dim",
        type=int,
        default=0,
        help="Enable low-rank pairwise interactions with this dimension (0 disables; default: 0)",
    )
    parser.add_argument(
        "--pairwise-scale",
        type=float,
        default=1.0,
        help="Scale applied to pairwise interaction score (default: 1.0)",
    )
    parser.add_argument(
        "--filterbits-prior",
        action="store_true",
        help="Add filterbits-derived heuristic logits as a prior (model predicts residual)",
    )
    parser.add_argument(
        "--filterbits-prior-weight",
        type=float,
        default=1.0,
        help="Weight for filterbits prior logits (default: 1.0)",
    )
    parser.add_argument(
        "--filterbits-prior-temp",
        type=float,
        default=20.0,
        help="Temperature for filterbits prior logits (bits scale; default: 20.0)",
    )
    parser.add_argument(
        "--energy-prior",
        action="store_true",
        help="Add predictor/reorder heuristic logits derived from energy/TV arrays as a prior",
    )
    parser.add_argument(
        "--energy-prior-weight",
        type=float,
        default=1.0,
        help="Weight for energy prior logits (default: 1.0)",
    )
    parser.add_argument(
        "--energy-prior-temp",
        type=float,
        default=1.0,
        help="Temperature for energy prior logits (default: 1.0)",
    )
    parser.add_argument(
        "--reranker-embed-dim",
        type=int,
        default=0,
        help="Enable an optional joint tuple reranker with this embedding dim (0 disables; default: 0)",
    )
    parser.add_argument(
        "--reranker-hidden",
        type=int,
        default=128,
        help="Hidden width for reranker MLP (default: 128)",
    )
    parser.add_argument(
        "--reranker-epochs",
        type=int,
        default=5,
        help="Epochs to train reranker after heads (default: 5)",
    )
    parser.add_argument(
        "--reranker-lr",
        type=float,
        default=1e-3,
        help="Learning rate for reranker (default: 1e-3)",
    )
    parser.add_argument(
        "--beam-widths",
        type=str,
        default="",
        help=(
            "Beam widths as comma-separated head=K pairs (e.g. predictor=8,cf_perm=6,cf_primary=4,cf_secondary=4,reorder=8,interleave=2); "
            "empty uses defaults"
        ),
    )
    parser.add_argument(
        "--beam-full",
        action="store_true",
        help="Set beam widths to full class counts (8,6,4,4,8,2)",
    )
    parser.add_argument(
        "--beam-candidate-chunk",
        type=int,
        default=2048,
        help="Chunk size for enumerating beam candidates (default: 2048)",
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


def _candidate_count(beam_widths: dict[str, int]) -> int:
    total = 1
    for head in HEAD_ORDER:
        total *= int(beam_widths[head])
    return int(total)


def _parse_beam_widths(spec: str, *, beam_full: bool) -> dict[str, int]:
    if beam_full:
        return {h: int(HEADS[h]) for h in HEAD_ORDER}
    if not spec.strip():
        return dict(DEFAULT_BEAM_WIDTHS)
    out = dict(DEFAULT_BEAM_WIDTHS)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"--beam-widths expects head=K pairs, got: {part!r}")
        head, val = part.split("=", 1)
        head = head.strip()
        if head not in HEADS:
            raise ValueError(f"--beam-widths unknown head: {head!r}")
        k = int(val.strip())
        if k <= 0:
            raise ValueError(f"--beam-widths {head} must be positive, got {k}")
        out[head] = min(k, int(HEADS[head]))
    for head in HEAD_ORDER:
        if head not in out:
            raise ValueError(f"--beam-widths missing head: {head}")
    return out


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

    # Pixel-local (8x8-only) gradient/frequency summaries already present in JSONL.
    freq_grad_indices: list[int] = []
    freq_grad_indices += _find_prefix("luma_pool4x4[")
    freq_grad_indices += _find_prefix("luma_dct4x4[")
    freq_grad_indices += (
        _find_exact("luma_mean")
        + _find_exact("luma_var")
        + _find_exact("grad_abs_dx_mean")
        + _find_exact("grad_abs_dy_mean")
        + _find_exact("grad_abs_mean")
        + _find_exact("edge_density")
        + _find_exact("mean_r")
        + _find_exact("mean_g")
        + _find_exact("mean_b")
        + _find_exact("var_r")
        + _find_exact("var_g")
        + _find_exact("var_b")
    )
    # Optionally add a tiny amount of reorder complexity signal.
    freq_grad_indices += reorder_tv_mean + reorder_tv2_mean
    freq_grad_indices = freq_grad_indices[:256]

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
    if len(freq_grad_indices) >= 16:
        specs.append(
            FeatureSpec(
                name="freq_grad_raw",
                raw_indices=freq_grad_indices,
                transforms=[],
                include_raw=True,
            )
        )
        specs.append(
            FeatureSpec(
                name="freq_grad_abs",
                raw_indices=freq_grad_indices,
                transforms=[{"kind": "abs", "apply_to": "all"}],
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


def _find_feature_indices(raw_names: list[str], prefix: str, expected: int) -> list[int]:
    indices = [i for i, name in enumerate(raw_names) if name.startswith(prefix)]
    if expected and len(indices) != expected:
        return []
    return indices


def _topk_smallest(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((arr.shape[0], 0), dtype=np.int64)
    k = min(k, arr.shape[1])
    part = np.argpartition(arr, k - 1, axis=1)[:, :k]
    # Sort within top-k for deterministic ordering.
    row = np.arange(arr.shape[0])[:, None]
    sub = arr[row, part]
    order = np.argsort(sub, axis=1)
    return part[row, order].astype(np.int64, copy=False)


def _topk_largest(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((arr.shape[0], 0), dtype=np.int64)
    k = min(k, arr.shape[1])
    part = np.argpartition(-arr, k - 1, axis=1)[:, :k]
    row = np.arange(arr.shape[0])[:, None]
    sub = arr[row, part]
    order = np.argsort(-sub, axis=1)
    return part[row, order].astype(np.int64, copy=False)


def _topk_near_best(arr: np.ndarray, k: int, margin: float) -> np.ndarray:
    if k <= 0:
        return np.empty((arr.shape[0], 0), dtype=np.int64)
    k = min(k, arr.shape[1])
    best = arr.max(axis=1, keepdims=True)
    mask = arr >= (best - float(margin))
    order = np.argsort(-arr, axis=1)
    out = np.empty((arr.shape[0], k), dtype=np.int64)
    for i in range(arr.shape[0]):
        cand = order[i][mask[i][order[i]]]
        if cand.size < k:
            fill = order[i][: k - cand.size]
            cand = np.concatenate([cand, fill], axis=0)
        out[i] = cand[:k]
    return out


def _filterbits_heuristic_targets(
    raw_numeric: np.ndarray,
    raw_names: list[str],
    indices: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Build top-2 targets per head from filterbits features (none/interleave)."""
    bits_none_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_none_min_by_filter[", 96)
    bits_inter_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_interleave_min_by_filter[", 96)
    if not bits_none_idx or not bits_inter_idx:
        return None
    bits_none = raw_numeric[indices][:, bits_none_idx]
    bits_inter = raw_numeric[indices][:, bits_inter_idx]
    bits_best = np.minimum(bits_none, bits_inter)

    # Map filter->perm/primary/secondary by min bits.
    perm_bits = np.full((bits_best.shape[0], 6), np.inf, dtype=np.float32)
    primary_bits = np.full((bits_best.shape[0], 4), np.inf, dtype=np.float32)
    secondary_bits = np.full((bits_best.shape[0], 4), np.inf, dtype=np.float32)
    for code in range(96):
        perm = ((code >> 4) & 0x7) % 6
        primary = ((code >> 2) & 0x3) % 4
        secondary = (code & 0x3) % 4
        vals = bits_best[:, code]
        perm_bits[:, perm] = np.minimum(perm_bits[:, perm], vals)
        primary_bits[:, primary] = np.minimum(primary_bits[:, primary], vals)
        secondary_bits[:, secondary] = np.minimum(secondary_bits[:, secondary], vals)

    interleave_bits = np.stack([bits_none.min(axis=1), bits_inter.min(axis=1)], axis=1)

    targets: dict[str, np.ndarray] = {}
    for head, arr in (
        ("cf_perm", perm_bits),
        ("cf_primary", primary_bits),
        ("cf_secondary", secondary_bits),
        ("interleave", interleave_bits),
    ):
        order = np.argsort(arr, axis=1)
        top2 = order[:, :2].astype(np.int16, copy=False)
        targets[head] = top2
    return targets


def _filterbits_prior_logits(
    raw_numeric: np.ndarray,
    raw_names: list[str],
    indices: np.ndarray,
    *,
    temp: float,
) -> dict[str, np.ndarray] | None:
    """Build heuristic prior logits per head from filterbits features.

    Returns logits (higher is better) for heads: cf_perm/cf_primary/cf_secondary/interleave.
    """
    bits_none_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_none_min_by_filter[", 96)
    bits_inter_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_interleave_min_by_filter[", 96)
    if not bits_none_idx or not bits_inter_idx:
        return None
    if temp <= 0:
        raise ValueError("--filterbits-prior-temp must be positive")

    bits_none = raw_numeric[indices][:, bits_none_idx].astype(np.float32, copy=False)
    bits_inter = raw_numeric[indices][:, bits_inter_idx].astype(np.float32, copy=False)
    bits_best = np.minimum(bits_none, bits_inter)

    perm_bits = np.full((bits_best.shape[0], 6), np.inf, dtype=np.float32)
    primary_bits = np.full((bits_best.shape[0], 4), np.inf, dtype=np.float32)
    secondary_bits = np.full((bits_best.shape[0], 4), np.inf, dtype=np.float32)
    for code in range(96):
        perm = ((code >> 4) & 0x7) % 6
        primary = ((code >> 2) & 0x3) % 4
        secondary = (code & 0x3) % 4
        vals = bits_best[:, code]
        perm_bits[:, perm] = np.minimum(perm_bits[:, perm], vals)
        primary_bits[:, primary] = np.minimum(primary_bits[:, primary], vals)
        secondary_bits[:, secondary] = np.minimum(secondary_bits[:, secondary], vals)

    interleave_bits = np.stack([bits_none.min(axis=1), bits_inter.min(axis=1)], axis=1).astype(np.float32, copy=False)

    def _to_logits(bits: np.ndarray) -> np.ndarray:
        base = bits - bits.min(axis=1, keepdims=True)
        return (-base / float(temp)).astype(np.float32, copy=False)

    return {
        "cf_perm": _to_logits(perm_bits),
        "cf_primary": _to_logits(primary_bits),
        "cf_secondary": _to_logits(secondary_bits),
        "interleave": _to_logits(interleave_bits),
    }


def _energy_prior_logits(
    raw_numeric: np.ndarray,
    raw_names: list[str],
    indices: np.ndarray,
    *,
    temp: float,
) -> dict[str, np.ndarray] | None:
    pred_energy_idx = _find_feature_indices(raw_names, "score_residual_energy_by_predictor[", 8)
    reorder_tv_mean_idx = _find_feature_indices(raw_names, "reorder_tv_mean[", 8)
    if not pred_energy_idx or not reorder_tv_mean_idx:
        return None
    if temp <= 0:
        raise ValueError("--energy-prior-temp must be positive")

    eps = 1e-6
    pred_energy = raw_numeric[indices][:, pred_energy_idx].astype(np.float32, copy=False)
    reorder_tv = raw_numeric[indices][:, reorder_tv_mean_idx].astype(np.float32, copy=False)
    pred_logits = -np.log1p(np.maximum(pred_energy, 0.0) + eps) / float(temp)
    reorder_logits = -np.log1p(np.maximum(reorder_tv, 0.0) + eps) / float(temp)
    return {
        "predictor": pred_logits.astype(np.float32, copy=False),
        "reorder": reorder_logits.astype(np.float32, copy=False),
    }


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


def split_filter_code(filter_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = (filter_code >> 4) & 0x7
    primary = (filter_code >> 2) & 0x3
    secondary = filter_code & 0x3
    return perm, primary, secondary


def build_hard_negatives(
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    hard_filters_none: torch.Tensor,
    hard_filters_interleave: torch.Tensor,
    *,
    neg_k: int,
    topk: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build hard negatives from filterbits-derived top-K filter candidates."""
    labels_best = labels_best.to(torch.int64)
    labels_second = labels_second.to(torch.int64)
    bsz = int(labels_best.shape[0])
    if bsz == 0:
        return torch.empty((0, neg_k, labels_best.shape[1]), device=labels_best.device, dtype=torch.int64)

    topk = max(1, int(topk))
    # Build candidate tuples using top-2 predictors and top-2 reorders from {best, second}.
    pred_cand = torch.stack([labels_best[:, 0], labels_second[:, 0]], dim=1)
    reorder_cand = torch.stack([labels_best[:, 4], labels_second[:, 4]], dim=1)
    pool_size = topk * 2 * 2 * 2  # filters * interleave * pred * reorder
    cand = torch.empty((bsz, pool_size, 6), device=labels_best.device, dtype=torch.int64)

    filters0 = hard_filters_none[:, :topk].to(torch.int64)
    filters1 = hard_filters_interleave[:, :topk].to(torch.int64)
    perm0, prim0, sec0 = split_filter_code(filters0)
    perm1, prim1, sec1 = split_filter_code(filters1)

    # Layout: interleave(0/1) x pred(2) x reorder(2) x filter(topk)
    idx = 0
    for inter in (0, 1):
        perm = perm0 if inter == 0 else perm1
        prim = prim0 if inter == 0 else prim1
        sec = sec0 if inter == 0 else sec1
        for pi in range(2):
            for ri in range(2):
                sl = slice(idx, idx + topk)
                cand[:, sl, 0] = pred_cand[:, pi, None]
                cand[:, sl, 1] = perm
                cand[:, sl, 2] = prim
                cand[:, sl, 3] = sec
                cand[:, sl, 4] = reorder_cand[:, ri, None]
                cand[:, sl, 5] = inter
                idx += topk

    # Avoid sampling current positives as negatives.
    best = labels_best[:, None, :]
    second = labels_second[:, None, :]
    same_best = (cand == best).all(dim=2)
    same_second = (cand == second).all(dim=2)
    valid_mask = ~(same_best | same_second)

    # Sample neg_k indices from valid candidates, with fallback to random.
    if pool_size < neg_k:
        neg_k = pool_size
    idx = torch.arange(pool_size, device=labels_best.device)[None, :].repeat(bsz, 1)
    if generator is None:
        perm_idx = torch.rand_like(idx.float()).argsort(dim=1)
    else:
        perm_idx = torch.rand((bsz, pool_size), device=labels_best.device, generator=generator).argsort(dim=1)
    idx = idx.gather(1, perm_idx)
    valid_mask = valid_mask.gather(1, perm_idx)
    # Keep only valid; if too few, allow invalid to fill remaining slots.
    out = []
    for i in range(bsz):
        valid_i = idx[i][valid_mask[i]]
        if valid_i.numel() < neg_k:
            fill = idx[i][: neg_k - valid_i.numel()]
            pick = torch.cat([valid_i, fill], dim=0)
        else:
            pick = valid_i[:neg_k]
        out.append(cand[i].index_select(0, pick))
    return torch.stack(out, dim=0)


def build_hard_negatives_from_proj(
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    topk_per_head: dict[str, torch.Tensor],
    *,
    neg_k: int,
    generator: torch.Generator | None = None,
    fix_pred_reorder: bool = False,
) -> torch.Tensor:
    """Sample hard negatives by drawing each head from proj-derived top-K lists."""
    labels_best = labels_best.to(torch.int64)
    labels_second = labels_second.to(torch.int64)
    bsz = int(labels_best.shape[0])
    if bsz == 0:
        return torch.empty((0, neg_k, labels_best.shape[1]), device=labels_best.device, dtype=torch.int64)

    neg_k = max(1, int(neg_k))
    device = labels_best.device
    out = torch.empty((bsz, neg_k, labels_best.shape[1]), device=device, dtype=torch.int64)

    gen = generator
    for head_idx, head in enumerate(HEAD_ORDER):
        if fix_pred_reorder and head in ("predictor", "reorder"):
            cand = torch.stack([labels_best[:, head_idx], labels_second[:, head_idx]], dim=1)
            k = 2
            if gen is None:
                pick = torch.randint(0, k, (bsz, neg_k), device=device)
            else:
                pick = torch.randint(0, k, (bsz, neg_k), device=device, generator=gen)
            out[:, :, head_idx] = cand.gather(1, pick)
        else:
            topk = topk_per_head[head]
            k = int(topk.shape[1])
            if gen is None:
                pick = torch.randint(0, k, (bsz, neg_k), device=device)
            else:
                pick = torch.randint(0, k, (bsz, neg_k), device=device, generator=gen)
            out[:, :, head_idx] = topk.gather(1, pick)

    # Avoid sampling current positives as negatives: if any match, perturb predictor id.
    best = labels_best[:, None, :]
    second = labels_second[:, None, :]
    same_best = (out == best).all(dim=2)
    same_second = (out == second).all(dim=2)
    conflict = same_best | same_second
    if conflict.any():
        if not fix_pred_reorder:
            # Flip predictor class id to a different value deterministically.
            classes = int(HEADS["predictor"])
            pred = out[:, :, 0]
            pred = (pred + 1) % classes
            out[:, :, 0] = pred
    return out


def tuple_ranking_loss(
    logits: dict[str, torch.Tensor],
    labels_best: torch.Tensor,
    labels_second: torch.Tensor,
    *,
    neg_k: int,
    x: torch.Tensor | None = None,
    pairwise: PairwiseInteractions | None = None,
    neg_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz = int(labels_best.shape[0])
    s_best = tuple_score_from_logits(logits, labels_best)
    s_second = tuple_score_from_logits(logits, labels_second)
    if neg_labels is None:
        neg_labels = sample_negative_tuples_from_batch(labels_best, labels_second, neg_k=neg_k)
    s_neg = tuple_scores_for_neg_tuples(logits, neg_labels)
    if pairwise is not None:
        if x is None:
            raise ValueError("pairwise requires x")
        s_best = s_best + pairwise.score_pos(x, labels_best)
        s_second = s_second + pairwise.score_pos(x, labels_second)
        s_neg = s_neg + pairwise.score_neg(x, neg_labels)
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
    x: torch.Tensor | None = None,
    pairwise: PairwiseInteractions | None = None,
    neg_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    labels_best = labels_best.to(torch.int64)
    labels_second = labels_second.to(torch.int64)
    if neg_k <= 0:
        raise ValueError("--neg-k must be positive")

    log_probs = {head: torch.log_softmax(head_logits, dim=1) for head, head_logits in logits.items()}
    s_best = tuple_score_from_logits(log_probs, labels_best)
    s_second = tuple_score_from_logits(log_probs, labels_second)
    if neg_labels is None:
        neg_labels = sample_negative_tuples_from_batch(labels_best, labels_second, neg_k=neg_k, generator=generator)
    s_neg = tuple_scores_for_neg_tuples(log_probs, neg_labels)
    if pairwise is not None:
        if x is None:
            raise ValueError("pairwise requires x")
        s_best = s_best + pairwise.score_pos(x, labels_best)
        s_second = s_second + pairwise.score_pos(x, labels_second)
        s_neg = s_neg + pairwise.score_neg(x, neg_labels)

    scores = torch.cat([s_best[:, None], s_second[:, None], s_neg], dim=1)
    logz = torch.logsumexp(scores, dim=1, keepdim=True)
    logp = scores - logz
    return -(0.5 * logp[:, 0] + 0.5 * logp[:, 1]).mean()


def nce_loss_from_scores(s_best: torch.Tensor, s_second: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    scores = torch.cat([s_best[:, None], s_second[:, None], s_neg], dim=1)
    logz = torch.logsumexp(scores, dim=1, keepdim=True)
    logp = scores - logz
    return -(0.5 * logp[:, 0] + 0.5 * logp[:, 1]).mean()


def train_all_heads(
    device: torch.device,
    models: dict[str, nn.Module],
    pairwise: PairwiseInteractions | None,
    x_train: torch.Tensor,
    labels_best_train: torch.Tensor,
    labels_second_train: torch.Tensor,
    x_valid: torch.Tensor,
    labels_best_valid: torch.Tensor,
    labels_second_valid: torch.Tensor,
    *,
    prior_logits_train: dict[str, torch.Tensor] | None = None,
    prior_logits_valid: dict[str, torch.Tensor] | None = None,
    prior_weight: float = 0.0,
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
    hard_neg_filters_none: torch.Tensor | None = None,
    hard_neg_filters_interleave: torch.Tensor | None = None,
    hard_neg_topk: int = 0,
    hard_neg_proj_topk: dict[str, torch.Tensor] | None = None,
    hard_neg_proj_fix_pr: bool = False,
    eval_topn: list[int],
    trial_id: int,
) -> tuple[float, int, dict[int, float]]:
    params: list[torch.Tensor] = []
    for m in models.values():
        params.extend(list(m.parameters()))
    if pairwise is not None:
        params.extend(list(pairwise.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    best_hit_rate = -1.0
    best_hit_epoch = 0
    best_hit_rates_at: dict[int, float] = {n: 0.0 for n in eval_topn}
    best_hit_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_pairwise_state: dict[str, torch.Tensor] | None = None

    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    train_slices = _batch_slices(n_train, batch_size)
    valid_slices = _batch_slices(n_valid, batch_size)

    pbar = tqdm(total=max_epochs, desc=f"trial {trial_id:04d} joint", leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        for m in models.values():
            m.train()
        if pairwise is not None:
            pairwise.train()
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
            if prior_logits_train is not None and abs(float(prior_weight)) > 0.0:
                for head, prior_all in prior_logits_train.items():
                    logits[head] = logits[head] + float(prior_weight) * prior_all.index_select(0, idx)
            neg_labels = None
            if hard_neg_proj_topk is not None:
                neg_labels = build_hard_negatives_from_proj(
                    best_b,
                    second_b,
                    hard_neg_proj_topk,
                    neg_k=neg_k,
                    generator=gen,
                    fix_pred_reorder=hard_neg_proj_fix_pr,
                )
            if (
                hard_neg_filters_none is not None
                and hard_neg_filters_interleave is not None
                and hard_neg_topk > 0
            ):
                neg_labels = build_hard_negatives(
                    best_b,
                    second_b,
                    hard_neg_filters_none.index_select(0, idx),
                    hard_neg_filters_interleave.index_select(0, idx),
                    neg_k=neg_k,
                    topk=hard_neg_topk,
                    generator=gen,
                )
            if loss_kind in ("tuple_nce", "tuple_nce_proj"):
                loss = tuple_nce_loss(
                    logits,
                    best_b,
                    second_b,
                    neg_k=neg_k,
                    generator=gen,
                    x=xb,
                    pairwise=pairwise,
                    neg_labels=neg_labels,
                )
            else:
                if neg_labels is None:
                    neg_labels = sample_negative_tuples_from_batch(best_b, second_b, neg_k=neg_k, generator=gen)
                loss = tuple_ranking_loss(
                    logits,
                    best_b,
                    second_b,
                    neg_k=neg_k,
                    x=xb,
                    pairwise=pairwise,
                    neg_labels=neg_labels,
                )

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
        if pairwise is not None:
            pairwise.eval()
        gen_v = torch.Generator(device=device)
        gen_v.manual_seed(0xBEE7 + int(trial_id) * 1000 + int(epoch))
        valid_losses: list[float] = []
        with torch.no_grad():
            for s, e in valid_slices:
                xb = x_valid[s:e]
                best_b = labels_best_valid[s:e]
                second_b = labels_second_valid[s:e]
                logits = {name: model(xb) for name, model in models.items()}
                if prior_logits_valid is not None and abs(float(prior_weight)) > 0.0:
                    for head, prior_all in prior_logits_valid.items():
                        logits[head] = logits[head] + float(prior_weight) * prior_all[s:e]
                if loss_kind in ("tuple_nce", "tuple_nce_proj"):
                    loss = tuple_nce_loss(
                        logits, best_b, second_b, neg_k=neg_k, generator=gen_v, x=xb, pairwise=pairwise
                    )
                else:
                    neg_labels = sample_negative_tuples_from_batch(best_b, second_b, neg_k=neg_k, generator=gen_v)
                    loss = tuple_ranking_loss(
                        logits,
                        best_b,
                        second_b,
                        neg_k=neg_k,
                        x=xb,
                        pairwise=pairwise,
                        neg_labels=neg_labels,
                    )
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
            pairwise,
            x_valid,
            labels_best_valid.detach().cpu().numpy(),
            labels_second_valid.detach().cpu().numpy(),
            batch_size,
            topn=eval_topn,
            score_mode="log_softmax",
            prior_logits=prior_logits_valid,
            prior_weight=float(prior_weight),
            trial_id=trial_id,
        )
        valid_hit_rate = float(valid_hit_rates_at.get(3, 0.0))

        if valid_hit_rate >= best_hit_rate + 1e-9:
            best_hit_rate = valid_hit_rate
            best_hit_epoch = epoch
            best_hit_rates_at = valid_hit_rates_at
            best_hit_state = {k: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()} for k, m in models.items()}
            best_pairwise_state = None if pairwise is None else {n: v.detach().cpu().clone() for n, v in pairwise.state_dict().items()}

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
    if pairwise is not None and best_pairwise_state is not None:
        pairwise.load_state_dict(best_pairwise_state)
    return best_valid_loss, best_hit_epoch, best_hit_rates_at


def train_reranker(
    device: torch.device,
    reranker: TupleReranker,
    x_train: torch.Tensor,
    labels_best_train: torch.Tensor,
    labels_second_train: torch.Tensor,
    x_valid: torch.Tensor,
    labels_best_valid: torch.Tensor,
    labels_second_valid: torch.Tensor,
    *,
    neg_k: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    models: dict[str, nn.Module],
    pairwise: PairwiseInteractions | None,
    eval_topn: list[int],
    prior_logits_train: dict[str, torch.Tensor] | None,
    prior_logits_valid: dict[str, torch.Tensor] | None,
    prior_weight: float,
    trial_id: int,
    beam_widths: dict[str, int],
    beam_candidate_chunk: int,
) -> tuple[float, int, dict[int, float]]:
    optimizer = torch.optim.Adam(reranker.parameters(), lr=lr)
    best_valid_loss = float("inf")
    epochs_no_improve = 0

    best_hit_rate = -1.0
    best_hit_epoch = 0
    best_hit_rates_at: dict[int, float] = {n: 0.0 for n in eval_topn}
    best_state: dict[str, torch.Tensor] | None = None

    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    train_slices = _batch_slices(n_train, batch_size)
    valid_slices = _batch_slices(n_valid, batch_size)

    pbar = tqdm(total=max_epochs, desc=f"trial {trial_id:04d} reranker", leave=False, dynamic_ncols=True)
    for epoch in range(max_epochs):
        reranker.train()
        order = torch.randperm(n_train, device=device, dtype=torch.int64)
        for s, e in train_slices:
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            best_b = labels_best_train.index_select(0, idx)
            second_b = labels_second_train.index_select(0, idx)
            optimizer.zero_grad()
            # Optimize reranker on the full beam candidate set without materializing [B, K, 6] all at once.
            with torch.no_grad():
                head_logits = {name: model(xb) for name, model in models.items()}
                if prior_logits_train is not None and abs(float(prior_weight)) > 0.0:
                    for head, prior_all in prior_logits_train.items():
                        head_logits[head] = head_logits[head] + float(prior_weight) * prior_all.index_select(0, idx)
                head_logits = {name: torch.log_softmax(v, dim=1) for name, v in head_logits.items()}
                top_ids: dict[str, torch.Tensor] = {}
                for head in HEAD_ORDER:
                    _sc, ids = _topk_with_tiebreak_torch(head_logits[head], int(beam_widths[head]))
                    top_ids[head] = ids

            bsz = int(xb.shape[0])
            found_best = torch.zeros((bsz,), device=device, dtype=torch.bool)
            found_second = torch.zeros((bsz,), device=device, dtype=torch.bool)
            pos_best = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
            pos_second = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
            m = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
            ssum = torch.zeros((bsz,), device=device, dtype=torch.float32)
            row = torch.arange(bsz, device=device)
            for labels_chunk, _ in _iter_beam_candidate_chunks(
                top_ids, None, beam_widths, chunk_size=int(beam_candidate_chunk)
            ):
                scores_chunk = reranker.score_many(xb, labels_chunk).to(torch.float32)
                chunk_max = scores_chunk.max(dim=1).values
                new_m = torch.maximum(m, chunk_max)
                ssum = ssum * torch.exp(m - new_m) + torch.exp(scores_chunk - new_m[:, None]).sum(dim=1)
                m = new_m

                eq_best = (labels_chunk == best_b[:, None, :]).all(dim=2)
                hit_best = eq_best.any(dim=1) & ~found_best
                if hit_best.any():
                    idx_best = eq_best.to(torch.int64).argmax(dim=1)
                    gathered = scores_chunk[row, idx_best]
                    pos_best = torch.where(hit_best, gathered, pos_best)
                    found_best = found_best | hit_best

                eq_second = (labels_chunk == second_b[:, None, :]).all(dim=2)
                hit_second = eq_second.any(dim=1) & ~found_second
                if hit_second.any():
                    idx_second = eq_second.to(torch.int64).argmax(dim=1)
                    gathered = scores_chunk[row, idx_second]
                    pos_second = torch.where(hit_second, gathered, pos_second)
                    found_second = found_second | hit_second

            logz = m + torch.log(ssum + 1e-12)
            any_pos = found_best | found_second
            loss_vec = torch.zeros((bsz,), device=device, dtype=torch.float32)
            both = found_best & found_second
            only_best = found_best & ~found_second
            only_second = found_second & ~found_best
            loss_vec = torch.where(only_best, -(pos_best - logz), loss_vec)
            loss_vec = torch.where(only_second, -(pos_second - logz), loss_vec)
            loss_vec = torch.where(both, -0.5 * ((pos_best - logz) + (pos_second - logz)), loss_vec)
            loss = loss_vec[any_pos].mean() if any_pos.any() else (logz.sum() * 0.0)
            loss.backward()
            optimizer.step()

        reranker.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for s, e in valid_slices:
                xb = x_valid[s:e]
                best_b = labels_best_valid[s:e]
                second_b = labels_second_valid[s:e]
                head_logits = {name: model(xb) for name, model in models.items()}
                if prior_logits_valid is not None and abs(float(prior_weight)) > 0.0:
                    for head, prior_all in prior_logits_valid.items():
                        head_logits[head] = head_logits[head] + float(prior_weight) * prior_all[s:e]
                head_logits = {name: torch.log_softmax(v, dim=1) for name, v in head_logits.items()}
                top_ids: dict[str, torch.Tensor] = {}
                for head in HEAD_ORDER:
                    _sc, ids = _topk_with_tiebreak_torch(head_logits[head], int(beam_widths[head]))
                    top_ids[head] = ids

                bsz = int(xb.shape[0])
                found_best = torch.zeros((bsz,), device=device, dtype=torch.bool)
                found_second = torch.zeros((bsz,), device=device, dtype=torch.bool)
                pos_best = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
                pos_second = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
                m = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
                ssum = torch.zeros((bsz,), device=device, dtype=torch.float32)
                row = torch.arange(bsz, device=device)
                for labels_chunk, _ in _iter_beam_candidate_chunks(
                    top_ids, None, beam_widths, chunk_size=int(beam_candidate_chunk)
                ):
                    scores_chunk = reranker.score_many(xb, labels_chunk).to(torch.float32)
                    chunk_max = scores_chunk.max(dim=1).values
                    new_m = torch.maximum(m, chunk_max)
                    ssum = ssum * torch.exp(m - new_m) + torch.exp(scores_chunk - new_m[:, None]).sum(dim=1)
                    m = new_m

                    eq_best = (labels_chunk == best_b[:, None, :]).all(dim=2)
                    hit_best = eq_best.any(dim=1) & ~found_best
                    if hit_best.any():
                        idx_best = eq_best.to(torch.int64).argmax(dim=1)
                        gathered = scores_chunk[row, idx_best]
                        pos_best = torch.where(hit_best, gathered, pos_best)
                        found_best = found_best | hit_best

                    eq_second = (labels_chunk == second_b[:, None, :]).all(dim=2)
                    hit_second = eq_second.any(dim=1) & ~found_second
                    if hit_second.any():
                        idx_second = eq_second.to(torch.int64).argmax(dim=1)
                        gathered = scores_chunk[row, idx_second]
                        pos_second = torch.where(hit_second, gathered, pos_second)
                        found_second = found_second | hit_second

                logz = m + torch.log(ssum + 1e-12)
                any_pos = found_best | found_second
                if any_pos.any():
                    loss_vec = torch.zeros((bsz,), device=device, dtype=torch.float32)
                    both = found_best & found_second
                    only_best = found_best & ~found_second
                    only_second = found_second & ~found_best
                    loss_vec = torch.where(only_best, -(pos_best - logz), loss_vec)
                    loss_vec = torch.where(only_second, -(pos_second - logz), loss_vec)
                    loss_vec = torch.where(both, -0.5 * ((pos_best - logz) + (pos_second - logz)), loss_vec)
                    valid_losses.append(float(loss_vec[any_pos].mean().item()))
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")

        valid_hit_rates_at = evaluate_hit_rates_at(
            device,
            models,
            pairwise,
            x_valid,
            labels_best_valid.detach().cpu().numpy(),
            labels_second_valid.detach().cpu().numpy(),
            batch_size,
            topn=eval_topn,
            score_mode="log_softmax",
            prior_logits=prior_logits_valid,
            prior_weight=float(prior_weight),
            trial_id=trial_id,
            reranker=reranker,
            beam_widths=beam_widths,
            beam_candidate_chunk=int(beam_candidate_chunk),
        )
        valid_hit_rate = float(valid_hit_rates_at.get(3, 0.0))

        if valid_hit_rate >= best_hit_rate + 1e-9:
            best_hit_rate = valid_hit_rate
            best_hit_epoch = epoch
            best_hit_rates_at = valid_hit_rates_at
            best_state = {k: v.detach().cpu().clone() for k, v in reranker.state_dict().items()}

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

    if best_state is None:
        raise RuntimeError("failed to capture best reranker state")
    reranker.load_state_dict(best_state)
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
    prior_train: torch.Tensor | None = None,
    prior_valid: torch.Tensor | None = None,
    prior_weight: float = 0.0,
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
            if prior_train is not None and abs(float(prior_weight)) > 0.0:
                logits = logits + float(prior_weight) * prior_train.index_select(0, idx)
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
                if prior_valid is not None and abs(float(prior_weight)) > 0.0:
                    logits = logits + float(prior_weight) * prior_valid[s:e]
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
    prior_train: torch.Tensor | None = None,
    prior_valid: torch.Tensor | None = None,
    prior_weight: float = 0.0,
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
            if prior_train is not None and abs(float(prior_weight)) > 0.0:
                pred = pred + float(prior_weight) * prior_train.index_select(0, idx)
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
                if prior_valid is not None and abs(float(prior_weight)) > 0.0:
                    pred = pred + float(prior_weight) * prior_valid[s:e]
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
    prior_train: torch.Tensor | None = None,
    prior_valid: torch.Tensor | None = None,
    prior_weight: float = 0.0,
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
            if prior_train is not None and abs(float(prior_weight)) > 0.0:
                logits = logits + float(prior_weight) * prior_train.index_select(0, idx)
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
                if prior_valid is not None and abs(float(prior_weight)) > 0.0:
                    logits = logits + float(prior_weight) * prior_valid[s:e]
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


def _parse_eval_topn(spec: str, *, max_n: int) -> list[int]:
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
    if topn[-1] > int(max_n):
        raise ValueError(f"--eval-topn max N must be <= {int(max_n)}, got {topn[-1]}")
    return topn


def _iter_beam_candidate_chunks(
    top_ids: dict[str, torch.Tensor],
    top_scores: dict[str, torch.Tensor] | None,
    beam_widths: dict[str, int],
    *,
    chunk_size: int,
):
    device = next(iter(top_ids.values())).device
    widths = [int(beam_widths[h]) for h in HEAD_ORDER]
    if any(w <= 0 for w in widths):
        raise ValueError("beam widths must be positive")
    idx_ranges = [torch.arange(w, device=device) for w in widths]
    grid = torch.meshgrid(*idx_ranges, indexing="ij")
    flat = [g.reshape(-1) for g in grid]
    total_k = int(flat[0].shape[0])
    if chunk_size <= 0:
        chunk_size = total_k

    for s in range(0, total_k, int(chunk_size)):
        e = min(s + int(chunk_size), total_k)
        labels_parts: list[torch.Tensor] = []
        score_parts: list[torch.Tensor] = []
        for hi, head in enumerate(HEAD_ORDER):
            idx = flat[hi][s:e].to(torch.int64)
            idx2 = idx.unsqueeze(0).expand(top_ids[head].shape[0], -1)
            ids_h = top_ids[head].gather(1, idx2)
            labels_parts.append(ids_h)
            if top_scores is not None:
                score_parts.append(top_scores[head].gather(1, idx2))
        labels_chunk = torch.stack(labels_parts, dim=2)  # [B, Kc, 6]
        if top_scores is None:
            yield labels_chunk, None
        else:
            head_sum = score_parts[0]
            for sc in score_parts[1:]:
                head_sum = head_sum + sc
            yield labels_chunk, head_sum


def _streaming_topk_update(
    best_scores: torch.Tensor,
    best_ranks: torch.Tensor,
    best_labels: torch.Tensor,
    cand_scores: torch.Tensor,
    cand_ranks: torch.Tensor,
    cand_labels: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-6
    max_rank_all = float(8 * 96 * 8 * 2 - 1)
    cand_adj = cand_scores + (max_rank_all - cand_ranks.to(cand_scores.dtype)) * eps
    best_adj = best_scores + (max_rank_all - best_ranks.to(best_scores.dtype)) * eps
    merged_adj = torch.cat([best_adj, cand_adj], dim=1)
    merged_scores = torch.cat([best_scores, cand_scores], dim=1)
    merged_ranks = torch.cat([best_ranks, cand_ranks], dim=1)
    merged_labels = torch.cat([best_labels, cand_labels], dim=1)
    _, pos = torch.topk(merged_adj, k, dim=1)
    row = torch.arange(merged_adj.shape[0], device=merged_adj.device).unsqueeze(1)
    new_scores = merged_scores[row, pos]
    new_ranks = merged_ranks[row, pos]
    new_labels = merged_labels[row, pos]
    return new_scores, new_ranks, new_labels


def evaluate_hit_rates_at(
    device: torch.device,
    models: dict[str, nn.Module],
    pairwise: PairwiseInteractions | None,
    x_valid: torch.Tensor,
    labels_best_valid: np.ndarray,
    labels_second_valid: np.ndarray,
    batch_size: int,
    *,
    topn: list[int],
    score_mode: str = "log_softmax",
    prior_logits: dict[str, torch.Tensor] | None = None,
    prior_weight: float = 0.0,
    trial_id: int | None = None,
    reranker: TupleReranker | None = None,
    beam_widths: dict[str, int] | None = None,
    beam_candidate_chunk: int = 2048,
) -> dict[int, float]:
    total = int(x_valid.shape[0])
    if total <= 0:
        return {n: 0.0 for n in topn}

    for head in models.values():
        head.eval()
    if pairwise is not None:
        pairwise.eval()
    if reranker is not None:
        reranker.eval()

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
            logits = {name: model(xb) for name, model in models.items()}
            if prior_logits is not None and abs(float(prior_weight)) > 0.0:
                for head, prior_all in prior_logits.items():
                    logits[head] = logits[head] + float(prior_weight) * prior_all[s:e]
            if score_mode != "raw":
                logits = {name: torch.log_softmax(v, dim=1) for name, v in logits.items()}

            beam = dict(DEFAULT_BEAM_WIDTHS if beam_widths is None else beam_widths)
            top_ids: dict[str, torch.Tensor] = {}
            top_scores: dict[str, torch.Tensor] = {}
            for head in HEAD_ORDER:
                sc, ids = _topk_with_tiebreak_torch(logits[head], beam[head])
                top_ids[head] = ids
                top_scores[head] = sc

            bsz = int(xb.shape[0])
            best_scores = torch.full((bsz, max_n), -1e30, device=device, dtype=torch.float32)
            best_ranks = torch.full((bsz, max_n), (8 * 96 * 8 * 2 - 1), device=device, dtype=torch.int64)
            best_labels = torch.zeros((bsz, max_n, 6), device=device, dtype=torch.int64)
            for labels_chunk, head_sum_chunk in _iter_beam_candidate_chunks(
                top_ids,
                top_scores,
                beam,
                chunk_size=int(beam_candidate_chunk),
            ):
                scores_chunk = head_sum_chunk.to(torch.float32)
                if pairwise is not None:
                    scores_chunk = scores_chunk + pairwise.score_neg(xb, labels_chunk).to(torch.float32)
                if reranker is not None:
                    scores_chunk = reranker.score_many(xb, labels_chunk).to(torch.float32)
                pred = labels_chunk[:, :, 0].to(torch.int64)
                perm = labels_chunk[:, :, 1].to(torch.int64)
                prim = labels_chunk[:, :, 2].to(torch.int64)
                sec = labels_chunk[:, :, 3].to(torch.int64)
                reorder = labels_chunk[:, :, 4].to(torch.int64)
                inter = labels_chunk[:, :, 5].to(torch.int64)
                filt = (perm << 4) | (prim << 2) | sec
                rank_chunk = (((pred * 96 + filt) * 8 + reorder) * 2 + inter).to(torch.int64)
                best_scores, best_ranks, best_labels = _streaming_topk_update(
                    best_scores,
                    best_ranks,
                    best_labels,
                    scores_chunk,
                    rank_chunk,
                    labels_chunk,
                    k=max_n,
                )

            top_rank = best_ranks
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
    beam_widths = _parse_beam_widths(args.beam_widths, beam_full=bool(args.beam_full))
    max_candidates = _candidate_count(beam_widths)
    eval_topn = _parse_eval_topn(args.eval_topn, max_n=max_candidates)
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 64)
        args.max_trials = 1 if args.max_trials is None else min(args.max_trials, 1)
        args.pretrain_filterbits_epochs = min(args.pretrain_filterbits_epochs, 1)
        args.hard_neg_proj_topk = min(args.hard_neg_proj_topk, 2)
        args.hard_neg_proj_margin = min(args.hard_neg_proj_margin, 0.5)
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
        "per_head_target": str(args.per_head_target),
        "target_temp": args.target_temp,
        "proj_alpha": args.proj_alpha,
        "proj_temp": args.proj_temp,
        "hard_neg_filterbits": bool(args.hard_neg_filterbits),
        "hard_neg_topk": int(args.hard_neg_topk),
        "hard_neg_proj": bool(args.hard_neg_proj),
        "hard_neg_proj_topk": int(args.hard_neg_proj_topk),
        "hard_neg_proj_near": bool(args.hard_neg_proj_near),
        "hard_neg_proj_margin": float(args.hard_neg_proj_margin),
        "hard_neg_proj_fix_pr": bool(args.hard_neg_proj_fix_pr),
        "pretrain_filterbits_epochs": int(args.pretrain_filterbits_epochs),
        "pretrain_filterbits_lr": float(args.pretrain_filterbits_lr),
        "pairwise_dim": int(args.pairwise_dim),
        "pairwise_scale": float(args.pairwise_scale),
        "filterbits_prior": bool(args.filterbits_prior),
        "filterbits_prior_weight": float(args.filterbits_prior_weight),
        "filterbits_prior_temp": float(args.filterbits_prior_temp),
        "energy_prior": bool(args.energy_prior),
        "energy_prior_weight": float(args.energy_prior_weight),
        "energy_prior_temp": float(args.energy_prior_temp),
        "reranker_embed_dim": int(args.reranker_embed_dim),
        "reranker_hidden": int(args.reranker_hidden),
        "reranker_epochs": int(args.reranker_epochs),
        "reranker_lr": float(args.reranker_lr),
        "beam_widths": beam_widths,
        "beam_candidate_chunk": int(args.beam_candidate_chunk),
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
        if "per_head_target" not in existing:
            existing["per_head_target"] = str(args.per_head_target)
        if "target_temp" not in existing:
            existing["target_temp"] = args.target_temp
        if "proj_alpha" not in existing:
            existing["proj_alpha"] = args.proj_alpha
        if "proj_temp" not in existing:
            existing["proj_temp"] = args.proj_temp
        if "hard_neg_filterbits" not in existing:
            existing["hard_neg_filterbits"] = bool(args.hard_neg_filterbits)
        if "hard_neg_topk" not in existing:
            existing["hard_neg_topk"] = int(args.hard_neg_topk)
        if "hard_neg_proj" not in existing:
            existing["hard_neg_proj"] = bool(args.hard_neg_proj)
        if "hard_neg_proj_topk" not in existing:
            existing["hard_neg_proj_topk"] = int(args.hard_neg_proj_topk)
        if "hard_neg_proj_near" not in existing:
            existing["hard_neg_proj_near"] = bool(args.hard_neg_proj_near)
        if "hard_neg_proj_margin" not in existing:
            existing["hard_neg_proj_margin"] = float(args.hard_neg_proj_margin)
        if "hard_neg_proj_fix_pr" not in existing:
            existing["hard_neg_proj_fix_pr"] = bool(args.hard_neg_proj_fix_pr)
        if "pretrain_filterbits_epochs" not in existing:
            existing["pretrain_filterbits_epochs"] = int(args.pretrain_filterbits_epochs)
        if "pretrain_filterbits_lr" not in existing:
            existing["pretrain_filterbits_lr"] = float(args.pretrain_filterbits_lr)
        if "pairwise_dim" not in existing:
            existing["pairwise_dim"] = int(args.pairwise_dim)
        if "pairwise_scale" not in existing:
            existing["pairwise_scale"] = float(args.pairwise_scale)
        if "filterbits_prior" not in existing:
            existing["filterbits_prior"] = bool(args.filterbits_prior)
        if "filterbits_prior_weight" not in existing:
            existing["filterbits_prior_weight"] = float(args.filterbits_prior_weight)
        if "filterbits_prior_temp" not in existing:
            existing["filterbits_prior_temp"] = float(args.filterbits_prior_temp)
        if "energy_prior" not in existing:
            existing["energy_prior"] = bool(args.energy_prior)
        if "energy_prior_weight" not in existing:
            existing["energy_prior_weight"] = float(args.energy_prior_weight)
        if "energy_prior_temp" not in existing:
            existing["energy_prior_temp"] = float(args.energy_prior_temp)
        if "reranker_embed_dim" not in existing:
            existing["reranker_embed_dim"] = int(args.reranker_embed_dim)
        if "reranker_hidden" not in existing:
            existing["reranker_hidden"] = int(args.reranker_hidden)
        if "reranker_epochs" not in existing:
            existing["reranker_epochs"] = int(args.reranker_epochs)
        if "reranker_lr" not in existing:
            existing["reranker_lr"] = float(args.reranker_lr)
        if "beam_widths" not in existing:
            existing["beam_widths"] = beam_widths
        if "beam_candidate_chunk" not in existing:
            existing["beam_candidate_chunk"] = int(args.beam_candidate_chunk)
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
        per_head_target = str(args.per_head_target)
        if per_head_target == "auto":
            per_head_target = "proj_top2" if dataset.proj_top2 is not None else "best_second_bits"
        if per_head_target == "proj_top2":
            if dataset.proj_top2 is None:
                raise SystemExit("--per-head-target=proj_top2 requires dataset.proj_top2 (rebuild dataset)")
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
            "per_head_target": str(args.per_head_target),
            "score_mode": "raw" if args.loss == "per_head_proj" else "log_softmax",
            "proj_alpha": float(args.proj_alpha),
            "proj_temp": float(args.proj_temp),
            "hard_neg_filterbits": bool(args.hard_neg_filterbits),
            "hard_neg_topk": int(args.hard_neg_topk),
            "hard_neg_proj": bool(args.hard_neg_proj),
            "hard_neg_proj_topk": int(args.hard_neg_proj_topk),
            "hard_neg_proj_near": bool(args.hard_neg_proj_near),
            "hard_neg_proj_margin": float(args.hard_neg_proj_margin),
            "hard_neg_proj_fix_pr": bool(args.hard_neg_proj_fix_pr),
            "pretrain_filterbits_epochs": int(args.pretrain_filterbits_epochs),
            "pairwise_dim": int(args.pairwise_dim),
            "pairwise_scale": float(args.pairwise_scale),
            "pairwise_pairs": [list(p) for p in PAIRWISE_DEFAULT_PAIRS],
            "filterbits_prior": bool(args.filterbits_prior),
            "filterbits_prior_weight": float(args.filterbits_prior_weight),
            "filterbits_prior_temp": float(args.filterbits_prior_temp),
            "energy_prior": bool(args.energy_prior),
            "energy_prior_weight": float(args.energy_prior_weight),
            "energy_prior_temp": float(args.energy_prior_temp),
            "reranker_embed_dim": int(args.reranker_embed_dim),
            "reranker_hidden": int(args.reranker_hidden),
            "reranker_epochs": int(args.reranker_epochs),
            "beam_widths": beam_widths,
            "beam_candidate_chunk": int(args.beam_candidate_chunk),
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

            pairwise_model: PairwiseInteractions | None = None
            if args.pairwise_dim > 0:
                if not args.loss.startswith("tuple"):
                    raise SystemExit("--pairwise-dim requires a tuple_* loss")
                pairwise_model = PairwiseInteractions(
                    input_dim,
                    int(args.pairwise_dim),
                    pairs=list(PAIRWISE_DEFAULT_PAIRS),
                    scale=float(args.pairwise_scale),
                ).to(device)
                head_params["pairwise"] = count_params(pairwise_model)

            prior_train_t: dict[str, torch.Tensor] | None = None
            prior_valid_t: dict[str, torch.Tensor] | None = None
            # Combine multiple priors into per-head logits (already weighted), then apply with prior_weight=1.0.
            prior_weight = 0.0
            if args.filterbits_prior or args.energy_prior:
                prior_weight = 1.0
                prior_train_np: dict[str, np.ndarray] = {}
                prior_valid_np: dict[str, np.ndarray] = {}

                if args.filterbits_prior:
                    fb_train = _filterbits_prior_logits(
                        dataset.raw_numeric,
                        dataset.raw_names,
                        train_idx_sorted,
                        temp=float(args.filterbits_prior_temp),
                    )
                    fb_valid = _filterbits_prior_logits(
                        dataset.raw_numeric,
                        dataset.raw_names,
                        valid_idx_sorted,
                        temp=float(args.filterbits_prior_temp),
                    )
                    if fb_train is None or fb_valid is None:
                        raise SystemExit("--filterbits-prior requested but required filterbits features are missing")
                    w = float(args.filterbits_prior_weight)
                    for head, arr in fb_train.items():
                        prior_train_np[head] = prior_train_np.get(head, 0.0) + w * arr
                    for head, arr in fb_valid.items():
                        prior_valid_np[head] = prior_valid_np.get(head, 0.0) + w * arr

                if args.energy_prior:
                    en_train = _energy_prior_logits(
                        dataset.raw_numeric,
                        dataset.raw_names,
                        train_idx_sorted,
                        temp=float(args.energy_prior_temp),
                    )
                    en_valid = _energy_prior_logits(
                        dataset.raw_numeric,
                        dataset.raw_names,
                        valid_idx_sorted,
                        temp=float(args.energy_prior_temp),
                    )
                    if en_train is None or en_valid is None:
                        raise SystemExit("--energy-prior requested but required energy/TV features are missing")
                    w = float(args.energy_prior_weight)
                    for head, arr in en_train.items():
                        prior_train_np[head] = prior_train_np.get(head, 0.0) + w * arr
                    for head, arr in en_valid.items():
                        prior_valid_np[head] = prior_valid_np.get(head, 0.0) + w * arr

                prior_train_t = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device) for k, v in prior_train_np.items()}
                prior_valid_t = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device) for k, v in prior_valid_np.items()}

            reranker_model: TupleReranker | None = None

            if args.loss == "per_head_ce":
                for head_idx, head in enumerate(HEAD_ORDER):
                    head_t0 = time.perf_counter()
                    model = models[head]
                    if target_source == "proj_top2":
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
                            prior_train=prior_train_t.get(head) if prior_train_t is not None else None,
                            prior_valid=prior_valid_t.get(head) if prior_valid_t is not None else None,
                            prior_weight=prior_weight,
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
                            prior_train=prior_train_t.get(head) if prior_train_t is not None else None,
                            prior_valid=prior_valid_t.get(head) if prior_valid_t is not None else None,
                            prior_weight=prior_weight,
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

                if args.reranker_embed_dim > 0:
                    rerank_t0 = time.perf_counter()
                    labels_best_train = torch.from_numpy(dataset.labels_best[train_idx_sorted].astype(np.int64)).to(device)
                    labels_second_train = torch.from_numpy(dataset.labels_second[train_idx_sorted].astype(np.int64)).to(device)
                    labels_best_valid = torch.from_numpy(dataset.labels_best[valid_idx_sorted].astype(np.int64)).to(device)
                    labels_second_valid = torch.from_numpy(dataset.labels_second[valid_idx_sorted].astype(np.int64)).to(device)
                    reranker_model = TupleReranker(input_dim, int(args.reranker_embed_dim), int(args.reranker_hidden)).to(device)
                    head_params["reranker"] = count_params(reranker_model)
                    rerank_loss, rerank_epoch, rerank_hit_rates_at = train_reranker(
                        device,
                        reranker_model,
                        x_train,
                        labels_best_train,
                        labels_second_train,
                        x_valid,
                        labels_best_valid,
                        labels_second_valid,
                        neg_k=args.neg_k,
                        batch_size=args.batch_size,
                        max_epochs=int(args.reranker_epochs),
                        patience=min(2, int(args.reranker_epochs)),
                        lr=float(args.reranker_lr),
                        models=models,
                        pairwise=None,
                        eval_topn=eval_topn,
                        prior_logits_train=prior_train_t,
                        prior_logits_valid=prior_valid_t,
                        prior_weight=prior_weight,
                        trial_id=trial_id,
                        beam_widths=beam_widths,
                        beam_candidate_chunk=int(args.beam_candidate_chunk),
                    )
                    torch.cuda.synchronize(device)
                    timings["train_reranker_sec"] = float(time.perf_counter() - rerank_t0)
                    head_losses["reranker"] = float(rerank_loss)
                    head_epochs["reranker"] = int(rerank_epoch)
                    valid_hit_rates_at = rerank_hit_rates_at
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
                        prior_train=prior_train_t.get(head) if prior_train_t is not None else None,
                        prior_valid=prior_valid_t.get(head) if prior_valid_t is not None else None,
                        prior_weight=prior_weight,
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

                if args.reranker_embed_dim > 0:
                    rerank_t0 = time.perf_counter()
                    labels_best_train = torch.from_numpy(dataset.labels_best[train_idx_sorted].astype(np.int64)).to(device)
                    labels_second_train = torch.from_numpy(dataset.labels_second[train_idx_sorted].astype(np.int64)).to(device)
                    labels_best_valid = torch.from_numpy(dataset.labels_best[valid_idx_sorted].astype(np.int64)).to(device)
                    labels_second_valid = torch.from_numpy(dataset.labels_second[valid_idx_sorted].astype(np.int64)).to(device)
                    reranker_model = TupleReranker(input_dim, int(args.reranker_embed_dim), int(args.reranker_hidden)).to(device)
                    head_params["reranker"] = count_params(reranker_model)
                    rerank_loss, rerank_epoch, rerank_hit_rates_at = train_reranker(
                        device,
                        reranker_model,
                        x_train,
                        labels_best_train,
                        labels_second_train,
                        x_valid,
                        labels_best_valid,
                        labels_second_valid,
                        neg_k=args.neg_k,
                        batch_size=args.batch_size,
                        max_epochs=int(args.reranker_epochs),
                        patience=min(2, int(args.reranker_epochs)),
                        lr=float(args.reranker_lr),
                        models=models,
                        pairwise=None,
                        eval_topn=eval_topn,
                        prior_logits_train=prior_train_t,
                        prior_logits_valid=prior_valid_t,
                        prior_weight=prior_weight,
                        trial_id=trial_id,
                        beam_widths=beam_widths,
                        beam_candidate_chunk=int(args.beam_candidate_chunk),
                    )
                    torch.cuda.synchronize(device)
                    timings["train_reranker_sec"] = float(time.perf_counter() - rerank_t0)
                    head_losses["reranker"] = float(rerank_loss)
                    head_epochs["reranker"] = int(rerank_epoch)
                    valid_hit_rates_at = rerank_hit_rates_at
            else:
                # Optional filterbits-based pretrain to initialize filter-related heads.
                if args.pretrain_filterbits_epochs > 0:
                    heuristic_targets_train = _filterbits_heuristic_targets(
                        dataset.raw_numeric, dataset.raw_names, train_idx_sorted
                    )
                    heuristic_targets_valid = _filterbits_heuristic_targets(
                        dataset.raw_numeric, dataset.raw_names, valid_idx_sorted
                    )
                    if heuristic_targets_train and heuristic_targets_valid:
                        for head_idx, head in enumerate(["cf_perm", "cf_primary", "cf_secondary", "interleave"]):
                            if head not in models:
                                continue
                            head_t0 = time.perf_counter()
                            top2_train = torch.from_numpy(heuristic_targets_train[head]).to(device)
                            top2_valid = torch.from_numpy(heuristic_targets_valid[head]).to(device)
                            _loss, _epoch = train_head_top2(
                                head,
                                models[head],
                                device,
                                x_train,
                                top2_train,
                                x_valid,
                                top2_valid,
                                prior_train=prior_train_t.get(head) if prior_train_t is not None else None,
                                prior_valid=prior_valid_t.get(head) if prior_valid_t is not None else None,
                                prior_weight=prior_weight,
                                batch_size=args.batch_size,
                                trial_id=trial_id,
                                head_index=head_idx,
                                total_heads=len(HEAD_ORDER),
                                max_epochs=args.pretrain_filterbits_epochs,
                                patience=max(1, args.pretrain_filterbits_epochs),
                                lr=args.pretrain_filterbits_lr,
                            )
                            torch.cuda.synchronize(device)
                            timings[f"pretrain_{head}_sec"] = float(time.perf_counter() - head_t0)
                train_t0 = time.perf_counter()
                labels_best_train = torch.from_numpy(dataset.labels_best[train_idx_sorted].astype(np.int64)).to(device)
                labels_second_train = torch.from_numpy(dataset.labels_second[train_idx_sorted].astype(np.int64)).to(device)
                labels_best_valid = torch.from_numpy(dataset.labels_best[valid_idx_sorted].astype(np.int64)).to(device)
                labels_second_valid = torch.from_numpy(dataset.labels_second[valid_idx_sorted].astype(np.int64)).to(device)
                proj_scores_train = None
                proj_scores_valid = None
                hard_filters_none = None
                hard_filters_interleave = None
                hard_proj_topk = None
                if args.hard_neg_filterbits:
                    bits_none_idx = _find_feature_indices(
                        dataset.raw_names, "score_bits_plain_hilbert_none_min_by_filter[", 96
                    )
                    bits_inter_idx = _find_feature_indices(
                        dataset.raw_names, "score_bits_plain_hilbert_interleave_min_by_filter[", 96
                    )
                    if bits_none_idx and bits_inter_idx:
                        bits_none = dataset.raw_numeric[:, bits_none_idx]
                        bits_inter = dataset.raw_numeric[:, bits_inter_idx]
                        hard_filters_none = _topk_smallest(bits_none, args.hard_neg_topk)
                        hard_filters_interleave = _topk_smallest(bits_inter, args.hard_neg_topk)
                if args.hard_neg_proj:
                    if dataset.proj_scores is None:
                        raise SystemExit("--hard-neg-proj requires dataset.proj_scores")
                    proj_topk: dict[str, np.ndarray] = {}
                    for head in HEAD_ORDER:
                        classes = int(HEADS[head])
                        off = int(PROJ_OFFSETS[head])
                        scores = dataset.proj_scores[:, off : off + classes]
                        if args.hard_neg_proj_near:
                            proj_topk[head] = _topk_near_best(scores, args.hard_neg_proj_topk, args.hard_neg_proj_margin)
                        else:
                            proj_topk[head] = _topk_largest(scores, args.hard_neg_proj_topk)
                    hard_proj_topk = proj_topk
                if args.loss == "tuple_nce_proj":
                    assert dataset.proj_scores is not None
                    proj_scores_train = torch.from_numpy(dataset.proj_scores[train_idx_sorted].astype(np.float32)).to(device)
                    proj_scores_valid = torch.from_numpy(dataset.proj_scores[valid_idx_sorted].astype(np.float32)).to(device)
                hard_filters_none_t = None
                hard_filters_interleave_t = None
                if hard_filters_none is not None and hard_filters_interleave is not None:
                    hard_filters_none_t = torch.from_numpy(hard_filters_none[train_idx_sorted].astype(np.int64)).to(device)
                    hard_filters_interleave_t = torch.from_numpy(
                        hard_filters_interleave[train_idx_sorted].astype(np.int64)
                    ).to(device)
                hard_proj_topk_t = None
                if hard_proj_topk is not None:
                    hard_proj_topk_t = {
                        head: torch.from_numpy(hard_proj_topk[head][train_idx_sorted].astype(np.int64)).to(device)
                        for head in HEAD_ORDER
                    }
                valid_loss, best_epoch, valid_hit_rates_at = train_all_heads(
                    device,
                    models,
                    pairwise_model,
                    x_train,
                    labels_best_train,
                    labels_second_train,
                    x_valid,
                    labels_best_valid,
                    labels_second_valid,
                    prior_logits_train=prior_train_t,
                    prior_logits_valid=prior_valid_t,
                    prior_weight=prior_weight,
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
                    hard_neg_filters_none=hard_filters_none_t,
                    hard_neg_filters_interleave=hard_filters_interleave_t,
                    hard_neg_topk=int(args.hard_neg_topk) if args.hard_neg_filterbits else 0,
                    hard_neg_proj_topk=hard_proj_topk_t,
                    hard_neg_proj_fix_pr=bool(args.hard_neg_proj_fix_pr),
                    eval_topn=eval_topn,
                    trial_id=trial_id,
                )
                torch.cuda.synchronize(device)
                timings["train_joint_sec"] = float(time.perf_counter() - train_t0)
                head_losses = {"joint": float(valid_loss)}
                head_epochs = {"joint": int(best_epoch)}

                if args.reranker_embed_dim > 0:
                    rerank_t0 = time.perf_counter()
                    reranker_model = TupleReranker(input_dim, int(args.reranker_embed_dim), int(args.reranker_hidden)).to(device)
                    head_params["reranker"] = count_params(reranker_model)
                    rerank_loss, rerank_epoch, rerank_hit_rates_at = train_reranker(
                        device,
                        reranker_model,
                        x_train,
                        labels_best_train,
                        labels_second_train,
                        x_valid,
                        labels_best_valid,
                        labels_second_valid,
                        neg_k=args.neg_k,
                        batch_size=args.batch_size,
                        max_epochs=int(args.reranker_epochs),
                        patience=min(2, int(args.reranker_epochs)),
                        lr=float(args.reranker_lr),
                        models=models,
                        pairwise=pairwise_model,
                        eval_topn=eval_topn,
                        prior_logits_train=prior_train_t,
                        prior_logits_valid=prior_valid_t,
                        prior_weight=prior_weight,
                        trial_id=trial_id,
                        beam_widths=beam_widths,
                        beam_candidate_chunk=int(args.beam_candidate_chunk),
                    )
                    torch.cuda.synchronize(device)
                    timings["train_reranker_sec"] = float(time.perf_counter() - rerank_t0)
                    head_losses["reranker"] = float(rerank_loss)
                    head_epochs["reranker"] = int(rerank_epoch)
                    valid_hit_rates_at = rerank_hit_rates_at

            eval_t0 = time.perf_counter()
            valid_hit_rates_at = evaluate_hit_rates_at(
                device,
                models,
                pairwise_model,
                x_valid,
                dataset.labels_best[valid_idx_sorted],
                dataset.labels_second[valid_idx_sorted],
                args.batch_size,
                topn=eval_topn,
                score_mode=model_bundle["score_mode"],
                prior_logits=prior_valid_t,
                prior_weight=prior_weight,
                trial_id=trial_id,
                reranker=reranker_model,
                beam_widths=beam_widths,
                beam_candidate_chunk=int(args.beam_candidate_chunk),
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
            if pairwise_model is not None:
                pairwise_path = trial_dir / "pairwise.pt"
                torch.save(pairwise_model.state_dict(), pairwise_path)
                model_bundle["pairwise"] = {
                    "path": str(pairwise_path),
                    "dim": int(args.pairwise_dim),
                    "pairs": [list(p) for p in pairwise_model.pairs],
                    "scale": float(args.pairwise_scale),
                }
            if reranker_model is not None:
                reranker_path = trial_dir / "reranker.pt"
                torch.save(reranker_model.state_dict(), reranker_path)
                model_bundle["reranker"] = {
                    "path": str(reranker_path),
                    "embed_dim": int(args.reranker_embed_dim),
                    "hidden": int(args.reranker_hidden),
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
                "per_head_target": str(args.per_head_target),
                "target_temp": args.target_temp,
                "pairwise_dim": int(args.pairwise_dim),
                "pairwise_scale": float(args.pairwise_scale),
                "filterbits_prior": bool(args.filterbits_prior),
                "filterbits_prior_weight": float(args.filterbits_prior_weight),
                "filterbits_prior_temp": float(args.filterbits_prior_temp),
                "energy_prior": bool(args.energy_prior),
                "energy_prior_weight": float(args.energy_prior_weight),
                "energy_prior_temp": float(args.energy_prior_temp),
                "reranker_embed_dim": int(args.reranker_embed_dim),
                "reranker_hidden": int(args.reranker_hidden),
                "reranker_epochs": int(args.reranker_epochs),
                "reranker_lr": float(args.reranker_lr),
                "beam_widths": beam_widths,
                "beam_candidate_chunk": int(args.beam_candidate_chunk),
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
