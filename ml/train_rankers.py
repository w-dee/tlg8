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


@dataclass
class FeatureSpec:
    name: str
    raw_indices: list[int]
    transforms: list[dict[str, Any]]


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
        fallback = Path("ml/runs/packed_smoke")
        if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
            return fallback
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
    features = [raw_sel]
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

    train_raw = raw_numeric[dataset.train_indices]
    variances = np.var(train_raw, axis=0)
    order = np.argsort(-variances)

    specs: list[FeatureSpec] = [FeatureSpec(name="raw_top0", raw_indices=[], transforms=[])]
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
    return specs


def build_arch_specs() -> list[dict[str, list[int]]]:
    candidates = [
        [],
        [32],
        [64],
        [96],
        [128],
        [64, 32],
        [128, 64],
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


def build_soft_targets(labels_best: np.ndarray, bits_best: np.ndarray) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for head_idx, head in enumerate(HEAD_ORDER):
        classes = HEADS[head]
        proj_scores = np.full((classes,), np.inf, dtype=np.float64)
        for i in range(labels_best.shape[0]):
            v = int(labels_best[i, head_idx])
            proj_scores[v] = min(proj_scores[v], float(bits_best[i]))
        order = np.argsort(proj_scores)
        if np.isinf(proj_scores[order[1]]):
            raise ValueError(f"not enough classes present for head {head}")
        pos = order[:2]
        target = np.zeros((labels_best.shape[0], classes), dtype=np.float32)
        target[:, pos[0]] = 0.5
        target[:, pos[1]] = 0.5
        targets[head] = target
    return targets


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
            logits: dict[str, torch.Tensor] = {name: model(xb) for name, model in models.items()}

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


def load_progress(progress_path: Path) -> tuple[int, float, int]:
    if not progress_path.exists():
        return 0, -1.0, 0
    best_rate = -1.0
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
            if rate >= best_rate + 0.001:
                best_rate = rate
                last_best = rate
                no_improve = 0
            else:
                if last_best >= 0:
                    no_improve += 1
    return max_trial + 1, best_rate, no_improve


def get_trial_config(
    trial_id: int, feature_specs: list[FeatureSpec], arch_specs: list[dict[str, list[int]]], seed: int
) -> tuple[FeatureSpec, dict[str, list[int]]]:
    if trial_id < len(feature_specs) * len(arch_specs):
        feat_idx = trial_id % len(feature_specs)
        arch_idx = (trial_id // len(feature_specs)) % len(arch_specs)
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
    if not args.smoke and max_blocks is None and args.in_dir.resolve() == Path("ml/source/packed_out").resolve():
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
        "eval_topn": eval_topn,
        "run_id": run_id,
        "input_dir": str(args.in_dir),
        "max_blocks": max_blocks,
        "feature_specs": [
            {"name": spec.name, "raw_indices": spec.raw_indices, "transforms": spec.transforms}
            for spec in feature_specs
        ],
        "arch_specs": arch_specs,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if "eval_topn" not in existing:
            existing["eval_topn"] = eval_topn
        if existing != config:
            raise SystemExit("config.json exists and differs; use a new run_id for material changes")
    else:
        save_json(config_path, config)

    start_trial, best_rate, no_improve = load_progress(run_dir / "progress.jsonl")
    max_trials = args.max_trials or len(feature_specs) * len(arch_specs)

    progress_path = run_dir / "progress.jsonl"
    best_path = run_dir / "best.json"

    bits_best = dataset.bits[:, 0]
    targets = build_soft_targets(dataset.labels_best, bits_best)

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

        x_train_np = np.concatenate([dataset.pixels[train_idx_sorted], non_pixel_train], axis=1).astype(
            np.float32, copy=False
        )
        x_valid_np = np.concatenate([dataset.pixels[valid_idx_sorted], non_pixel_valid], axis=1).astype(
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
            for head_idx, head in enumerate(HEAD_ORDER):
                head_t0 = time.perf_counter()
                hidden_sizes = arch_spec[head]
                model = MLP(input_dim, hidden_sizes, HEADS[head]).to(device)
                models[head] = model
                head_params[head] = count_params(model)

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

                model_path = trial_dir / f"{head}.pt"
                torch.save(model.state_dict(), model_path)
                model_bundle["heads"][head] = {
                    "path": str(model_path),
                    "hidden_sizes": hidden_sizes,
                    "classes": HEADS[head],
                }

            eval_t0 = time.perf_counter()
            valid_hit_rates_at = evaluate_hit_rates_at(
                device,
                models,
                x_valid,
                dataset.labels_best[valid_idx_sorted],
                dataset.labels_second[valid_idx_sorted],
                args.batch_size,
                topn=eval_topn,
                trial_id=trial_id,
            )
            valid_hit_rate = float(valid_hit_rates_at.get(3, 0.0))
            torch.cuda.synchronize(device)
            timings["eval_sec"] = float(time.perf_counter() - eval_t0)
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
            if best_path.exists():
                best_prev = json.loads(best_path.read_text(encoding="utf-8"))
                best_params = int(best_prev.get("params", {}).get("total", total_params + 1))
            else:
                best_params = total_params + 1
            if total_params < best_params:
                updated = True

        if updated:
            best_rate = valid_hit_rate
            no_improve = 0
        else:
            no_improve += 1

        best_payload = {
            "status": "ok" if best_rate >= 0.9 else "below_threshold",
            "trial_id": trial_id,
            "valid_hit_rate": best_rate,
            "latest_valid_hit_rate": valid_hit_rate,
            "valid_loss_mean": valid_loss_mean,
            "params": {"per_head": head_params, "total": total_params},
            "bundle_path": str(bundle_path),
            "no_improve": no_improve,
        }
        write_best_state(best_payload)

        if no_improve >= 10:
            failure_payload = {
                "status": "failed_no_improve",
                "trial_id": trial_id,
                "best_valid_hit_rate": best_rate,
                "no_improve": no_improve,
            }
            write_best_state(failure_payload)
            break


def time_run_id() -> str:
    return now_iso().replace(":", "").replace("-", "").replace("T", "_")


if __name__ == "__main__":
    main()
