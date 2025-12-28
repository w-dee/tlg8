#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset_builder import build_dataset
from tlg_ml_utils import RAW_RGB_DIMS, ensure_dir, join_filter, now_iso, save_json, topk_with_tiebreak


HEAD_ORDER = ["predictor", "filter_i", "reorder"]
HEADS = {"predictor": 8, "filter_i": 192, "reorder": 8}


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


@dataclass
class FeatureSpec:
    name: str
    raw_indices: list[int]
    include_raw: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train triad rankers: predictor + (filter,interleave) + reorder")
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-blocks", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="512,256", help="Comma-separated hidden widths")
    parser.add_argument("--beam", type=str, default="predictor=4,filter_i=8,reorder=4")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--filterbits-temp", type=float, default=20.0)
    parser.add_argument("--energy-temp", type=float, default=1.0)
    parser.add_argument("--target-temp", type=float, default=10.0, help="Temperature for best/second bits soft targets")
    parser.add_argument("--eval-topn", type=str, default="1,3,10")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ML tasks, but torch.cuda.is_available() is False")
    return torch.device("cuda")


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        out.append(int(part))
    return out


def _parse_beam(spec: str) -> dict[str, int]:
    out = {"predictor": 4, "filter_i": 8, "reorder": 4}
    if not spec.strip():
        return out
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"unknown beam head: {k}")
        out[k] = max(1, min(int(v), int(HEADS[k])))
    return out


def _parse_eval_topn(spec: str) -> list[int]:
    out = sorted(set(int(p.strip()) for p in spec.split(",") if p.strip()))
    if not out:
        raise ValueError("--eval-topn must not be empty")
    return out


def _find_feature_indices(raw_names: list[str], prefix: str, expected: int) -> list[int]:
    idx = [i for i, name in enumerate(raw_names) if name.startswith(prefix)]
    if expected and len(idx) != expected:
        return []
    return idx


def _filteri_prior(raw_numeric: np.ndarray, raw_names: list[str], idx: np.ndarray, *, temp: float) -> np.ndarray:
    # Prefer predictor-budgeted variants if present (cheap predictor↔filter interaction hint).
    bits0_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_none_min_by_filter_top2pred[", 96)
    bits1_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_interleave_min_by_filter_top2pred[", 96)
    if not bits0_idx or not bits1_idx:
        bits0_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_none_min_by_filter[", 96)
        bits1_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_interleave_min_by_filter[", 96)
    if not bits0_idx or not bits1_idx:
        raise SystemExit("missing filterbits arrays for filter_i prior")
    if temp <= 0:
        raise SystemExit("--filterbits-temp must be positive")
    bits0 = raw_numeric[idx][:, bits0_idx].astype(np.float32, copy=False)
    bits1 = raw_numeric[idx][:, bits1_idx].astype(np.float32, copy=False)
    bits = np.concatenate([bits0, bits1], axis=1)  # [N,192]
    base = bits - bits.min(axis=1, keepdims=True)
    return (-base / float(temp)).astype(np.float32, copy=False)


def _energy_prior(raw_numeric: np.ndarray, raw_names: list[str], idx: np.ndarray, *, temp: float) -> tuple[np.ndarray, np.ndarray]:
    pred_energy_idx = _find_feature_indices(raw_names, "score_residual_energy_by_predictor[", 8)
    reorder_tv_mean_idx = _find_feature_indices(raw_names, "reorder_tv_mean[", 8)
    if not pred_energy_idx or not reorder_tv_mean_idx:
        raise SystemExit("missing energy/TV arrays for predictor/reorder prior")
    if temp <= 0:
        raise SystemExit("--energy-temp must be positive")
    eps = 1e-6
    pe = raw_numeric[idx][:, pred_energy_idx].astype(np.float32, copy=False)
    tv = raw_numeric[idx][:, reorder_tv_mean_idx].astype(np.float32, copy=False)
    pred = -np.log1p(np.maximum(pe, 0.0) + eps) / float(temp)
    reorder = -np.log1p(np.maximum(tv, 0.0) + eps) / float(temp)
    return pred.astype(np.float32, copy=False), reorder.astype(np.float32, copy=False)


def _targets_bits_weighted(
    labels_best: np.ndarray, labels_second: np.ndarray, bits: np.ndarray, *, target_temp: float
) -> dict[str, np.ndarray]:
    # bits: [N,2] uint64 best/second bits
    n = int(labels_best.shape[0])
    out: dict[str, np.ndarray] = {}
    bits_best = bits[:, 0].astype(np.float64)
    bits_second = bits[:, 1].astype(np.float64)
    if target_temp <= 0:
        raise ValueError("--target-temp must be positive")
    # Soft preference for the better (smaller bits) tuple.
    s_best = np.exp(-bits_best / float(target_temp))
    s_second = np.exp(-bits_second / float(target_temp))
    denom = np.maximum(1e-12, s_best + s_second)
    w_best = s_best / denom
    w_second = s_second / denom

    best_filter = (labels_best[:, 1].astype(np.int64) << 4) | (labels_best[:, 2].astype(np.int64) << 2) | labels_best[:, 3].astype(np.int64)
    second_filter = (labels_second[:, 1].astype(np.int64) << 4) | (labels_second[:, 2].astype(np.int64) << 2) | labels_second[:, 3].astype(np.int64)
    best_filter_i = best_filter + 96 * labels_best[:, 5].astype(np.int64)
    second_filter_i = second_filter + 96 * labels_second[:, 5].astype(np.int64)

    mapping = {
        "predictor": (labels_best[:, 0].astype(np.int64), labels_second[:, 0].astype(np.int64), 8),
        "filter_i": (best_filter_i, second_filter_i, 192),
        "reorder": (labels_best[:, 4].astype(np.int64), labels_second[:, 4].astype(np.int64), 8),
    }
    for head, (b, s, c) in mapping.items():
        t = np.zeros((n, c), dtype=np.float32)
        same = b == s
        t[np.arange(n), b] += w_best.astype(np.float32)
        t[np.arange(n), s] += w_second.astype(np.float32)
        # If same, renormalize to 1.
        t[same, :] = 0.0
        t[same, b[same]] = 1.0
        out[head] = t
    return out


def _batch_slices(n: int, bs: int) -> list[tuple[int, int]]:
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]


def _train_head(
    device: torch.device,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    prior_train: torch.Tensor | None,
    prior_valid: torch.Tensor | None,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[float, int]:
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_imp = 0
    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=device)
        for s, e in _batch_slices(n_train, batch_size):
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            opt.zero_grad()
            logits = model(xb)
            if prior_train is not None:
                logits = logits + prior_train.index_select(0, idx)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(yb * logp).sum(dim=1).mean()
            loss.backward()
            opt.step()
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for s, e in _batch_slices(n_valid, batch_size):
                xb = x_valid[s:e]
                yb = y_valid[s:e]
                logits = model(xb)
                if prior_valid is not None:
                    logits = logits + prior_valid[s:e]
                logp = torch.log_softmax(logits, dim=1)
                losses.append(float((-(yb * logp).sum(dim=1).mean()).item()))
        vloss = float(np.mean(losses)) if losses else float("inf")
        if vloss < best - 1e-6:
            best = vloss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, best_epoch


def _topk_np(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    return topk_with_tiebreak(scores, k)


def evaluate_hit_rates(
    logits: dict[str, np.ndarray],
    labels_best: np.ndarray,
    labels_second: np.ndarray,
    *,
    topn: list[int],
    beam: dict[str, int],
) -> dict[int, float]:
    max_n = int(max(topn))
    hits = {n: 0 for n in topn}
    total = int(labels_best.shape[0])
    for i in range(total):
        b = labels_best[i]
        s = labels_second[i]
        bt = (int(b[0]), join_filter(int(b[1]), int(b[2]), int(b[3])), int(b[4]), int(b[5]))
        st = (int(s[0]), join_filter(int(s[1]), int(s[2]), int(s[3])), int(s[4]), int(s[5]))
        p_top = _topk_np(logits["predictor"][i], beam["predictor"])
        f_top = _topk_np(logits["filter_i"][i], beam["filter_i"])
        r_top = _topk_np(logits["reorder"][i], beam["reorder"])
        cand: list[tuple[float, tuple[int, int, int, int]]] = []
        for pid, ps in p_top:
            for fid, fs in f_top:
                filt = int(fid % 96)
                inter = int(fid // 96)
                for rid, rs in r_top:
                    cand.append((float(ps + fs + rs), (int(pid), filt, int(rid), inter)))
        cand.sort(key=lambda x: (-x[0], x[1]))
        top = [t for _, t in cand[:max_n]]
        for n_top in topn:
            if bt in top[:n_top] or st in top[:n_top]:
                hits[n_top] += 1
    return {k: float(hits[k]) / float(total) for k in topn}


def main() -> None:
    args = parse_args()
    device = ensure_cuda()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    run_dir = args.run_root / args.run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "dataset")
    ensure_dir(run_dir / "splits")

    if args.smoke:
        args.max_blocks = min(int(args.max_blocks), 50_000)
        args.max_epochs = min(int(args.max_epochs), 2)
        args.patience = min(int(args.patience), 1)
        args.batch_size = min(int(args.batch_size), 256)

    dataset = build_dataset(run_dir, args.in_dir, int(args.seed), rebuild=False, max_blocks=int(args.max_blocks))

    # Feature spec: reuse the same raw-index builder logic as train_rankers (filter budget).
    from train_rankers import apply_feature_pipeline, build_feature_specs, standardize_features  # type: ignore

    specs = build_feature_specs(dataset)
    spec = next((s for s in specs if s.name == "filter_budget_scores_raw"), specs[0])

    train_idx = dataset.train_indices
    valid_idx = dataset.valid_indices
    train_idx_sorted = np.sort(train_idx)
    valid_idx_sorted = np.sort(valid_idx)

    non_pixel_train = apply_feature_pipeline(dataset.raw_numeric[train_idx_sorted], spec)
    non_pixel_valid = apply_feature_pipeline(dataset.raw_numeric[valid_idx_sorted], spec)
    non_pixel_train_z, non_pixel_valid_z, non_pixel_mean, non_pixel_std = standardize_features(
        non_pixel_train, non_pixel_valid
    )
    x_train_np = np.concatenate([dataset.pixels[train_idx_sorted], non_pixel_train_z], axis=1).astype(np.float32, copy=False)
    x_valid_np = np.concatenate([dataset.pixels[valid_idx_sorted], non_pixel_valid_z], axis=1).astype(np.float32, copy=False)

    x_train = torch.from_numpy(x_train_np).to(device)
    x_valid = torch.from_numpy(x_valid_np).to(device)

    prior_w = float(args.prior_weight)
    prior_filter_i = _filteri_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=float(args.filterbits_temp)) * prior_w
    prior_filter_i_v = _filteri_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=float(args.filterbits_temp)) * prior_w
    prior_pred, prior_re = _energy_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=float(args.energy_temp))
    prior_pred_v, prior_re_v = _energy_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=float(args.energy_temp))
    prior_pred = prior_pred * prior_w
    prior_re = prior_re * prior_w
    prior_pred_v = prior_pred_v * prior_w
    prior_re_v = prior_re_v * prior_w

    priors_train = {
        "predictor": torch.from_numpy(prior_pred).to(device),
        "filter_i": torch.from_numpy(prior_filter_i).to(device),
        "reorder": torch.from_numpy(prior_re).to(device),
    }
    priors_valid = {
        "predictor": torch.from_numpy(prior_pred_v).to(device),
        "filter_i": torch.from_numpy(prior_filter_i_v).to(device),
        "reorder": torch.from_numpy(prior_re_v).to(device),
    }

    targets = _targets_bits_weighted(
        dataset.labels_best[train_idx_sorted].astype(np.int64),
        dataset.labels_second[train_idx_sorted].astype(np.int64),
        dataset.bits[train_idx_sorted],
        target_temp=float(args.target_temp),
    )
    targets_v = _targets_bits_weighted(
        dataset.labels_best[valid_idx_sorted].astype(np.int64),
        dataset.labels_second[valid_idx_sorted].astype(np.int64),
        dataset.bits[valid_idx_sorted],
        target_temp=float(args.target_temp),
    )

    hidden = _parse_int_list(args.hidden)
    models = {h: MLP(int(x_train.shape[1]), hidden, int(HEADS[h])).to(device) for h in HEAD_ORDER}
    losses: dict[str, float] = {}
    epochs: dict[str, int] = {}
    for head in HEAD_ORDER:
        y_train = torch.from_numpy(targets[head]).to(device)
        y_valid = torch.from_numpy(targets_v[head]).to(device)
        loss, best_ep = _train_head(
            device,
            models[head],
            x_train,
            y_train,
            x_valid,
            y_valid,
            priors_train.get(head),
            priors_valid.get(head),
            lr=float(args.lr),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
        )
        losses[head] = float(loss)
        epochs[head] = int(best_ep)

    beam = _parse_beam(args.beam)
    topn = _parse_eval_topn(args.eval_topn)
    # Evaluate on valid
    with torch.no_grad():
        logits_t = {h: models[h](x_valid) + priors_valid[h] for h in HEAD_ORDER}
        logits_np = {h: torch.log_softmax(v, dim=1).detach().cpu().numpy() for h, v in logits_t.items()}
    hit_rates = evaluate_hit_rates(
        logits_np,
        dataset.labels_best[valid_idx_sorted],
        dataset.labels_second[valid_idx_sorted],
        topn=topn,
        beam=beam,
    )

    trial_dir = run_dir / "artifacts" / "trial_0000"
    ensure_dir(trial_dir)
    for head in HEAD_ORDER:
        torch.save(models[head].state_dict(), trial_dir / f"{head}.pt")

    bundle = {
        "run_id": args.run_id,
        "trial_id": 0,
        "head_order": HEAD_ORDER,
        "heads": {
            h: {"path": str(trial_dir / f"{h}.pt"), "hidden_sizes": hidden, "classes": int(HEADS[h])} for h in HEAD_ORDER
        },
        "input_dim": int(x_train.shape[1]),
        "pixel_dim": RAW_RGB_DIMS,
        "raw_feature_names": dataset.raw_names,
        "feature_spec": {"name": spec.name, "raw_indices": spec.raw_indices, "transforms": [], "include_raw": True},
        "feature_norm": {"non_pixel_mean": non_pixel_mean.tolist(), "non_pixel_std": non_pixel_std.tolist()},
        "score_mode": "log_softmax",
        "prior": {
            "kind": "filteri_energy",
            "weight": float(args.prior_weight),
            "filterbits_temp": float(args.filterbits_temp),
            "energy_temp": float(args.energy_temp),
        },
        "eval_topn": topn,
        "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()},
        "beam": beam,
    }
    save_json(trial_dir / "bundle.json", bundle)
    save_json(trial_dir / "eval.json", {"valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()}})

    # Minimal progress/best tracking.
    entry = {
        "trial_id": 0,
        "timestamp": now_iso(),
        "dataset": {"in_dir": str(args.in_dir), "max_blocks": int(args.max_blocks)},
        "head_losses": losses,
        "head_best_epoch": epochs,
        "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()},
        "bundle_path": str(trial_dir / "bundle.json"),
        "status": "ok",
    }
    (run_dir / "progress.jsonl").open("a", encoding="utf-8").write(json.dumps(entry) + "\n")
    save_json(run_dir / "best.json", {"best_trial_id": 0, "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()}, "bundle_path": str(trial_dir / "bundle.json")})
    config = dict(vars(args))
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    save_json(run_dir / "config.json", config)

    hit_str = " ".join(f"hit@{k}={hit_rates[k]*100:.2f}%" for k in topn)
    print(f"[trial 0000] valid {hit_str}", flush=True)


if __name__ == "__main__":
    main()
