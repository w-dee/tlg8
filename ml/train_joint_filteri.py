#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset_builder import build_dataset
from tlg_ml_utils import RAW_RGB_DIMS, ensure_dir, join_filter, now_iso, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full joint scorer over (predictor, filter_i, reorder)")
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-blocks", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--filterbits-temp", type=float, default=20.0)
    parser.add_argument("--energy-temp", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=50_000)
    parser.add_argument("--eval-topn", type=str, default="1,3,10")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ML tasks, but torch.cuda.is_available() is False")
    return torch.device("cuda")


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
        raise SystemExit("missing filterbits arrays for filter_i prior (expected 96×2 arrays)")
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


class JointScorer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.x_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.embed_dim),
            nn.ReLU(),
        )
        self.emb_pred = nn.Embedding(8, self.embed_dim)
        self.emb_filteri = nn.Embedding(192, self.embed_dim)
        self.emb_reorder = nn.Embedding(8, self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def score_many(self, x: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        x_emb = self.x_net(x)  # [B,D]
        t_emb = self.emb_pred(tuples[:, 0]) + self.emb_filteri(tuples[:, 1]) + self.emb_reorder(tuples[:, 2])  # [T,D]
        x_exp = x_emb[:, None, :].expand(-1, t_emb.shape[0], -1)
        t_exp = t_emb[None, :, :].expand(x_emb.shape[0], -1, -1)
        feat = torch.cat([x_exp, t_exp], dim=2).reshape(-1, self.embed_dim * 2)
        return self.mlp(feat).reshape(x_emb.shape[0], t_emb.shape[0])


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
        args.max_epochs = min(int(args.max_epochs), 1)
        args.batch_size = min(int(args.batch_size), 64)
        args.eval_samples = min(int(args.eval_samples), 5_000)

    dataset = build_dataset(run_dir, args.in_dir, int(args.seed), rebuild=False, max_blocks=int(args.max_blocks))

    # Features: use the same pipeline/normalization as train_rankers for filter budget.
    from train_rankers import apply_feature_pipeline, build_feature_specs, standardize_features  # type: ignore

    specs = build_feature_specs(dataset)
    spec = next((s for s in specs if s.name == "filter_budget_scores_raw"), specs[0])
    train_idx_sorted = np.sort(dataset.train_indices)
    valid_idx_sorted = np.sort(dataset.valid_indices)
    non_pixel_train = apply_feature_pipeline(dataset.raw_numeric[train_idx_sorted], spec)
    non_pixel_valid = apply_feature_pipeline(dataset.raw_numeric[valid_idx_sorted], spec)
    non_pixel_train_z, non_pixel_valid_z, non_pixel_mean, non_pixel_std = standardize_features(
        non_pixel_train, non_pixel_valid
    )
    x_train_np = np.concatenate([dataset.pixels[train_idx_sorted], non_pixel_train_z], axis=1).astype(np.float32, copy=False)
    x_valid_np = np.concatenate([dataset.pixels[valid_idx_sorted], non_pixel_valid_z], axis=1).astype(np.float32, copy=False)
    x_train = torch.from_numpy(x_train_np).to(device)
    x_valid = torch.from_numpy(x_valid_np).to(device)

    # Full tuple list.
    tuples = np.empty((8 * 192 * 8, 3), dtype=np.int64)
    t = 0
    for p in range(8):
        for fi in range(192):
            for r in range(8):
                tuples[t, 0] = p
                tuples[t, 1] = fi
                tuples[t, 2] = r
                t += 1
    tuples_t = torch.from_numpy(tuples).to(device)

    # Priors per head on train/valid.
    prior_w = float(args.prior_weight)
    prior_fi_tr = _filteri_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=float(args.filterbits_temp)) * prior_w
    prior_fi_va = _filteri_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=float(args.filterbits_temp)) * prior_w
    prior_p_tr, prior_r_tr = _energy_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=float(args.energy_temp))
    prior_p_va, prior_r_va = _energy_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=float(args.energy_temp))
    prior_p_tr *= prior_w
    prior_r_tr *= prior_w
    prior_p_va *= prior_w
    prior_r_va *= prior_w

    prior_p_tr_t = torch.from_numpy(prior_p_tr).to(device)
    prior_r_tr_t = torch.from_numpy(prior_r_tr).to(device)
    prior_fi_tr_t = torch.from_numpy(prior_fi_tr).to(device)
    prior_p_va_t = torch.from_numpy(prior_p_va).to(device)
    prior_r_va_t = torch.from_numpy(prior_r_va).to(device)
    prior_fi_va_t = torch.from_numpy(prior_fi_va).to(device)

    # Best/second tuple indices for loss.
    lb = dataset.labels_best[train_idx_sorted].astype(np.int64)
    ls = dataset.labels_second[train_idx_sorted].astype(np.int64)
    b_filter = (lb[:, 1] << 4) | (lb[:, 2] << 2) | lb[:, 3]
    s_filter = (ls[:, 1] << 4) | (ls[:, 2] << 2) | ls[:, 3]
    b_fi = b_filter + 96 * lb[:, 5]
    s_fi = s_filter + 96 * ls[:, 5]
    b_idx = ((lb[:, 0] * 192 + b_fi) * 8 + lb[:, 4]).astype(np.int64)
    s_idx = ((ls[:, 0] * 192 + s_fi) * 8 + ls[:, 4]).astype(np.int64)
    b_idx_t = torch.from_numpy(b_idx).to(device)
    s_idx_t = torch.from_numpy(s_idx).to(device)

    model = JointScorer(int(x_train.shape[1]), int(args.embed_dim), int(args.hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    bs = int(args.batch_size)
    n_train = int(x_train.shape[0])
    for epoch in range(int(args.max_epochs)):
        model.train()
        order = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        loss_n = 0
        for s in range(0, n_train, bs):
            e = min(s + bs, n_train)
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            opt.zero_grad()
            prior = (
                prior_p_tr_t.index_select(0, idx)[:, tuples_t[:, 0]]
                + prior_fi_tr_t.index_select(0, idx)[:, tuples_t[:, 1]]
                + prior_r_tr_t.index_select(0, idx)[:, tuples_t[:, 2]]
            )
            scores = model.score_many(xb, tuples_t) + prior
            logp = torch.log_softmax(scores, dim=1)
            row = torch.arange(idx.shape[0], device=device)
            lp_b = logp[row, b_idx_t.index_select(0, idx)]
            lp_s = logp[row, s_idx_t.index_select(0, idx)]
            loss = -0.5 * (lp_b + lp_s).mean()
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()
            loss_sum += float(loss.item()) * int(idx.shape[0])
            loss_n += int(idx.shape[0])
        mean_loss = loss_sum / max(1, loss_n)
        print(f"[epoch {epoch}] loss={mean_loss:.5f}", flush=True)

    topn = sorted({int(p.strip()) for p in str(args.eval_topn).split(",") if p.strip()})
    if not topn:
        raise SystemExit("--eval-topn must not be empty")
    max_k = max(topn)

    # Evaluate: score tuples for a subset of valid rows, compute hit@K (true top-2 membership).
    model.eval()
    eval_n = min(int(args.eval_samples), int(x_valid.shape[0]))
    hits = {k: 0 for k in topn}
    with torch.no_grad():
        for s in range(0, eval_n, bs):
            e = min(s + bs, eval_n)
            xb = x_valid[s:e]
            prior = (
                prior_p_va_t[s:e][:, tuples_t[:, 0]]
                + prior_fi_va_t[s:e][:, tuples_t[:, 1]]
                + prior_r_va_t[s:e][:, tuples_t[:, 2]]
            )
            scores = model.score_many(xb, tuples_t) + prior
            topk = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu().numpy()

            # decode + compare
            b = dataset.labels_best[valid_idx_sorted[s:e]].astype(np.int64)
            sc = dataset.labels_second[valid_idx_sorted[s:e]].astype(np.int64)
            for i in range(e - s):
                bt = (int(b[i, 0]), join_filter(int(b[i, 1]), int(b[i, 2]), int(b[i, 3])), int(b[i, 4]), int(b[i, 5]))
                st = (int(sc[i, 0]), join_filter(int(sc[i, 1]), int(sc[i, 2]), int(sc[i, 3])), int(sc[i, 4]), int(sc[i, 5]))

                # Build a prefix set progressively to compute hit@k.
                # (max_k is small; keep this simple and deterministic.)
                prefix: list[tuple[int, int, int, int]] = []
                for j in topk[i]:
                    p, fi, r = tuples[j]
                    filt = int(fi % 96)
                    inter = int(fi // 96)
                    prefix.append((int(p), filt, int(r), inter))

                for k in topn:
                    cand = prefix[:k]
                    if bt in cand or st in cand:
                        hits[k] += 1

    hit_rates = {k: float(hits[k]) / float(eval_n) for k in topn}
    hit_str = " ".join(f"hit@{k}={hit_rates[k]*100:.2f}%" for k in topn)
    print(f"[eval] {hit_str} (N={eval_n})", flush=True)

    trial_dir = run_dir / "artifacts" / "trial_0000"
    ensure_dir(trial_dir)
    torch.save(model.state_dict(), trial_dir / "joint.pt")
    bundle = {
        "run_id": args.run_id,
        "trial_id": 0,
        "kind": "joint_filteri",
        "input_dim": int(x_train.shape[1]),
        "pixel_dim": RAW_RGB_DIMS,
        "raw_feature_names": dataset.raw_names,
        "feature_spec": {"name": spec.name, "raw_indices": spec.raw_indices, "transforms": [], "include_raw": True},
        "feature_norm": {"non_pixel_mean": non_pixel_mean.tolist(), "non_pixel_std": non_pixel_std.tolist()},
        "model": {
            "path": str(trial_dir / "joint.pt"),
            "embed_dim": int(args.embed_dim),
            "hidden": int(args.hidden),
        },
        "prior": {
            "weight": float(args.prior_weight),
            "filterbits_temp": float(args.filterbits_temp),
            "energy_temp": float(args.energy_temp),
        },
        "eval_topn": topn,
        "valid_hit_rates_at": {str(k): float(hit_rates[k]) for k in topn},
        "eval_n": eval_n,
    }
    save_json(trial_dir / "bundle.json", bundle)
    entry = {
        "trial_id": 0,
        "timestamp": now_iso(),
        "bundle_path": str(trial_dir / "bundle.json"),
        "valid_hit_rates_at": {str(k): float(hit_rates[k]) for k in topn},
        "eval_n": eval_n,
        "status": "ok",
    }
    (run_dir / "progress.jsonl").open("a", encoding="utf-8").write(json.dumps(entry) + "\n")
    config = dict(vars(args))
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    save_json(run_dir / "config.json", config)
    save_json(
        run_dir / "best.json",
        {"best_trial_id": 0, "valid_hit_rates_at": {str(k): float(hit_rates[k]) for k in topn}, "bundle_path": str(trial_dir / "bundle.json")},
    )


if __name__ == "__main__":
    main()
