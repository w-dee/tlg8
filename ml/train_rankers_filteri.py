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
from torch.nn import functional as F

from dataset_builder import build_dataset
from tlg_ml_utils import RAW_RGB_DIMS, ensure_dir, join_filter, now_iso, save_json, topk_with_tiebreak


HEAD_ORDER = ["predictor", "filter_i", "reorder"]
HEADS = {"predictor": 8, "filter_i": 192, "reorder": 8}
TRIAD_TUPLES = int(8 * 192 * 8)


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
    parser.add_argument("--reranker-embed-dim", type=int, default=0)
    parser.add_argument("--reranker-hidden", type=int, default=256)
    parser.add_argument("--reranker-epochs", type=int, default=5)
    parser.add_argument("--reranker-lr", type=float, default=1e-3)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--reranker-candidates", type=int, default=2048)
    parser.add_argument("--reranker-scale", type=float, default=0.1)
    parser.add_argument("--reranker-neg-k", type=int, default=64)
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


def _pred_reorder_bits_prior(
    raw_numeric: np.ndarray, raw_names: list[str], idx: np.ndarray, *, temp: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # Optional: min bits by predictor/reorder, computed by encoder (Plain entropy).
    # If present, use it as an additional logit prior (lower bits => higher logit).
    if temp <= 0:
        raise SystemExit("--filterbits-temp must be positive")

    p0_idx = _find_feature_indices(raw_names, "score_bits_plain_none_min_by_predictor[", 8)
    p1_idx = _find_feature_indices(raw_names, "score_bits_plain_interleave_min_by_predictor[", 8)
    r0_idx = _find_feature_indices(raw_names, "score_bits_plain_none_min_by_reorder[", 8)
    r1_idx = _find_feature_indices(raw_names, "score_bits_plain_interleave_min_by_reorder[", 8)
    if not p0_idx or not p1_idx or not r0_idx or not r1_idx:
        return None, None

    p0 = raw_numeric[idx][:, p0_idx].astype(np.float32, copy=False)
    p1 = raw_numeric[idx][:, p1_idx].astype(np.float32, copy=False)
    r0 = raw_numeric[idx][:, r0_idx].astype(np.float32, copy=False)
    r1 = raw_numeric[idx][:, r1_idx].astype(np.float32, copy=False)

    p = np.minimum(p0, p1)
    r = np.minimum(r0, r1)

    p_base = p - p.min(axis=1, keepdims=True)
    r_base = r - r.min(axis=1, keepdims=True)
    return (-p_base / float(temp)).astype(np.float32, copy=False), (-r_base / float(temp)).astype(np.float32, copy=False)


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


def _triad_tuple_id(pred: np.ndarray, filter_i: np.ndarray, reorder: np.ndarray) -> np.ndarray:
    return ((pred.astype(np.int64) * 192 + filter_i.astype(np.int64)) * 8 + reorder.astype(np.int64)).astype(np.int64)


class TriadReranker(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        if self.embed_dim <= 0:
            raise ValueError("TriadReranker embed_dim must be positive")
        if self.hidden <= 0:
            raise ValueError("TriadReranker hidden must be positive")
        self.x_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.embed_dim),
            nn.ReLU(),
        )
        self.emb_pred = nn.Embedding(8, self.embed_dim)
        self.emb_fi = nn.Embedding(192, self.embed_dim)
        self.emb_re = nn.Embedding(8, self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def score_many(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B,K,3] with columns (pred, filter_i, reorder)
        x_emb = self.x_net(x).to(torch.float32)  # [B,D]
        labels = labels.to(torch.int64)
        t_emb = self.emb_pred(labels[:, :, 0]) + self.emb_fi(labels[:, :, 1]) + self.emb_re(labels[:, :, 2])  # [B,K,D]
        x_exp = x_emb[:, None, :].expand(-1, t_emb.shape[1], -1)
        feat = torch.cat([x_exp, t_emb], dim=2).reshape(-1, self.embed_dim * 2)
        out = self.mlp(feat).reshape(t_emb.shape[0], t_emb.shape[1])
        return out

    def score_pos(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B,3]
        x_emb = self.x_net(x).to(torch.float32)
        labels = labels.to(torch.int64)
        t_emb = self.emb_pred(labels[:, 0]) + self.emb_fi(labels[:, 1]) + self.emb_re(labels[:, 2])
        feat = torch.cat([x_emb, t_emb], dim=1)
        return self.mlp(feat).squeeze(1)


def _stage1_topk_candidates(
    device: torch.device,
    pred_logp: torch.Tensor,
    fi_logp: torch.Tensor,
    re_logp: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    # pred_logp: [B,8], fi_logp: [B,192], re_logp: [B,8] (log-prob; higher is better)
    bsz = int(pred_logp.shape[0])
    k = int(min(k, TRIAD_TUPLES))
    # [B,8,192,8] -> [B,12288]
    scores = pred_logp[:, :, None, None] + fi_logp[:, None, :, None] + re_logp[:, None, None, :]
    flat = scores.reshape(bsz, -1)
    top = torch.topk(flat, k=k, dim=1).indices  # [B,k] of tuple ids in the fixed order
    # Decode tuple ids to (pred, fi, re).
    tid = top.to(torch.int64)
    re = (tid % 8).to(torch.int64)
    tmp = (tid // 8).to(torch.int64)
    fi = (tmp % 192).to(torch.int64)
    pred = (tmp // 192).to(torch.int64)
    out = torch.stack([pred, fi, re], dim=2).to(device)
    return out


def _find_pos_in_candidates(cand: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cand: [B,K,3], b/s: [B,3] -> returns indices in [0..K-1] or -1.
    # K is modest (<=4096); O(B*K) is acceptable.
    bsz, k, _ = cand.shape
    eq_b = (cand == b[:, None, :]).all(dim=2)  # [B,K]
    eq_s = (cand == s[:, None, :]).all(dim=2)  # [B,K]
    has_b = eq_b.any(dim=1)
    has_s = eq_s.any(dim=1)
    idx_b = torch.where(has_b, eq_b.to(torch.int64).argmax(dim=1), torch.full((bsz,), -1, device=cand.device, dtype=torch.int64))
    idx_s = torch.where(has_s, eq_s.to(torch.int64).argmax(dim=1), torch.full((bsz,), -1, device=cand.device, dtype=torch.int64))
    return idx_b, idx_s


def _train_reranker(
    device: torch.device,
    reranker: TriadReranker,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    pred_logp_train: torch.Tensor,
    fi_logp_train: torch.Tensor,
    re_logp_train: torch.Tensor,
    pred_logp_valid: torch.Tensor,
    fi_logp_valid: torch.Tensor,
    re_logp_valid: torch.Tensor,
    best_ids_train: torch.Tensor,
    second_ids_train: torch.Tensor,
    best_ids_valid: torch.Tensor,
    second_ids_valid: torch.Tensor,
    *,
    candidates: int,
    batch_size: int,
    lr: float,
    epochs: int,
    scale: float,
    neg_k: int,
) -> dict[str, Any]:
    opt = torch.optim.Adam(reranker.parameters(), lr=float(lr))
    n_train = int(x_train.shape[0])
    n_valid = int(x_valid.shape[0])
    cand_k = int(min(int(candidates), TRIAD_TUPLES))
    neg_k = int(max(1, min(int(neg_k), cand_k - 1)))

    best_hit32 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}

    for epoch in range(int(epochs)):
        reranker.train()
        order = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        loss_n = 0
        recall_sum = 0.0
        recall_n = 0
        for s, e in _batch_slices(n_train, batch_size):
            idx = order[s:e]
            xb = x_train.index_select(0, idx)
            pb = pred_logp_train.index_select(0, idx)
            fb = fi_logp_train.index_select(0, idx)
            rb = re_logp_train.index_select(0, idx)
            cand = _stage1_topk_candidates(device, pb, fb, rb, k=cand_k)  # [B,K,3]

            b_pos = best_ids_train.index_select(0, idx)
            s_pos = second_ids_train.index_select(0, idx)
            idx_b, idx_s = _find_pos_in_candidates(cand, b_pos, s_pos)
            has_b = idx_b >= 0
            has_s = idx_s >= 0
            any_pos = has_b | has_s
            if any_pos.any():
                bsz = int(xb.shape[0])
                row = torch.arange(bsz, device=device)

                # Sample negatives from the candidate set, excluding positives.
                w = torch.ones((bsz, cand.shape[1]), device=device, dtype=torch.float32)
                w[row[has_b], idx_b[has_b]] = 0.0
                w[row[has_s], idx_s[has_s]] = 0.0
                neg_idx = torch.multinomial(w, num_samples=neg_k, replacement=False)  # [B,neg_k]

                neg_labels = cand.gather(1, neg_idx[:, :, None].expand(-1, -1, 3))  # [B,neg_k,3]

                # Base scores (stage-1) for pos/neg.
                neg_base = pb.gather(1, neg_labels[:, :, 0]) + fb.gather(1, neg_labels[:, :, 1]) + rb.gather(1, neg_labels[:, :, 2])

                # Reranker scores for neg.
                neg_r = reranker.score_many(xb, neg_labels)
                neg_score = neg_base + float(scale) * neg_r

                loss_terms: list[torch.Tensor] = []
                if has_b.any():
                    b_lbl = b_pos
                    b_base = pb.gather(1, b_lbl[:, 0:1]).squeeze(1) + fb.gather(1, b_lbl[:, 1:2]).squeeze(1) + rb.gather(1, b_lbl[:, 2:3]).squeeze(1)
                    b_score = b_base + float(scale) * reranker.score_pos(xb, b_lbl)
                    loss_b = -F.logsigmoid((b_score[:, None] - neg_score)).mean(dim=1)
                    loss_terms.append(torch.where(has_b, loss_b, torch.zeros_like(loss_b)))
                if has_s.any():
                    s_lbl = s_pos
                    s_base = pb.gather(1, s_lbl[:, 0:1]).squeeze(1) + fb.gather(1, s_lbl[:, 1:2]).squeeze(1) + rb.gather(1, s_lbl[:, 2:3]).squeeze(1)
                    s_score = s_base + float(scale) * reranker.score_pos(xb, s_lbl)
                    loss_s = -F.logsigmoid((s_score[:, None] - neg_score)).mean(dim=1)
                    loss_terms.append(torch.where(has_s, loss_s, torch.zeros_like(loss_s)))
                loss_vec = sum(loss_terms) / float(len(loss_terms)) if loss_terms else torch.zeros((bsz,), device=device, dtype=torch.float32)
                loss = loss_vec[any_pos].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += float(loss.item()) * int(any_pos.sum().item())
                loss_n += int(any_pos.sum().item())
            recall_sum += float(any_pos.to(torch.float32).mean().item()) * int(xb.shape[0])
            recall_n += int(xb.shape[0])

        train_loss = loss_sum / max(1, loss_n)
        train_recall = recall_sum / max(1, recall_n)

        # Eval: hit@{3,10,16,32} on full valid split for the reranker output (within candidate set).
        reranker.eval()
        topn = [3, 10, 16, 32]
        hits = {k: 0 for k in topn}
        recall_any = 0
        with torch.no_grad():
            for s, e in _batch_slices(n_valid, batch_size):
                xb = x_valid[s:e]
                pb = pred_logp_valid[s:e]
                fb = fi_logp_valid[s:e]
                rb = re_logp_valid[s:e]
                cand = _stage1_topk_candidates(device, pb, fb, rb, k=cand_k)
                b_pos = best_ids_valid[s:e]
                s_pos = second_ids_valid[s:e]
                idx_b, idx_s = _find_pos_in_candidates(cand, b_pos, s_pos)
                has_b = idx_b >= 0
                has_s = idx_s >= 0
                any_pos = has_b | has_s
                recall_any += int(any_pos.sum().item())

                pred_ids = cand[:, :, 0]
                fi_ids = cand[:, :, 1]
                re_ids = cand[:, :, 2]
                base = pb.gather(1, pred_ids) + fb.gather(1, fi_ids) + rb.gather(1, re_ids)
                scores = base + float(scale) * reranker.score_many(xb, cand)
                max_k = max(topn)
                top = torch.topk(scores, k=min(max_k, cand.shape[1]), dim=1).indices  # [B,max_k]
                cand_cpu = cand.detach().cpu()
                top_cpu = top.detach().cpu()
                b_cpu = b_pos.detach().cpu()
                s_cpu = s_pos.detach().cpu()
                for i in range(int(xb.shape[0])):
                    rows = cand_cpu[i, top_cpu[i], :]  # [max_k,3]
                    for k in topn:
                        r = rows[:k]
                        if bool(((r == b_cpu[i]).all(dim=1)).any() or ((r == s_cpu[i]).all(dim=1)).any()):
                            hits[k] += 1
        eval_hit = {k: float(hits[k]) / float(n_valid) for k in topn}
        eval_recall = float(recall_any) / float(n_valid)

        hit32 = float(eval_hit[32])
        if hit32 > best_hit32 + 1e-9:
            best_hit32 = hit32
            best_metrics = {
                "train_loss": float(train_loss),
                "train_recall_any": float(train_recall),
                "valid_recall_any": float(eval_recall),
                **{f"valid_hit@{k}": float(eval_hit[k]) for k in topn},
            }
            best_state = {n: v.detach().cpu().clone() for n, v in reranker.state_dict().items()}

        hit_str = " ".join(f"hit@{k}={eval_hit[k]*100:.2f}%" for k in topn)
        print(
            f"[reranker epoch {epoch}] train_loss={train_loss:.4f} train_recall={train_recall*100:.2f}% "
            f"valid_recall={eval_recall*100:.2f}% valid {hit_str}",
            flush=True,
        )

    if best_state is not None:
        reranker.load_state_dict(best_state)
    return {
        "best_metrics": best_metrics,
        "best_hit32": float(best_hit32),
        "candidates": cand_k,
        "scale": float(scale),
        "neg_k": int(neg_k),
    }


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

    pK = int(beam["predictor"])
    fK = int(beam["filter_i"])
    rK = int(beam["reorder"])

    for i in range(total):
        b = labels_best[i]
        s = labels_second[i]
        bt = (int(b[0]), join_filter(int(b[1]), int(b[2]), int(b[3])), int(b[4]), int(b[5]))
        st = (int(s[0]), join_filter(int(s[1]), int(s[2]), int(s[3])), int(s[4]), int(s[5]))

        # Stable top-k by class id (deterministic tie-break).
        p_idx = np.argsort(-logits["predictor"][i], kind="stable")[:pK]
        f_idx = np.argsort(-logits["filter_i"][i], kind="stable")[:fK]
        r_idx = np.argsort(-logits["reorder"][i], kind="stable")[:rK]

        p_scores = logits["predictor"][i][p_idx]  # [pK]
        f_scores = logits["filter_i"][i][f_idx]  # [fK]
        r_scores = logits["reorder"][i][r_idx]  # [rK]

        # Scores in deterministic tuple order (pid-major, then fid, then rid).
        score = (p_scores[:, None, None] + f_scores[None, :, None] + r_scores[None, None, :]).astype(np.float32, copy=False)
        flat = score.reshape(-1)

        # Sort by (-score, tuple_id) with tuple_id ascending.
        order = np.lexsort((np.arange(flat.shape[0], dtype=np.int64), -flat))[:max_n]

        top: list[tuple[int, int, int, int]] = []
        for t in order.tolist():
            pid = int(p_idx[t // (fK * rK)])
            rem = int(t % (fK * rK))
            fid = int(f_idx[rem // rK])
            rid = int(r_idx[rem % rK])
            filt = int(fid % 96)
            inter = int(fid // 96)
            top.append((pid, filt, rid, inter))

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
        args.reranker_epochs = min(int(args.reranker_epochs), 1)
        args.reranker_batch_size = min(int(args.reranker_batch_size), 128)
        args.reranker_candidates = min(int(args.reranker_candidates), 512)
        args.reranker_neg_k = min(int(args.reranker_neg_k), 32)

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
    bits_pred, bits_re = _pred_reorder_bits_prior(
        dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=float(args.filterbits_temp)
    )
    bits_pred_v, bits_re_v = _pred_reorder_bits_prior(
        dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=float(args.filterbits_temp)
    )
    prior_pred = prior_pred * prior_w
    prior_re = prior_re * prior_w
    prior_pred_v = prior_pred_v * prior_w
    prior_re_v = prior_re_v * prior_w
    if bits_pred is not None and bits_re is not None and bits_pred_v is not None and bits_re_v is not None:
        prior_pred = prior_pred + bits_pred * prior_w
        prior_re = prior_re + bits_re * prior_w
        prior_pred_v = prior_pred_v + bits_pred_v * prior_w
        prior_re_v = prior_re_v + bits_re_v * prior_w

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

    rerank_metrics: dict[str, Any] | None = None
    reranker: TriadReranker | None = None
    if int(args.reranker_embed_dim) > 0:
        # Stage-1 log-prob tables used to generate candidates.
        with torch.no_grad():
            logp_train = {h: torch.log_softmax(models[h](x_train) + priors_train[h], dim=1) for h in HEAD_ORDER}
            logp_valid = {h: torch.log_softmax(models[h](x_valid) + priors_valid[h], dim=1) for h in HEAD_ORDER}

        # Best/second labels for triad (predictor, filter_i, reorder).
        def _labels_to_triad(lb: np.ndarray, ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lb = lb.astype(np.int64, copy=False)
            ls = ls.astype(np.int64, copy=False)
            b_filter = (lb[:, 1] << 4) | (lb[:, 2] << 2) | lb[:, 3]
            s_filter = (ls[:, 1] << 4) | (ls[:, 2] << 2) | ls[:, 3]
            b_fi = b_filter + 96 * lb[:, 5]
            s_fi = s_filter + 96 * ls[:, 5]
            b = np.stack([lb[:, 0], b_fi, lb[:, 4]], axis=1)
            s = np.stack([ls[:, 0], s_fi, ls[:, 4]], axis=1)
            return b, s

        b_tr_np, s_tr_np = _labels_to_triad(dataset.labels_best[train_idx_sorted], dataset.labels_second[train_idx_sorted])
        b_va_np, s_va_np = _labels_to_triad(dataset.labels_best[valid_idx_sorted], dataset.labels_second[valid_idx_sorted])
        b_tr = torch.from_numpy(b_tr_np).to(device)
        s_tr = torch.from_numpy(s_tr_np).to(device)
        b_va = torch.from_numpy(b_va_np).to(device)
        s_va = torch.from_numpy(s_va_np).to(device)

        reranker = TriadReranker(int(x_train.shape[1]), int(args.reranker_embed_dim), int(args.reranker_hidden)).to(device)
        rerank_metrics = _train_reranker(
            device,
            reranker,
            x_train,
            x_valid,
            logp_train["predictor"],
            logp_train["filter_i"],
            logp_train["reorder"],
            logp_valid["predictor"],
            logp_valid["filter_i"],
            logp_valid["reorder"],
            b_tr,
            s_tr,
            b_va,
            s_va,
            candidates=int(args.reranker_candidates),
            batch_size=int(args.reranker_batch_size),
            lr=float(args.reranker_lr),
            epochs=int(args.reranker_epochs),
            scale=float(args.reranker_scale),
            neg_k=int(args.reranker_neg_k),
        )

    trial_dir = run_dir / "artifacts" / "trial_0000"
    ensure_dir(trial_dir)
    for head in HEAD_ORDER:
        torch.save(models[head].state_dict(), trial_dir / f"{head}.pt")
    if reranker is not None:
        torch.save(reranker.state_dict(), trial_dir / "reranker.pt")

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
        "reranker": (
            None
            if reranker is None
            else {
            "path": str(trial_dir / "reranker.pt"),
            "embed_dim": int(args.reranker_embed_dim),
            "hidden": int(args.reranker_hidden),
            "epochs": int(args.reranker_epochs),
            "lr": float(args.reranker_lr),
            "batch_size": int(args.reranker_batch_size),
            "candidates": int(args.reranker_candidates),
            "scale": float(args.reranker_scale),
            "neg_k": int(args.reranker_neg_k),
            "best_metrics": None if rerank_metrics is None else rerank_metrics.get("best_metrics", {}),
            }
        ),
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
        "reranker": None if rerank_metrics is None else rerank_metrics.get("best_metrics", {}),
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
