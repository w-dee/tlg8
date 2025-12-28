#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tlg_ml_utils import HEADS, apply_transform, ensure_dir, extract_pixels_rgb, join_filter, topk_with_tiebreak

HEAD_ORDER = ["predictor", "cf_perm", "cf_primary", "cf_secondary", "reorder", "interleave"]
DEFAULT_BEAM_WIDTHS = {
    "predictor": 4,
    "cf_perm": 4,
    "cf_primary": 3,
    "cf_secondary": 2,
    "reorder": 4,
    "interleave": 2,
}

PAIRWISE_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("predictor", "cf_perm"),
    ("predictor", "cf_primary"),
    ("predictor", "cf_secondary"),
    ("reorder", "cf_perm"),
    ("reorder", "cf_primary"),
    ("reorder", "cf_secondary"),
]


def _pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def _topk_with_tiebreak_torch(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k <= 0:
        empty = torch.empty((scores.shape[0], 0), device=scores.device, dtype=scores.dtype)
        empty_i = torch.empty((scores.shape[0], 0), device=scores.device, dtype=torch.int64)
        return empty, empty_i
    classes = int(scores.shape[1])
    ids = torch.arange(classes, device=scores.device, dtype=torch.float32).unsqueeze(0)
    eps = 1e-6
    adjusted = scores + (float(classes) - ids) * eps
    _, top_idx = torch.topk(adjusted, min(k, classes), dim=1)
    top_scores = scores.gather(1, top_idx)
    return top_scores, top_idx


def _iter_beam_candidate_chunks(
    top_ids: dict[str, torch.Tensor],
    top_scores: dict[str, torch.Tensor],
    beam_widths: dict[str, int],
    *,
    chunk_size: int,
):
    device = next(iter(top_ids.values())).device
    widths = [int(beam_widths[h]) for h in HEAD_ORDER]
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
            score_parts.append(top_scores[head].gather(1, idx2))
        labels_chunk = torch.stack(labels_parts, dim=2)  # [B, Kc, 6]
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


def _bundle_beam_widths(bundle: dict[str, Any]) -> dict[str, int]:
    bw = bundle.get("beam_widths")
    if not isinstance(bw, dict):
        return dict(DEFAULT_BEAM_WIDTHS)
    out = dict(DEFAULT_BEAM_WIDTHS)
    for head in HEAD_ORDER:
        if head in bw:
            out[head] = int(bw[head])
    for head in HEAD_ORDER:
        out[head] = max(1, min(out[head], int(HEADS[head])))
    return out


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
        used_heads: set[str] = set()
        for a, b in self.pairs:
            used_heads.add(a)
            used_heads.add(b)
        self.emb = nn.ModuleDict({h: nn.Embedding(int(HEADS[h]), self.dim) for h in sorted(used_heads)})
        self.gates = nn.ModuleDict({_pair_key(a, b): nn.Linear(self.input_dim, self.dim) for a, b in self.pairs})

    def _gate(self, x: torch.Tensor, a: str, b: str) -> torch.Tensor:
        return torch.tanh(self.gates[_pair_key(a, b)](x))

    def score_neg(self, x: torch.Tensor, neg_labels: torch.Tensor) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer top-3 tuples using trained rankers")
    parser.add_argument("--bundle", required=True, type=Path, help="Path to bundle.json")
    parser.add_argument("--input-jsonl", required=True, type=Path, help="Input JSONL to run inference on")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL for predictions")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--require-eligible", action="store_true", help="Skip blocks not eligible")
    return parser.parse_args()


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ML tasks, but torch.cuda.is_available() is False")
    return torch.device("cuda")


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


class TupleReranker(nn.Module):
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
        self.emb = nn.ModuleDict({h: nn.Embedding(int(HEADS[h]), self.embed_dim) for h in HEAD_ORDER})
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def _tuple_embed_many(self, labels: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((labels.shape[0], labels.shape[1], self.embed_dim), device=labels.device, dtype=torch.float32)
        for head_idx, head in enumerate(HEAD_ORDER):
            out = out + self.emb[head](labels[:, :, head_idx])
        return out

    def score_many(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # x: [B,input_dim], labels: [B,K,6]
        x_emb = self.x_net(x)  # [B,D]
        t_emb = self._tuple_embed_many(labels)  # [B,K,D]
        x_exp = x_emb[:, None, :].expand(-1, t_emb.shape[1], -1)
        feat = torch.cat([x_exp, t_emb], dim=2).reshape(-1, self.embed_dim * 2)
        out = self.mlp(feat).reshape(t_emb.shape[0], t_emb.shape[1])
        return out


def build_raw_name_index(raw_names: list[str]) -> list[tuple[str, int | None]]:
    parsed: list[tuple[str, int | None]] = []
    for name in raw_names:
        if "[" in name and name.endswith("]"):
            key, rest = name.split("[", 1)
            idx = int(rest[:-1])
            parsed.append((key, idx))
        else:
            parsed.append((name, None))
    return parsed


def extract_numeric_by_names(row: dict[str, Any], parsed: list[tuple[str, int | None]]) -> np.ndarray:
    values: list[float] = []
    for key, idx in parsed:
        value = row.get(key)
        if idx is None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            else:
                values.append(0.0)
        else:
            if isinstance(value, list) and idx < len(value):
                v = value[idx]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def apply_feature_pipeline(raw_numeric: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    raw_indices = spec.get("raw_indices", [])
    if raw_numeric.shape[1] == 0 or not raw_indices:
        raw_sel = np.empty((raw_numeric.shape[0], 0), dtype=np.float32)
    else:
        raw_sel = raw_numeric[:, raw_indices]
    features: list[np.ndarray] = []
    if bool(spec.get("include_raw", True)):
        features.append(raw_sel)
    for transform in spec.get("transforms", []):
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

def apply_non_pixel_norm(non_pixel: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    norm = bundle.get("feature_norm") or {}
    mean = np.asarray(norm.get("non_pixel_mean", []), dtype=np.float32)
    std = np.asarray(norm.get("non_pixel_std", []), dtype=np.float32)
    if mean.size == 0 or std.size == 0:
        return non_pixel.astype(np.float32, copy=False)
    if non_pixel.shape[1] != mean.shape[0]:
        raise ValueError("non-pixel feature dimension mismatch vs bundle feature_norm")
    return ((non_pixel - mean[None, :]) / std[None, :]).astype(np.float32, copy=False)


def _find_feature_indices(raw_names: list[str], prefix: str, expected: int) -> list[int]:
    idx = [i for i, name in enumerate(raw_names) if name.startswith(prefix)]
    if expected and len(idx) != expected:
        return []
    return idx


def filterbits_prior_logits(
    raw_numeric: np.ndarray,
    raw_names: list[str],
    *,
    temp: float,
) -> dict[str, np.ndarray] | None:
    bits_none_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_none_min_by_filter[", 96)
    bits_inter_idx = _find_feature_indices(raw_names, "score_bits_plain_hilbert_interleave_min_by_filter[", 96)
    if not bits_none_idx or not bits_inter_idx:
        return None
    if temp <= 0:
        raise ValueError("filterbits_prior temp must be positive")

    bits_none = raw_numeric[:, bits_none_idx].astype(np.float32, copy=False)
    bits_inter = raw_numeric[:, bits_inter_idx].astype(np.float32, copy=False)
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


def energy_prior_logits(
    raw_numeric: np.ndarray,
    raw_names: list[str],
    *,
    temp: float,
) -> dict[str, np.ndarray] | None:
    pred_energy_idx = _find_feature_indices(raw_names, "score_residual_energy_by_predictor[", 8)
    reorder_tv_mean_idx = _find_feature_indices(raw_names, "reorder_tv_mean[", 8)
    if not pred_energy_idx or not reorder_tv_mean_idx:
        return None
    if temp <= 0:
        raise ValueError("energy prior temp must be positive")
    eps = 1e-6
    pred_energy = raw_numeric[:, pred_energy_idx].astype(np.float32, copy=False)
    reorder_tv = raw_numeric[:, reorder_tv_mean_idx].astype(np.float32, copy=False)
    pred_logits = -np.log1p(np.maximum(pred_energy, 0.0) + eps) / float(temp)
    reorder_logits = -np.log1p(np.maximum(reorder_tv, 0.0) + eps) / float(temp)
    return {
        "predictor": pred_logits.astype(np.float32, copy=False),
        "reorder": reorder_logits.astype(np.float32, copy=False),
    }


def _top3_from_heads(
    x: torch.Tensor,
    head_logits: dict[str, torch.Tensor],
    *,
    beam_widths: dict[str, int],
    beam_candidate_chunk: int,
    pairwise: PairwiseInteractions | None,
    reranker: TupleReranker | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_ids: dict[str, torch.Tensor] = {}
    top_scores: dict[str, torch.Tensor] = {}
    for head in HEAD_ORDER:
        sc, ids = _topk_with_tiebreak_torch(head_logits[head], int(beam_widths[head]))
        top_ids[head] = ids
        top_scores[head] = sc
    bsz = int(x.shape[0])
    k = 3
    best_scores = torch.full((bsz, k), -1e30, device=x.device, dtype=torch.float32)
    best_ranks = torch.full((bsz, k), (8 * 96 * 8 * 2 - 1), device=x.device, dtype=torch.int64)
    best_labels = torch.zeros((bsz, k, 6), device=x.device, dtype=torch.int64)
    for labels_chunk, head_sum_chunk in _iter_beam_candidate_chunks(
        top_ids, top_scores, beam_widths, chunk_size=int(beam_candidate_chunk)
    ):
        scores_chunk = head_sum_chunk.to(torch.float32)
        if reranker is not None:
            scores_chunk = reranker.score_many(x, labels_chunk).to(torch.float32)
        else:
            if pairwise is not None:
                scores_chunk = scores_chunk + pairwise.score_neg(x, labels_chunk).to(torch.float32)
        pred = labels_chunk[:, :, 0].to(torch.int64)
        perm = labels_chunk[:, :, 1].to(torch.int64)
        prim = labels_chunk[:, :, 2].to(torch.int64)
        sec = labels_chunk[:, :, 3].to(torch.int64)
        reorder = labels_chunk[:, :, 4].to(torch.int64)
        inter = labels_chunk[:, :, 5].to(torch.int64)
        filt = (perm << 4) | (prim << 2) | sec
        rank_chunk = (((pred * 96 + filt) * 8 + reorder) * 2 + inter).to(torch.int64)
        best_scores, best_ranks, best_labels = _streaming_topk_update(
            best_scores, best_ranks, best_labels, scores_chunk, rank_chunk, labels_chunk, k=k
        )
    return best_scores, best_labels


def is_eligible(row: dict[str, Any]) -> bool:
    if row.get("block_size") != [8, 8]:
        return False
    components = int(row.get("components", 0))
    return components in (3, 4)


def load_models(bundle: dict[str, Any], device: torch.device) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    for head in HEAD_ORDER:
        head_info = bundle["heads"][head]
        model = MLP(bundle["input_dim"], head_info["hidden_sizes"], head_info["classes"]).to(device)
        state = torch.load(head_info["path"], map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models[head] = model
    return models


def main() -> None:
    args = parse_args()
    device = ensure_cuda()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    raw_names = bundle.get("raw_feature_names", [])
    parsed_names = build_raw_name_index(raw_names)
    feature_spec = bundle["feature_spec"]

    models = load_models(bundle, device)
    pairwise = None
    if isinstance(bundle.get("pairwise"), dict):
        info = bundle["pairwise"]
        pairwise = PairwiseInteractions(
            int(bundle["input_dim"]),
            int(info["dim"]),
            pairs=[(str(a), str(b)) for a, b in (info.get("pairs") or [])],
            scale=float(info.get("scale", 1.0)),
        ).to(device)
        state = torch.load(info["path"], map_location="cpu")
        pairwise.load_state_dict(state)
        pairwise.eval()
    reranker = None
    if isinstance(bundle.get("reranker"), dict):
        info = bundle["reranker"]
        reranker = TupleReranker(bundle["input_dim"], int(info["embed_dim"]), int(info["hidden"])).to(device)
        state = torch.load(info["path"], map_location="cpu")
        reranker.load_state_dict(state)
        reranker.eval()
    use_filterbits_prior = bool(bundle.get("filterbits_prior", False))
    filterbits_prior_weight = float(bundle.get("filterbits_prior_weight", 1.0))
    filterbits_prior_temp = float(bundle.get("filterbits_prior_temp", 20.0))
    use_energy_prior = bool(bundle.get("energy_prior", False))
    energy_prior_weight = float(bundle.get("energy_prior_weight", 1.0))
    energy_prior_temp = float(bundle.get("energy_prior_temp", 1.0))
    beam_widths = _bundle_beam_widths(bundle)
    beam_candidate_chunk = int(bundle.get("beam_candidate_chunk", 2048))

    ensure_dir(args.out.parent)
    out_fp = args.out.open("w", encoding="utf-8")

    batch_features: list[np.ndarray] = []
    batch_raw_numeric: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch_features:
            return
        feats = np.stack(batch_features, axis=0).astype(np.float32)
        raw_mat = np.stack(batch_raw_numeric, axis=0).astype(np.float32)
        feats_t = torch.from_numpy(feats).to(device)
        with torch.no_grad():
            score_mode = str(bundle.get("score_mode", "log_softmax"))
            head_logits = {head: model(feats_t) for head, model in models.items()}
            if use_filterbits_prior and abs(filterbits_prior_weight) > 0.0:
                prior = filterbits_prior_logits(raw_mat, raw_names, temp=filterbits_prior_temp)
                if prior is not None:
                    for head, arr in prior.items():
                        head_logits[head] = head_logits[head] + float(filterbits_prior_weight) * torch.from_numpy(arr).to(device)
            if use_energy_prior and abs(energy_prior_weight) > 0.0:
                prior = energy_prior_logits(raw_mat, raw_names, temp=energy_prior_temp)
                if prior is not None:
                    for head, arr in prior.items():
                        head_logits[head] = head_logits[head] + float(energy_prior_weight) * torch.from_numpy(arr).to(device)
            if score_mode != "raw":
                head_logits = {head: torch.log_softmax(v, dim=1) for head, v in head_logits.items()}
            top_scores, top_labels = _top3_from_heads(
                feats_t,
                head_logits,
                beam_widths=beam_widths,
                beam_candidate_chunk=beam_candidate_chunk,
                pairwise=pairwise,
                reranker=reranker,
            )
            top_scores_np = top_scores.detach().cpu().numpy()
            top_labels_np = top_labels.detach().cpu().numpy()
        for i, row in enumerate(batch_rows):
            index = batch_indices[i]
            if args.require_eligible and not is_eligible(row):
                payload = {"index": index, "eligible": False, "predictions": []}
                out_fp.write(json.dumps(payload) + "\n")
                continue
            labels3 = top_labels_np[i]  # [3,6]
            scores3 = top_scores_np[i]  # [3]
            preds: list[dict[str, Any]] = []
            for j in range(3):
                pred = int(labels3[j][0])
                perm = int(labels3[j][1])
                prim = int(labels3[j][2])
                sec = int(labels3[j][3])
                reorder = int(labels3[j][4])
                inter = int(labels3[j][5])
                preds.append(
                    {
                        "predictor": pred,
                        "filter": join_filter(perm, prim, sec),
                        "reorder": reorder,
                        "interleave": inter,
                        "entropy": 0,
                        "score": float(scores3[j]),
                    }
                )
            payload = {"index": index, "eligible": True, "predictions": preds}
            out_fp.write(json.dumps(payload) + "\n")
        batch_features.clear()
        batch_raw_numeric.clear()
        batch_rows.clear()
        batch_indices.clear()

    with args.input_jsonl.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.require_eligible and not is_eligible(row):
                payload = {"index": idx, "eligible": False, "predictions": []}
                out_fp.write(json.dumps(payload) + "\n")
                continue
            pixels = extract_pixels_rgb(row)
            raw_numeric = extract_numeric_by_names(row, parsed_names)
            non_pixel = apply_feature_pipeline(raw_numeric[None, :], feature_spec)
            if non_pixel.shape[1] > 256:
                payload = {"index": idx, "eligible": False, "predictions": []}
                out_fp.write(json.dumps(payload) + "\n")
                continue
            non_pixel = apply_non_pixel_norm(non_pixel, bundle)
            features = np.concatenate([pixels, non_pixel.squeeze(0)], axis=0)
            if features.shape[0] != bundle["input_dim"]:
                raise ValueError("feature dimension mismatch vs bundle input_dim")
            batch_features.append(features)
            batch_raw_numeric.append(raw_numeric)
            batch_rows.append(row)
            batch_indices.append(idx)
            if len(batch_features) >= args.batch_size:
                flush_batch()
        flush_batch()

    out_fp.close()


if __name__ == "__main__":
    main()
