#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tlg_ml_utils import apply_transform, ensure_dir, extract_pixels_rgb, join_filter, topk_with_tiebreak

HEAD_ORDER = ["predictor", "cf_perm", "cf_primary", "cf_secondary", "reorder", "interleave"]
BEAM_WIDTHS = {
    "predictor": 4,
    "cf_perm": 4,
    "cf_primary": 3,
    "cf_secondary": 2,
    "reorder": 4,
    "interleave": 2,
}


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
    features = [raw_sel]
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


def top3_candidates(logits: dict[str, np.ndarray]) -> list[tuple[float, tuple[int, int, int, int]]]:
    top_lists = {head: topk_with_tiebreak(logits[head], BEAM_WIDTHS[head]) for head in HEAD_ORDER}
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for pred_id, pred_score in top_lists["predictor"]:
        for perm_id, perm_score in top_lists["cf_perm"]:
            for prim_id, prim_score in top_lists["cf_primary"]:
                for sec_id, sec_score in top_lists["cf_secondary"]:
                    filter_code = join_filter(perm_id, prim_id, sec_id)
                    for re_id, re_score in top_lists["reorder"]:
                        for inter_id, inter_score in top_lists["interleave"]:
                            score = (
                                pred_score + perm_score + prim_score + sec_score + re_score + inter_score
                            )
                            candidates.append((float(score), (pred_id, filter_code, re_id, inter_id)))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[:3]


def main() -> None:
    args = parse_args()
    device = ensure_cuda()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    raw_names = bundle.get("raw_feature_names", [])
    parsed_names = build_raw_name_index(raw_names)
    feature_spec = bundle["feature_spec"]

    models = load_models(bundle, device)

    ensure_dir(args.out.parent)
    out_fp = args.out.open("w", encoding="utf-8")

    batch_features: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch_features:
            return
        feats = np.stack(batch_features, axis=0).astype(np.float32)
        feats_t = torch.from_numpy(feats).to(device)
        with torch.no_grad():
            logits = {
                head: torch.log_softmax(model(feats_t), dim=1).detach().cpu().numpy() for head, model in models.items()
            }
        for i, row in enumerate(batch_rows):
            index = batch_indices[i]
            if args.require_eligible and not is_eligible(row):
                payload = {"index": index, "eligible": False, "predictions": []}
                out_fp.write(json.dumps(payload) + "\n")
                continue
            per_head = {head: logits[head][i] for head in HEAD_ORDER}
            top3 = top3_candidates(per_head)
            preds = [
                {
                    "predictor": t[1][0],
                    "filter": t[1][1],
                    "reorder": t[1][2],
                    "interleave": t[1][3],
                    "entropy": 0,
                    "score": t[0],
                }
                for t in top3
            ]
            payload = {"index": index, "eligible": True, "predictions": preds}
            out_fp.write(json.dumps(payload) + "\n")
        batch_features.clear()
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
            batch_rows.append(row)
            batch_indices.append(idx)
            if len(batch_features) >= args.batch_size:
                flush_batch()
        flush_batch()

    out_fp.close()


if __name__ == "__main__":
    main()
