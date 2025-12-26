#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset_builder import build_dataset
from tlg_ml_utils import apply_transform, ensure_dir, join_filter, save_json, topk_with_tiebreak

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
    parser = argparse.ArgumentParser(description="Evaluate hit-rate using trained rankers")
    parser.add_argument("--bundle", required=True, type=Path, help="Path to bundle.json")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory with training.all.jsonl/labels.all.bin")
    parser.add_argument("--run-id", required=True, type=str, help="Run ID under ml/runs")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--use-valid", action="store_true", help="Evaluate on valid split only")
    parser.add_argument("--seed", type=int, default=None, help="Seed (used only if split needs rebuild)")
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


def evaluate_hit_rate(
    device: torch.device,
    models: dict[str, nn.Module],
    features: np.ndarray,
    labels_best: np.ndarray,
    labels_second: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    hits = 0
    total = indices.shape[0]
    if total == 0:
        return 0.0

    data = torch.from_numpy(features[indices]).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=False)

    offset = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            batch_size_local = xb.shape[0]
            logits = {head: model(xb).detach().cpu().numpy() for head, model in models.items()}
            for i in range(batch_size_local):
                idx = indices[offset + i]
                best = labels_best[idx]
                second = labels_second[idx]
                best_filter = join_filter(int(best[1]), int(best[2]), int(best[3]))
                second_filter = join_filter(int(second[1]), int(second[2]), int(second[3]))
                true_tuples = {
                    (int(best[0]), best_filter, int(best[4]), int(best[5])),
                    (int(second[0]), second_filter, int(second[4]), int(second[5])),
                }
                top_lists = {
                    head: topk_with_tiebreak(logits[head][i], BEAM_WIDTHS[head]) for head in HEAD_ORDER
                }
                candidates: list[tuple[float, tuple[int, int, int, int]]] = []
                for pred_id, pred_score in top_lists["predictor"]:
                    for perm_id, perm_score in top_lists["cf_perm"]:
                        for prim_id, prim_score in top_lists["cf_primary"]:
                            for sec_id, sec_score in top_lists["cf_secondary"]:
                                filter_code = join_filter(perm_id, prim_id, sec_id)
                                for re_id, re_score in top_lists["reorder"]:
                                    for inter_id, inter_score in top_lists["interleave"]:
                                        score = (
                                            pred_score
                                            + perm_score
                                            + prim_score
                                            + sec_score
                                            + re_score
                                            + inter_score
                                        )
                                        candidates.append(
                                            (float(score), (pred_id, filter_code, re_id, inter_id))
                                        )
                candidates.sort(key=lambda x: (-x[0], x[1]))
                top3 = {c[1] for c in candidates[:3]}
                if top3 & true_tuples:
                    hits += 1
            offset += batch_size_local
    return hits / total


def main() -> None:
    args = parse_args()
    device = ensure_cuda()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    run_dir = args.run_root / args.run_id
    seed = args.seed
    config_path = run_dir / "config.json"
    if seed is None and config_path.exists():
        seed = json.loads(config_path.read_text(encoding="utf-8")).get("seed", 1234)
    if seed is None:
        seed = 1234
    dataset = build_dataset(run_dir, args.in_dir, seed=seed, rebuild=False)

    non_pixel = apply_feature_pipeline(dataset.raw_numeric, bundle["feature_spec"])
    if non_pixel.shape[1] > 256:
        raise SystemExit("non-pixel feature dimension exceeds 256")
    features = np.concatenate([dataset.pixels, non_pixel], axis=1).astype(np.float32)
    if features.shape[1] != bundle["input_dim"]:
        raise SystemExit("feature dimension mismatch vs bundle input_dim")

    indices = dataset.valid_indices if args.use_valid else np.arange(features.shape[0], dtype=np.int64)
    models = load_models(bundle, device)
    hit_rate = evaluate_hit_rate(
        device,
        models,
        features,
        dataset.labels_best,
        dataset.labels_second,
        indices,
        args.batch_size,
    )

    payload = {
        "valid_hit_rate" if args.use_valid else "hit_rate": hit_rate,
        "count": int(indices.shape[0]),
    }
    print(json.dumps(payload, indent=2))

    out_path = args.bundle.parent / "eval_hit_rate.json"
    ensure_dir(out_path.parent)
    save_json(out_path, payload)


if __name__ == "__main__":
    main()
