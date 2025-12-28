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
from tlg_ml_utils import HEADS, ensure_dir, now_iso

HEAD_ORDER = ["predictor", "cf_perm", "cf_primary", "cf_secondary", "reorder", "interleave"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-head prediction quality for a trained bundle")
    parser.add_argument("--bundle", required=True, type=Path, help="Path to bundle.json")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory with training.all.jsonl/labels.all.bin")
    parser.add_argument("--run-id", required=True, type=str, help="Run ID under ml/runs (for writing report)")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=200_000, help="Max samples to evaluate (valid split)")
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=200_000,
        help="Cap eligible blocks when building the dataset cache (default: 200000)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed (only used if split needs rebuild)")
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


def load_models(bundle: dict[str, Any], device: torch.device) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    for head in HEAD_ORDER:
        info = bundle["heads"][head]
        model = MLP(int(bundle["input_dim"]), list(info["hidden_sizes"]), int(info["classes"])).to(device)
        state = torch.load(info["path"], map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models[head] = model
    return models


def _class_hist(labels: np.ndarray, classes: int) -> dict[str, Any]:
    counts = np.bincount(labels.astype(np.int64), minlength=int(classes)).astype(np.int64)
    total = int(counts.sum())
    p = counts / max(1, total)
    eps = 1e-12
    ent = float(-(p[p > 0] * np.log2(p[p > 0] + eps)).sum())
    top = np.argsort(-counts)[: min(10, int(classes))]
    return {
        "total": total,
        "entropy_bits": ent,
        "top_classes": [{"id": int(i), "count": int(counts[i])} for i in top],
    }


def main() -> None:
    args = parse_args()
    device = ensure_cuda()
    run_dir = args.run_root / args.run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    score_mode = str(bundle.get("score_mode", "log_softmax"))
    if score_mode not in ("raw", "log_softmax"):
        raise SystemExit(f"unsupported bundle score_mode: {score_mode}")

    seed = args.seed
    if seed is None:
        config_path = run_dir / "config.json"
        if config_path.exists():
            seed = int(json.loads(config_path.read_text(encoding="utf-8")).get("seed", 1234))
        else:
            seed = 1234
    dataset = build_dataset(run_dir, args.in_dir, seed=seed, rebuild=False, max_blocks=int(args.max_blocks))

    # Reconstruct the same feature matrix used by training for this bundle.
    from eval_rankers import apply_feature_pipeline, apply_non_pixel_norm  # local import to avoid duplication

    non_pixel = apply_feature_pipeline(dataset.raw_numeric, bundle["feature_spec"])
    non_pixel = apply_non_pixel_norm(non_pixel, bundle)
    features = np.concatenate([dataset.pixels, non_pixel], axis=1).astype(np.float32, copy=False)
    if int(features.shape[1]) != int(bundle["input_dim"]):
        raise SystemExit("feature dim mismatch vs bundle input_dim")

    valid_idx = dataset.valid_indices
    if args.max_samples and valid_idx.shape[0] > int(args.max_samples):
        valid_idx = valid_idx[: int(args.max_samples)]
    x = torch.from_numpy(features[valid_idx]).to(device=device, dtype=torch.float32)
    y_best = dataset.labels_best[valid_idx].astype(np.int64, copy=False)
    y_second = dataset.labels_second[valid_idx].astype(np.int64, copy=False)
    n = int(x.shape[0])
    bs = max(1, int(args.batch_size))

    models = load_models(bundle, device)
    t0 = time.time()

    metrics: dict[str, Any] = {
        "created_at": now_iso(),
        "bundle": str(args.bundle),
        "in_dir": str(args.in_dir),
        "run_id": args.run_id,
        "score_mode": score_mode,
        "n_valid_used": n,
        "batch_size": bs,
        "heads": {},
    }

    # Class balance stats.
    for head_idx, head in enumerate(HEAD_ORDER):
        classes = int(HEADS[head])
        metrics["heads"][head] = {
            "classes": classes,
            "label_hist_best": _class_hist(y_best[:, head_idx], classes),
            "label_hist_second": _class_hist(y_second[:, head_idx], classes),
        }

    # Prediction quality.
    correct_top1_best: dict[str, int] = {h: 0 for h in HEAD_ORDER}
    hit_top2_best_or_second: dict[str, int] = {h: 0 for h in HEAD_ORDER}
    hit_top4_best_or_second: dict[str, int] = {h: 0 for h in HEAD_ORDER}
    mean_logp_best: dict[str, float] = {h: 0.0 for h in HEAD_ORDER}
    mean_logp_second: dict[str, float] = {h: 0.0 for h in HEAD_ORDER}

    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            xb = x[s:e]
            logits = {h: models[h](xb) for h in HEAD_ORDER}
            if score_mode != "raw":
                logits = {h: torch.log_softmax(v, dim=1) for h, v in logits.items()}
            for head_idx, head in enumerate(HEAD_ORDER):
                lb = torch.from_numpy(y_best[s:e, head_idx]).to(device=device)
                ls = torch.from_numpy(y_second[s:e, head_idx]).to(device=device)
                logp = logits[head]
                mean_logp_best[head] += float(logp.gather(1, lb[:, None]).mean().item()) * (e - s)
                mean_logp_second[head] += float(logp.gather(1, ls[:, None]).mean().item()) * (e - s)
                top1 = logp.argmax(dim=1)
                correct_top1_best[head] += int((top1 == lb).sum().item())

                # Top-k hit: whether either best or second is contained in top-k.
                # (This upper-bounds what a perfect downstream combiner can do if a head is too uncertain.)
                for k, acc in ((2, hit_top2_best_or_second), (4, hit_top4_best_or_second)):
                    topk = torch.topk(logp, k=min(k, int(HEADS[head])), dim=1).indices
                    hit = ((topk == lb[:, None]) | (topk == ls[:, None])).any(dim=1)
                    acc[head] += int(hit.sum().item())

    for head in HEAD_ORDER:
        metrics["heads"][head].update(
            {
                "top1_matches_best": float(correct_top1_best[head]) / max(1, n),
                "top2_contains_best_or_second": float(hit_top2_best_or_second[head]) / max(1, n),
                "top4_contains_best_or_second": float(hit_top4_best_or_second[head]) / max(1, n),
                "mean_logp_best": float(mean_logp_best[head]) / max(1, n),
                "mean_logp_second": float(mean_logp_second[head]) / max(1, n),
            }
        )

    # Also compute a simple baseline: always predict the single most frequent class.
    baselines: dict[str, Any] = {}
    for head_idx, head in enumerate(HEAD_ORDER):
        classes = int(HEADS[head])
        counts = np.bincount(y_best[:, head_idx], minlength=classes)
        mode = int(counts.argmax())
        acc = float((y_best[:, head_idx] == mode).mean())
        baselines[head] = {"mode_class": mode, "mode_acc_best": acc}
    metrics["baseline"] = baselines
    metrics["elapsed_sec"] = float(time.time() - t0)

    out_path = run_dir / "artifacts" / "head_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    for head in HEAD_ORDER:
        h = metrics["heads"][head]
        print(
            f"{head:12s} top1(best)={h['top1_matches_best']*100:6.2f}% "
            f"top2(any)={h['top2_contains_best_or_second']*100:6.2f}% "
            f"top4(any)={h['top4_contains_best_or_second']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()
