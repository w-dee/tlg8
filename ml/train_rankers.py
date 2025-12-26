#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_builder import DatasetCache, build_dataset
from make_bits_sidecar import write_bits_sidecar
from tlg_ml_utils import (
    HEADS,
    RAW_RGB_DIMS,
    apply_transform,
    ensure_dir,
    join_filter,
    now_iso,
    save_json,
    topk_with_tiebreak,
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
        help="Directory with training.all.jsonl/labels.all.bin (defaults to ml/runs/packed_smoke if present)",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under ml/runs (auto if omitted)")
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"), help="Run root directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--max-epochs", type=int, default=40, help="Max epochs per head")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--max-trials", type=int, default=None, help="Max trials to run")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild dataset cache")
    parser.add_argument("--smoke", action="store_true", help="Run a small, fast smoke trial")
    return parser.parse_args()


def resolve_in_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    fallback = Path("ml/runs/packed_smoke")
    if (fallback / "training.all.jsonl").is_file() and (fallback / "labels.all.bin").is_file():
        return fallback
    raise SystemExit("Missing --in-dir and no fallback dataset found at ml/runs/packed_smoke")


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


def train_head(
    head: str,
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = soft_ce_loss(logits, yb)
                losses.append(float(loss.item()))
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, best_epoch


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
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)

    for head in models.values():
        head.eval()

    offset = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            batch_size_local = xb.shape[0]
            logits = {}
            for head_name, model in models.items():
                logits[head_name] = model(xb).detach().cpu().numpy()
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
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 64)
        args.max_trials = 1 if args.max_trials is None else min(args.max_trials, 1)
    args.in_dir = resolve_in_dir(args.in_dir)
    device = ensure_cuda()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or time_run_id()
    run_dir = args.run_root / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "dataset")
    ensure_dir(run_dir / "splits")

    bits_path = run_dir / "dataset" / "bits.all.npy"
    if not bits_path.exists():
        write_bits_sidecar(args.in_dir, run_dir, force=False)

    dataset = build_dataset(run_dir, args.in_dir, args.seed, rebuild=args.rebuild_dataset)

    feature_specs = build_feature_specs(dataset)
    arch_specs = build_arch_specs()
    config_path = run_dir / "config.json"
    config = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "run_id": run_id,
        "input_dir": str(args.in_dir),
        "feature_specs": [
            {"name": spec.name, "raw_indices": spec.raw_indices, "transforms": spec.transforms}
            for spec in feature_specs
        ],
        "arch_specs": arch_specs,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
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
        feature_spec, arch_spec = get_trial_config(trial_id, feature_specs, arch_specs, args.seed)
        non_pixel = apply_feature_pipeline(dataset.raw_numeric, feature_spec)
        if non_pixel.shape[1] > 256:
            continue
        features = np.concatenate([dataset.pixels, non_pixel], axis=1).astype(np.float32)
        input_dim = features.shape[1]

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

        try:
            for head in HEAD_ORDER:
                hidden_sizes = arch_spec[head]
                model = MLP(input_dim, hidden_sizes, HEADS[head]).to(device)
                models[head] = model
                head_params[head] = count_params(model)

                x_train = torch.from_numpy(features[train_idx]).float()
                y_train = torch.from_numpy(targets[head][train_idx]).float()
                x_valid = torch.from_numpy(features[valid_idx]).float()
                y_valid = torch.from_numpy(targets[head][valid_idx]).float()

                train_loader = DataLoader(
                    TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
                )
                valid_loader = DataLoader(
                    TensorDataset(x_valid, y_valid), batch_size=args.batch_size, shuffle=False
                )
                valid_loss, best_epoch = train_head(
                    head,
                    model,
                    device,
                    train_loader,
                    valid_loader,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    lr=args.lr,
                )
                head_losses[head] = valid_loss
                head_epochs[head] = best_epoch

                model_path = trial_dir / f"{head}.pt"
                torch.save(model.state_dict(), model_path)
                model_bundle["heads"][head] = {
                    "path": str(model_path),
                    "hidden_sizes": hidden_sizes,
                    "classes": HEADS[head],
                }

            valid_hit_rate = evaluate_hit_rate(
                device,
                models,
                features,
                dataset.labels_best,
                dataset.labels_second,
                valid_idx,
                args.batch_size,
            )
        except Exception as exc:
            valid_hit_rate = 0.0
            status = "failed"
            error_summary = str(exc)

        total_params = int(sum(head_params.values()))
        valid_loss_mean = float(np.mean(list(head_losses.values()))) if head_losses else float("inf")

        save_json(trial_dir / "feature_spec.json", model_bundle["feature_spec"])
        save_json(
            trial_dir / "eval.json",
            {
                "valid_hit_rate": valid_hit_rate,
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
            "params": {
                "per_head": head_params,
                "total": total_params,
            },
            "head_losses": head_losses,
            "head_best_epoch": head_epochs,
            "bundle_path": str(bundle_path),
            "status": status,
            "error": error_summary,
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
