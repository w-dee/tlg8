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
from tlg_ml_utils import RAW_RGB_DIMS, ensure_dir, now_iso, save_json


HEAD_ORDER = ["predictor", "filter_i", "reorder"]
HEADS = {"predictor": 8, "filter_i": 192, "reorder": 8}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden)))
            layers.append(nn.ReLU())
            last_dim = int(hidden)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ML tasks, but torch.cuda.is_available() is False")
    return torch.device("cuda")


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        out.append(int(part))
    return out


def _parse_eval_topn(spec: str) -> list[int]:
    out = sorted(set(int(p.strip()) for p in spec.split(",") if p.strip()))
    if not out:
        raise ValueError("--eval-topn must not be empty")
    return out


def _count_params(model: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def _batch_slices(n: int, bs: int) -> list[tuple[int, int]]:
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_teacher_bundle(path: Path) -> dict[str, Any]:
    b = _load_json(path)
    if b.get("head_order") != HEAD_ORDER:
        raise SystemExit(f"unexpected teacher head_order: {b.get('head_order')} expected={HEAD_ORDER}")
    return b


def _resume_state(run_dir: Path) -> tuple[int, dict[str, Any] | None]:
    progress = run_dir / "progress.jsonl"
    if not progress.exists():
        return 0, None
    best: dict[str, Any] | None = None
    best_hit32 = -1.0
    last_trial = -1
    with progress.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = int(obj.get("trial_id", -1))
            last_trial = max(last_trial, tid)
            rates = obj.get("valid_hit_rates_at", {})
            hit32 = float(rates.get("32", -1.0)) if isinstance(rates, dict) else -1.0
            if hit32 > best_hit32:
                best_hit32 = hit32
                best = obj
    return last_trial + 1, best


def _targets_bits_weighted(
    labels_best: np.ndarray, labels_second: np.ndarray, bits: np.ndarray, *, target_temp: float
) -> dict[str, np.ndarray]:
    # Same semantics as train_rankers_filteri.py: soft preference based on best/second bits.
    n = int(labels_best.shape[0])
    out: dict[str, np.ndarray] = {}
    bits_best = bits[:, 0].astype(np.float64)
    bits_second = bits[:, 1].astype(np.float64)
    if target_temp <= 0:
        raise ValueError("--target-temp must be positive")
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
        t[same, :] = 0.0
        t[same, b[same]] = 1.0
        out[head] = t
    return out


def _distill_epoch_kl(
    *,
    teacher: nn.Module,
    student: nn.Module,
    x: torch.Tensor,
    prior: torch.Tensor | None,
    batch_size: int,
    temp: float,
    opt: torch.optim.Optimizer | None,
    hard_targets: torch.Tensor | None,
    hard_weight: float,
) -> float:
    if temp <= 0:
        raise ValueError("--distill-temp must be positive")
    if hard_weight < 0:
        raise ValueError("--hard-weight must be non-negative")
    n = int(x.shape[0])
    losses: list[float] = []
    for s, e in _batch_slices(n, batch_size):
        xb = x[s:e]
        tb = teacher(xb)
        sb = student(xb)
        if prior is not None:
            tb = tb + prior[s:e]
            sb = sb + prior[s:e]

        # Teacher probabilities (no grad).
        with torch.no_grad():
            t_logp = torch.log_softmax(tb / float(temp), dim=1)
            t_p = t_logp.exp()
        s_logp = torch.log_softmax(sb / float(temp), dim=1)
        kl = (t_p * (t_logp - s_logp)).sum(dim=1).mean() * (float(temp) ** 2)
        loss = kl
        if hard_targets is not None and hard_weight > 0.0:
            ht = hard_targets[s:e]
            s_logp_hard = torch.log_softmax(sb, dim=1)
            ce = (-(ht * s_logp_hard).sum(dim=1).mean())
            loss = (1.0 - float(hard_weight)) * loss + float(hard_weight) * ce

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if losses else float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill triad rankers (predictor + filter_i + reorder)")
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--teacher-bundle", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("ml/runs"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-blocks", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--student-hidden-grid", type=str, default="128,64;64,32;32")
    parser.add_argument("--distill-temp", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=0.0)
    parser.add_argument("--target-temp", type=float, default=10.0)

    parser.add_argument("--eval-topn", type=str, default="3,10,16,32")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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

    teacher_bundle = _load_teacher_bundle(args.teacher_bundle)
    teacher_hidden = [int(x) for x in teacher_bundle["heads"]["predictor"]["hidden_sizes"]]
    teacher_input_dim = int(teacher_bundle["input_dim"])

    dataset = build_dataset(run_dir, args.in_dir, int(args.seed), rebuild=False, max_blocks=int(args.max_blocks))

    # Feature spec: same as train_rankers_filteri (filter budget).
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
    if int(x_train_np.shape[1]) != teacher_input_dim:
        raise SystemExit(f"teacher input_dim mismatch: teacher={teacher_input_dim} dataset={int(x_train_np.shape[1])}")

    x_train = torch.from_numpy(x_train_np).to(device)
    x_valid = torch.from_numpy(x_valid_np).to(device)

    # Priors: load from teacher bundle and recompute from dataset using train_rankers_filteri functions.
    # Keep this distiller independent by importing the existing prior helpers.
    from train_rankers_filteri import _energy_prior, _filteri_prior, _pred_reorder_bits_prior  # type: ignore

    prior_cfg = teacher_bundle.get("prior", {})
    prior_w = float(prior_cfg.get("weight", 0.0))
    filterbits_temp = float(prior_cfg.get("filterbits_temp", 20.0))
    energy_temp = float(prior_cfg.get("energy_temp", 1.0))

    prior_filter_i = _filteri_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=filterbits_temp) * prior_w
    prior_filter_i_v = _filteri_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=filterbits_temp) * prior_w
    prior_pred, prior_re = _energy_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=energy_temp)
    prior_pred_v, prior_re_v = _energy_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=energy_temp)
    prior_pred = prior_pred * prior_w
    prior_re = prior_re * prior_w
    prior_pred_v = prior_pred_v * prior_w
    prior_re_v = prior_re_v * prior_w
    bits_pred, bits_re = _pred_reorder_bits_prior(dataset.raw_numeric, dataset.raw_names, train_idx_sorted, temp=filterbits_temp)
    bits_pred_v, bits_re_v = _pred_reorder_bits_prior(dataset.raw_numeric, dataset.raw_names, valid_idx_sorted, temp=filterbits_temp)
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

    hard_targets: dict[str, torch.Tensor] = {}
    if float(args.hard_weight) > 0.0:
        t_tr = _targets_bits_weighted(
            dataset.labels_best[train_idx_sorted].astype(np.int64),
            dataset.labels_second[train_idx_sorted].astype(np.int64),
            dataset.bits[train_idx_sorted],
            target_temp=float(args.target_temp),
        )
        t_va = _targets_bits_weighted(
            dataset.labels_best[valid_idx_sorted].astype(np.int64),
            dataset.labels_second[valid_idx_sorted].astype(np.int64),
            dataset.bits[valid_idx_sorted],
            target_temp=float(args.target_temp),
        )
        hard_targets = {
            f"{head}_train": torch.from_numpy(t_tr[head]).to(device)
            for head in HEAD_ORDER
        } | {f"{head}_valid": torch.from_numpy(t_va[head]).to(device) for head in HEAD_ORDER}

    # Teacher models.
    teacher_models = {h: MLP(teacher_input_dim, teacher_hidden, int(HEADS[h])).to(device) for h in HEAD_ORDER}
    for h in HEAD_ORDER:
        teacher_models[h].load_state_dict(torch.load(Path(teacher_bundle["heads"][h]["path"]), map_location="cpu"))
        teacher_models[h].eval()
        for p in teacher_models[h].parameters():
            p.requires_grad_(False)

    # Trial grid.
    grid: list[list[int]] = []
    for part in [p.strip() for p in str(args.student_hidden_grid).split(";") if p.strip()]:
        grid.append(_parse_int_list(part))
    if not grid:
        raise SystemExit("--student-hidden-grid must not be empty")

    topn = _parse_eval_topn(args.eval_topn)
    start_trial, best_prev = _resume_state(run_dir)
    if best_prev is not None:
        save_json(run_dir / "best.json", {"best_trial_id": int(best_prev["trial_id"]), "valid_hit_rates_at": best_prev.get("valid_hit_rates_at", {}), "bundle_path": best_prev.get("bundle_path", "")})

    for trial_id, hidden_student in enumerate(grid[start_trial:], start=start_trial):
        trial_dir = run_dir / "artifacts" / f"trial_{trial_id:04d}"
        ensure_dir(trial_dir)

        student_models = {h: MLP(teacher_input_dim, hidden_student, int(HEADS[h])).to(device) for h in HEAD_ORDER}
        opts = {h: torch.optim.Adam(student_models[h].parameters(), lr=float(args.lr)) for h in HEAD_ORDER}

        best_hit32 = -1.0
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        best_loss = float("inf")
        no_imp = 0
        best_epoch = 0

        for epoch in range(int(args.max_epochs)):
            # Train each head for one epoch.
            train_losses: dict[str, float] = {}
            for h in HEAD_ORDER:
                train_losses[h] = _distill_epoch_kl(
                    teacher=teacher_models[h],
                    student=student_models[h],
                    x=x_train,
                    prior=priors_train[h],
                    batch_size=int(args.batch_size),
                    temp=float(args.distill_temp),
                    opt=opts[h],
                    hard_targets=hard_targets.get(f"{h}_train"),
                    hard_weight=float(args.hard_weight),
                )

            # Validate.
            valid_losses: dict[str, float] = {}
            with torch.no_grad():
                for h in HEAD_ORDER:
                    valid_losses[h] = _distill_epoch_kl(
                        teacher=teacher_models[h],
                        student=student_models[h],
                        x=x_valid,
                        prior=priors_valid[h],
                        batch_size=int(args.batch_size),
                        temp=float(args.distill_temp),
                        opt=None,
                        hard_targets=hard_targets.get(f"{h}_valid"),
                        hard_weight=float(args.hard_weight),
                    )
            vloss = float(np.mean(list(valid_losses.values())))

            # Evaluate hit@K on valid.
            from train_rankers_filteri import evaluate_hit_rates  # type: ignore

            with torch.no_grad():
                logits_t = {h: student_models[h](x_valid) + priors_valid[h] for h in HEAD_ORDER}
                logits_np = {h: torch.log_softmax(v, dim=1).detach().cpu().numpy() for h, v in logits_t.items()}
            hit_rates = evaluate_hit_rates(
                logits_np,
                dataset.labels_best[valid_idx_sorted],
                dataset.labels_second[valid_idx_sorted],
                topn=topn,
                beam=teacher_bundle.get("beam", {"predictor": 8, "filter_i": 192, "reorder": 8}),
            )
            hit32 = float(hit_rates.get(32, 0.0))

            if hit32 >= best_hit32 + 1e-9:
                best_hit32 = hit32
                best_epoch = epoch
                best_state = {h: {n: v.detach().cpu().clone() for n, v in student_models[h].state_dict().items()} for h in HEAD_ORDER}

            if vloss < best_loss - 1e-6:
                best_loss = vloss
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= int(args.patience):
                    break

            print(
                f"[trial {trial_id:04d} epoch {epoch}] valid_loss={vloss:.5f} hit@32={hit32*100:.2f}%",
                flush=True,
            )

        if best_state is not None:
            for h in HEAD_ORDER:
                student_models[h].load_state_dict(best_state[h])

        # Final eval (best state) and save.
        with torch.no_grad():
            logits_t = {h: student_models[h](x_valid) + priors_valid[h] for h in HEAD_ORDER}
            logits_np = {h: torch.log_softmax(v, dim=1).detach().cpu().numpy() for h, v in logits_t.items()}
        from train_rankers_filteri import evaluate_hit_rates  # type: ignore

        hit_rates = evaluate_hit_rates(
            logits_np,
            dataset.labels_best[valid_idx_sorted],
            dataset.labels_second[valid_idx_sorted],
            topn=topn,
            beam=teacher_bundle.get("beam", {"predictor": 8, "filter_i": 192, "reorder": 8}),
        )

        for h in HEAD_ORDER:
            torch.save(student_models[h].state_dict(), trial_dir / f"{h}.pt")

        params = {h: _count_params(student_models[h]) for h in HEAD_ORDER}
        bundle = {
            "run_id": args.run_id,
            "trial_id": int(trial_id),
            "kind": "distilled_filteri_triad",
            "teacher_bundle": str(args.teacher_bundle),
            "head_order": HEAD_ORDER,
            "heads": {h: {"path": str(trial_dir / f"{h}.pt"), "hidden_sizes": hidden_student, "classes": int(HEADS[h])} for h in HEAD_ORDER},
            "input_dim": teacher_input_dim,
            "pixel_dim": RAW_RGB_DIMS,
            "raw_feature_names": dataset.raw_names,
            "feature_spec": {"name": spec.name, "raw_indices": spec.raw_indices, "transforms": [], "include_raw": True},
            "feature_norm": {"non_pixel_mean": non_pixel_mean.tolist(), "non_pixel_std": non_pixel_std.tolist()},
            "score_mode": "log_softmax",
            "prior": teacher_bundle.get("prior", {}),
            "beam": teacher_bundle.get("beam", {}),
            "eval_topn": topn,
            "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()},
            "distill": {"temp": float(args.distill_temp), "hard_weight": float(args.hard_weight), "target_temp": float(args.target_temp)},
            "params": {"per_head": params, "total": int(sum(params.values()))},
        }
        save_json(trial_dir / "bundle.json", bundle)
        save_json(trial_dir / "eval.json", {"valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()}})

        entry = {
            "trial_id": int(trial_id),
            "timestamp": now_iso(),
            "teacher_bundle": str(args.teacher_bundle),
            "dataset": {"in_dir": str(args.in_dir), "max_blocks": int(args.max_blocks)},
            "student_hidden": hidden_student,
            "distill_temp": float(args.distill_temp),
            "hard_weight": float(args.hard_weight),
            "target_temp": float(args.target_temp),
            "prior": teacher_bundle.get("prior", {}),
            "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()},
            "params": {"per_head": params, "total": int(sum(params.values()))},
            "bundle_path": str(trial_dir / "bundle.json"),
            "status": "ok",
        }
        (run_dir / "progress.jsonl").open("a", encoding="utf-8").write(json.dumps(entry) + "\n")

        # Best tracking by hit@32.
        best_path = run_dir / "best.json"
        prev_best = _load_json(best_path) if best_path.exists() else None
        prev_hit32 = float(prev_best.get("valid_hit_rates_at", {}).get("32", -1.0)) if isinstance(prev_best, dict) else -1.0
        cur_hit32 = float(hit_rates.get(32, 0.0))
        if cur_hit32 > prev_hit32 + 1e-9:
            save_json(run_dir / "best.json", {"best_trial_id": int(trial_id), "valid_hit_rates_at": {str(k): float(v) for k, v in hit_rates.items()}, "bundle_path": str(trial_dir / "bundle.json")})

    config = dict(vars(args))
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    save_json(run_dir / "config.json", config)


if __name__ == "__main__":
    main()
