#!/usr/bin/env bash
set -euo pipefail

# Path to sweep result CSV
CSV="runs/sweeps/reorder_v2/results_v2.csv"
# 起点チェックポイント（必要に応じて変更）
INIT_FROM="runs/ranker_R0_reorder_only_v2/multitask_best.pt"

python3 - << 'PY'
import csv
import re
import shlex
import subprocess
from pathlib import Path

CSV = "runs/sweeps/reorder_v2/results_v2.csv"
INIT_FROM = "runs/ranker_R0_reorder_only_v2/multitask_best.pt"

if not Path(CSV).is_file():
    raise SystemExit(f"CSV not found: {CSV}")
if not Path(INIT_FROM).is_file():
    raise SystemExit(f"init-from checkpoint not found: {INIT_FROM}")

rows = []
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 必須メトリクスが無ければスキップ or エラー
        if "val_reorder_top3" not in row or "val_reorder_top1" not in row:
            raise SystemExit("CSV must contain val_reorder_top3 and val_reorder_top1 columns")
        rows.append(row)

if not rows:
    raise SystemExit("No data rows in CSV")

# val_reorder_top3 降順 → val_reorder_top1 降順 でソート
def key_fn(r):
    def to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("-inf")
    return (to_float(r["val_reorder_top3"]), to_float(r["val_reorder_top1"]))

rows.sort(key=key_fn, reverse=True)
top5 = rows[:5]

def parse_from_id(pattern, run_id, fallback=None):
    """
    例: pattern=r"b([0-9\.]+)" などで id から数値を拾う。
    """
    m = re.search(pattern, run_id)
    if m:
        return m.group(1)
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not parse pattern {pattern!r} from id {run_id!r}")

def parse_sched_and_warm(run_id: str, row: dict):
    """
    lr-schedule と warmup-epochs を決める。
    - idに 'schcosine_warm5' → schedule='cosine', warm=5
    - idに 'schconstant' などで warm 無し → schedule='constant', warm=0
    """
    # まず sched カラムを優先（あれば）
    sched = (row.get("sched") or "").strip()
    warm = (row.get("warmup_epochs") or "").strip()

    if not sched:
        m = re.search(r"sch([a-zA-Z0-9_]+)", run_id)
        if m:
            sched = m.group(1)

    # id 例: schcosine_warm5 / schcosine / schconstant / schplateau
    sched_lower = sched.lower()

    if "cosine" in sched_lower:
        lr_sched = "cosine"
    elif "plateau" in sched_lower:
        lr_sched = "plateau"
    elif "constant" in sched_lower:
        lr_sched = "constant"
    else:
        # よく分からない場合は safe default
        lr_sched = "constant"

    if not warm:
        m = re.search(r"warm(\d+)", run_id)
        if m:
            warm = m.group(1)
        else:
            warm = "0"  # warm が id に無ければ 0 とする

    return lr_sched, warm

for idx, row in enumerate(top5, start=1):
    run_id = row.get("id") or f"run_{idx}"
    # beta, alpha, gamma, eps, lr は列があればそれを使い、なければ id からパース
    beta  = (row.get("beta")  or "").strip()
    alpha = (row.get("alpha") or "").strip()
    gamma = (row.get("gamma") or "").strip()
    eps   = (row.get("eps")   or "").strip()
    lr    = (row.get("lr")    or "").strip()

    # id 例: b0.995_a0.5_g1.8_e0.02_bs16384_lr0.00003_schcosine_warm5_seed1
    if not beta:
        beta  = parse_from_id(r"b([0-9\.]+)", run_id)
    if not alpha:
        alpha = parse_from_id(r"a([0-9\.]+)", run_id)
    if not gamma:
        gamma = parse_from_id(r"g([0-9\.]+)", run_id)
    if not eps:
        # e0 or e0.02 のような形を想定
        eps   = parse_from_id(r"e([0-9\.]+)", run_id)
    if not lr:
        lr    = parse_from_id(r"lr([0-9\.eE\-]+)", run_id)

    lr_sched, warm = parse_sched_and_warm(run_id, row)

    export_dir = f"runs/finalists/{run_id}"

    cmd = [
        "python3", "tools/train_multitask.py", "data",
        "--backend", "torch", "--device", "cuda", "--compile", "--tf32", "tf32",
        "--batch-size", "16384", "--eval-batch-size", "16384", "--max-batch", "262144",
        "--epochs", "40", "--patience", "15",
        "--condition-heads", "interleave", "--condition-encoding", "onehot",
        "--init-from", INIT_FROM,
        "--loss-focal-heads", "reorder", "--focal-gamma", gamma, "--focal-alpha", alpha,
        "--class-balance-heads", "reorder", "--class-balance-beta", beta,
        "--epsilon-soft-by-head", f"reorder={eps},default=0.05",
        "--lr", lr, "--lr-schedule", lr_sched, "--warmup-epochs", warm,
        "--head-loss-weights", "predictor=0,filter_primary=0,filter_secondary=0,reorder=1.8,interleave=0",
        "--export-dir", export_dir, "--max-threads", "12", "--seed", "1", "--progress",
    ]

    print(f"\n=== [{idx}/5] Running finalist: {run_id} ===")
    print("CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)

PY
