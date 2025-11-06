#!/usr/bin/env python3
"""reorder ヘッド向けハイパーパラメータスイープスクリプト。"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# スイープ対象のハイパーパラメータグリッド
GRID = {
    "beta": [0.999, 0.997, 0.995, 0.993],
    "alpha": [0.25, 0.35, 0.50],
    "eps": [0.01, 0.02, 0.03],
    "batch": [16384, 32768],
}

# 学習設定の共通パラメータ
FIXED = {
    "epochs": 20,
    "patience": 7,
    "lr": 3e-5,
    "weight_decay": 2e-4,
    "dropout": 0.2,
}

# ログ解析用の正規表現
EPOCH_LINE = re.compile(
    r"^epoch\s*(\d+):.*val_top1=(\d+\.\d+)%.*val_top3=(\d+\.\d+)%", re.IGNORECASE
)
REORDER_LINE = re.compile(
    r"^\s*reorder\s*:\s*top1=(\d+\.\d+)%\s*top2=(\d+\.\d+)%\s*top3=(\d+\.\d+)%", re.IGNORECASE
)
MACRO_LINE = re.compile(r"^\s*macro\s*:\s*top1=(\d+\.\d+)%", re.IGNORECASE)


def short_float(value: float) -> str:
    """ファイル名向けに小数を短く整形する。"""

    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="reorder ヘッドのハイパーパラメータスイープを実行し、集計結果を生成する"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data"],
        help="train_multitask.py に渡す入力パス群",
    )
    parser.add_argument(
        "--tag",
        default="reorder_v1",
        help="出力先サブディレクトリ (runs/sweeps/<tag>)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="実行する組み合わせ数の上限 (デバッグ用)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを表示するだけで実行しない",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="train_multitask.py を呼び出す Python 実行ファイル",
    )
    return parser.parse_args()


def iter_grid() -> Iterable[Tuple[float, float, float, int]]:
    """グリッド検索の全組み合わせを返す。"""

    return itertools.product(GRID["beta"], GRID["alpha"], GRID["eps"], GRID["batch"])


def run_command(cmd: List[str], log_path: Path) -> int:
    """学習コマンドを実行し、出力をログに保存する。"""

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        return proc.returncode


def parse_log(log_path: Path) -> Dict[str, Optional[object]]:
    """学習ログから評価指標とベストエポックを抽出する。"""

    reorder_top1: Optional[float] = None
    reorder_top3: Optional[float] = None
    macro_top1: Optional[float] = None
    best_epoch: Optional[int] = None
    best_val_top3 = -1.0
    last_epoch: Optional[int] = None

    with log_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            match_epoch = EPOCH_LINE.search(line)
            if match_epoch:
                epoch = int(match_epoch.group(1))
                last_epoch = epoch
                val_top1 = float(match_epoch.group(2))
                val_top3 = float(match_epoch.group(3))
                if val_top3 > best_val_top3:
                    best_val_top3 = val_top3
                    best_epoch = epoch
            match_reorder = REORDER_LINE.search(line)
            if match_reorder:
                reorder_top1 = float(match_reorder.group(1))
                reorder_top3 = float(match_reorder.group(3))
            match_macro = MACRO_LINE.search(line)
            if match_macro:
                macro_top1 = float(match_macro.group(1))

    if best_epoch is None:
        best_epoch = last_epoch

    return {
        "val_reorder_top1": reorder_top1,
        "val_reorder_top3": reorder_top3,
        "val_macro_top1": macro_top1,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }


def format_percent(value: object) -> str:
    """Markdown 表向けにパーセンテージを整形する。"""

    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.2f}%"


def main() -> None:
    """スイープ処理のエントリーポイント。"""

    args = parse_args()
    base_dir = Path("runs") / "sweeps" / args.tag
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "results.csv"
    md_path = base_dir / "results.md"

    combinations = list(iter_grid())
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]

    rows: List[Dict[str, object]] = []

    for beta, alpha, eps, batch in combinations:
        run_id = f"b{short_float(beta)}_a{short_float(alpha)}_e{short_float(eps)}_bs{batch}"
        log_path = log_dir / f"{run_id}.log"
        export_dir = base_dir / run_id
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            "tools/train_multitask.py",
            *args.inputs,
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--compile",
            "--batch-size",
            str(batch),
            "--eval-batch-size",
            str(batch),
            "--max-batch",
            "262144",
            "--epochs",
            str(FIXED["epochs"]),
            "--patience",
            str(FIXED["patience"]),
            "--lr",
            f"{FIXED['lr']:.6g}",
            "--weight-decay",
            f"{FIXED['weight_decay']:.6g}",
            "--dropout",
            f"{FIXED['dropout']:.2f}",
            "--condition-heads",
            "interleave",
            "--condition-encoding",
            "onehot",
            "--init-from",
            "runs/ranker_R0_reorder_only_v2/multitask_best.pt",
            "--epsilon-soft-by-head",
            f"reorder={eps},default=0.05",
            "--loss-focal-heads",
            "reorder",
            "--focal-gamma",
            "1.8",
            "--focal-alpha",
            f"{alpha}",
            "--class-balance-heads",
            "reorder",
            "--class-balance-beta",
            f"{beta}",
            "--head-loss-weights",
            "predictor=0,filter_primary=0,filter_secondary=0,reorder=1.8,interleave=0",
            "--export-dir",
            str(export_dir),
            "--max-threads",
            "12",
            "--progress",
        ]

        if args.dry_run:
            print("[DRY-RUN]", " ".join(cmd))
            continue

        print(f"[RUN] {run_id}: start")
        start = time.monotonic()
        returncode = run_command(cmd, log_path)
        wall = time.monotonic() - start
        metrics = parse_log(log_path)
        row: Dict[str, object] = {
            "run_id": run_id,
            "beta": beta,
            "alpha": alpha,
            "eps": eps,
            "batch": batch,
            "lr": FIXED["lr"],
            "val_reorder_top1": metrics["val_reorder_top1"],
            "val_reorder_top3": metrics["val_reorder_top3"],
            "val_macro_top1": metrics["val_macro_top1"],
            "best_epoch": metrics["best_epoch"],
            "wall_time_sec": wall,
        }
        if returncode != 0:
            print(f"[WARN] {run_id} は returncode={returncode} で終了しました")
        else:
            print(f"[DONE] {run_id}: {wall:.1f}s")
        rows.append(row)

    if not rows:
        print("実行されたジョブがありません")
        return

    fieldnames = [
        "run_id",
        "beta",
        "alpha",
        "eps",
        "batch",
        "lr",
        "val_reorder_top1",
        "val_reorder_top3",
        "val_macro_top1",
        "best_epoch",
        "wall_time_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def sort_key(row: Dict[str, object]) -> Tuple[float, float, int]:
        top1_raw = row.get("val_reorder_top1")
        top3_raw = row.get("val_reorder_top3")
        epoch_raw = row.get("best_epoch")
        top1_val = float(top1_raw) if isinstance(top1_raw, (int, float)) else -1e9
        top3_val = float(top3_raw) if isinstance(top3_raw, (int, float)) else -1e9
        epoch_val = int(epoch_raw) if isinstance(epoch_raw, (int, float)) else 1_000_000
        return (-top1_val, -top3_val, epoch_val)

    sorted_rows = sorted(rows, key=sort_key)

    md_lines = [
        f"# Sweep Results — {datetime.utcnow().isoformat()}Z",
        "",
        "**Grid:** "
        + ", ".join(
            [
                f"beta={GRID['beta']}",
                f"alpha={GRID['alpha']}",
                f"eps={GRID['eps']}",
                f"batch={GRID['batch']}",
            ]
        ),
        "",
        "**Fixed:** "
        + ", ".join(
            [
                f"epochs={FIXED['epochs']}",
                f"patience={FIXED['patience']}",
                f"lr={FIXED['lr']}",
                f"weight_decay={FIXED['weight_decay']}",
                f"dropout={FIXED['dropout']}",
            ]
        ),
        "",
        "| run_id | beta | alpha | eps | batch | val_reorder_top1 | val_reorder_top3 | val_macro_top1 | best_epoch | wall_time_sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        md_lines.append(
            "| {run_id} | {beta:.3f} | {alpha:.2f} | {eps:.2f} | {batch:.0f} | {top1} | {top3} | {macro} | {epoch} | {wall:.2f} |".format(
                run_id=row["run_id"],
                beta=row["beta"],
                alpha=row["alpha"],
                eps=row["eps"],
                batch=row["batch"],
                top1=format_percent(row["val_reorder_top1"]),
                top3=format_percent(row["val_reorder_top3"]),
                macro=format_percent(row["val_macro_top1"]),
                epoch="-" if row["best_epoch"] is None else f"{int(row['best_epoch'])}",
                wall=row["wall_time_sec"] if row["wall_time_sec"] is not None else 0.0,
            )
        )

    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines))

    print(f"CSV を {csv_path} に保存しました")
    print(f"Markdown を {md_path} に保存しました")


if __name__ == "__main__":
    main()
