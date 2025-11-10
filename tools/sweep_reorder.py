#!/usr/bin/env python3
"""reorder ヘッド向けのハイパーパラメータスイープ (v2)。"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# スイープ対象のハイパーパラメータグリッド (v2)
GRID: Dict[str, List[Any]] = {
    "beta": [0.9999, 0.999, 0.997, 0.995, 0.993, 0.99, 0.98],
    "alpha": [0.10, 0.25, 0.35, 0.50, 0.75, 1.00],
    "gamma": [1.5, 1.8, 2.0, 2.5],
    "eps": [0.00, 0.01, 0.02, 0.03, 0.05, 0.10],
    "batch": [16384, 32768, 65536],
    "sched": ["cosine_warm5", "cosine_warm3", "constant"],
    "lr": [3e-5, 5e-5],
    "seed": [1],
}

# CLI override 用の変換関数
GRID_PARSERS: Dict[str, Any] = {
    "beta": float,
    "alpha": float,
    "gamma": float,
    "eps": float,
    "batch": int,
    "sched": str,
    "lr": float,
    "seed": int,
}

# 学習共通設定
EPOCHS = 20
PATIENCE = 7
WEIGHT_DECAY = 2e-4
DROPOUT = 0.2
HEAD_LOSS_WEIGHTS = "predictor=0,filter_primary=0,filter_secondary=0,reorder=1.8,interleave=0"
INIT_FROM = "runs/ranker_R0_reorder_only_v2/multitask_best.pt"
MAX_BATCH = 262144

# OOM 判定用キーワード
OOM_PATTERNS = [
    re.compile(r"cuda\s+out\s+of\s+memory", re.IGNORECASE),
    re.compile(r"cublas.*alloc", re.IGNORECASE),
    re.compile(r"hip\s+out\s+of\s+memory", re.IGNORECASE),
]

# ログ解析用の正規表現
METRIC_REORDER = re.compile(
    r"^\s*reorder\s*:\s*top1=(\d+\.\d+)%\s*top2=(\d+\.\d+)%\s*top3=(\d+\.\d+)%",
    re.IGNORECASE,
)
VAL_LINE = re.compile(
    r"^epoch\s*(\d+):.*val_top1=(\d+\.\d+)%.*val_top3=(\d+\.\d+)%",
    re.IGNORECASE,
)
BEST_EPOCH = re.compile(r"^early stop.*best at epoch\s*(\d+)", re.IGNORECASE)
MACRO_LINE = re.compile(r"^\s*macro\s*:\s*top1=(\d+\.\d+)%", re.IGNORECASE)

# CSV 出力カラム
CSV_COLUMNS = [
    "run_id",
    "beta",
    "alpha",
    "gamma",
    "eps",
    "batch",
    "lr",
    "sched",
    "seed",
    "val_reorder_top1",
    "val_reorder_top3",
    "val_macro_top1",
    "best_epoch",
    "status",
    "wall_time_sec",
]

# resume 時にスキップ対象とするステータス
COMPLETED_STATUSES = {"ok", "oom"}

@dataclass(frozen=True)
class SweepParams:
    """スイープ一件分の設定。"""

    beta: float
    alpha: float
    gamma: float
    eps: float
    batch: int
    lr: float
    sched: str
    seed: int

    def with_batch(self, new_batch: int) -> "SweepParams":
        """バッチサイズのみ差し替えた新しいインスタンスを返す。"""

        return replace(self, batch=new_batch)


def short_float(value: float) -> str:
    """ファイル名に使いやすい短い浮動小数表現を返す。"""

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def sanitize_sched(value: str) -> str:
    """スケジュール名を ID 用にサニタイズする。"""

    return value.replace("-", "_")


def build_run_id(params: SweepParams) -> str:
    """ログディレクトリ用の一意な ID を生成する。"""

    return (
        f"b{short_float(params.beta)}_"
        f"a{short_float(params.alpha)}_"
        f"g{short_float(params.gamma)}_"
        f"e{short_float(params.eps)}_"
        f"bs{params.batch}_"
        f"lr{short_float(params.lr)}_"
        f"sch{sanitize_sched(params.sched)}_"
        f"seed{params.seed}"
    )


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="reorder ヘッドのハイパーパラメータスイープ (v2)"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data"],
        help="train_multitask.py に渡す入力パス群",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="train_multitask.py を呼び出す Python 実行ファイル",
    )
    parser.add_argument(
        "--tag",
        default="v2",
        help="出力をまとめるタグ (runs/sweeps/reorder_<tag>/)",
    )
    parser.add_argument(
        "--grid",
        nargs="*",
        default=[],
        metavar="KEY=V1,V2",
        help="指定したパラメータのみグリッドを上書きする (例: beta=0.9999,0.995)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="実行する組み合わせ数の上限 (スモークテスト用)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="同時実行するジョブ数 (GPU メモリと相談して設定)",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="CUDA デバイス ID をカンマ区切りで指定 (例: '0' または '0,1')",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=None,
        help="GPU セマフォのスロット数 (省略時は --gpus の ID 数)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存結果を読み込み、完了済みジョブをスキップする",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ジョブを実行せず、コマンドのみを表示する",
    )
    parser.add_argument(
        "--cuda-devices",
        nargs="*",
        default=None,
        metavar="ID",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_gpu_ids(args: argparse.Namespace) -> List[str]:
    """GPU 引数を解析して正規化した ID リストを返す。"""

    raw = (args.gpus or "").strip()
    legacy = getattr(args, "cuda_devices", None)
    if legacy:
        legacy_ids = [str(item).strip() for item in legacy if str(item).strip()]
        if legacy_ids:
            print("[WARN] --cuda-devices は非推奨です (--gpus を利用してください)")
            raw = ",".join(legacy_ids)
    ids = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return ids


def initialise_gpu_context(args: argparse.Namespace, lock_dir: Path) -> None:
    """GPU ロック設定を初期化する。"""

    gpu_ids = resolve_gpu_ids(args)
    args.gpu_ids = gpu_ids  # type: ignore[attr-defined]
    if not gpu_ids:
        args.gpu_slots_resolved = 0  # type: ignore[attr-defined]
        args.gpu_lock_dir = None  # type: ignore[attr-defined]
        return

    slots = args.gpu_slots if args.gpu_slots is not None else len(gpu_ids)
    if slots < 1:
        raise ValueError("--gpu-slots は 1 以上を指定してください")

    lock_dir.mkdir(parents=True, exist_ok=True)
    args.gpu_slots_resolved = slots  # type: ignore[attr-defined]
    args.gpu_lock_dir = lock_dir  # type: ignore[attr-defined]


def command_requests_cuda(cmd: Sequence[str]) -> bool:
    """コマンドラインが CUDA 実行を要求しているか検出する。"""

    for index, token in enumerate(cmd):
        if token == "--device" and index + 1 < len(cmd):
            target = cmd[index + 1]
            return target.startswith("cuda")
    return False


def apply_grid_overrides(grid: Dict[str, List[Any]], overrides: Sequence[str]) -> None:
    """--grid 指定を既定グリッドへ反映する。"""

    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--grid の指定が不正です: {spec}")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if key not in grid:
            raise ValueError(f"未対応のグリッドキーです: {key}")
        parser = GRID_PARSERS[key]
        values = []
        for item in filter(None, (chunk.strip() for chunk in raw_values.split(","))):
            try:
                values.append(parser(item))
            except ValueError as exc:  # 型変換失敗時
                raise ValueError(f"{key} の値を変換できません: {item}") from exc
        if not values:
            raise ValueError(f"{key} の値が空です")
        grid[key] = values


def generate_params(grid: Dict[str, List[Any]]) -> List[SweepParams]:
    """グリッドから SweepParams のリストを生成する。"""

    keys = ["beta", "alpha", "gamma", "eps", "batch", "sched", "lr", "seed"]
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combos.append(
            SweepParams(
                beta=float(values[0]),
                alpha=float(values[1]),
                gamma=float(values[2]),
                eps=float(values[3]),
                batch=int(values[4]),
                sched=str(values[5]),
                lr=float(values[6]),
                seed=int(values[7]),
            )
        )
    return combos


def next_lower_batch(current: int, candidates: Sequence[int]) -> Optional[int]:
    """候補リストから現在値より小さいバッチサイズを返す。"""

    unique = sorted(set(int(x) for x in candidates))
    if current not in unique:
        unique.append(current)
        unique.sort()
    index = unique.index(current)
    if index == 0:
        return None
    return unique[index - 1]


def build_command(args: argparse.Namespace, params: SweepParams, export_dir: Path) -> List[str]:
    """train_multitask.py のコマンド配列を構築する。"""

    cmd = [
        args.python,
        "tools/train_multitask.py",
        *args.inputs,
        "--backend",
        "torch",
        "--device",
        "cuda",
        "--compile",
        "--tf32",
        "tf32",
        "--batch-size",
        str(params.batch),
        "--eval-batch-size",
        str(params.batch),
        "--max-batch",
        str(MAX_BATCH),
        "--epochs",
        str(EPOCHS),
        "--patience",
        str(PATIENCE),
        "--lr",
        f"{params.lr:.6g}",
        "--weight-decay",
        f"{WEIGHT_DECAY:.6g}",
        "--dropout",
        f"{DROPOUT:.3f}",
        "--condition-heads",
        "interleave",
        "--condition-encoding",
        "onehot",
        "--init-from",
        INIT_FROM,
        "--epsilon-soft-by-head",
        f"reorder={params.eps},default=0.05",
        "--loss-focal-heads",
        "reorder",
        "--focal-gamma",
        f"{params.gamma}",
        "--focal-alpha",
        f"{params.alpha}",
        "--class-balance-heads",
        "reorder",
        "--class-balance-beta",
        f"{params.beta}",
        "--head-loss-weights",
        HEAD_LOSS_WEIGHTS,
        "--export-dir",
        str(export_dir),
        "--max-threads",
        "12",
        "--progress",
        "--seed",
        str(params.seed),
    ]

    if params.sched.startswith("cosine_warm"):
        warm = params.sched.replace("cosine_warm", "")
        cmd += ["--lr-schedule", "cosine", "--warmup-epochs", warm]
    elif params.sched == "constant":
        cmd += ["--lr-schedule", "constant"]
    else:
        raise ValueError(f"未対応のスケジュール指定です: {params.sched}")

    return cmd


def detect_oom(line: str) -> bool:
    """ログ一行から OOM パターンを検出する。"""

    for pattern in OOM_PATTERNS:
        if pattern.search(line):
            return True
    return False


def parse_log(log_path: Path) -> Tuple[Dict[str, Optional[float]], bool]:
    """ログファイルから評価値と OOM 兆候を抽出する。"""

    reorder_top1: Optional[float] = None
    reorder_top3: Optional[float] = None
    macro_top1: Optional[float] = None
    best_epoch: Optional[int] = None
    last_epoch: Optional[int] = None
    saw_oom = False

    with log_path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if detect_oom(line):
                saw_oom = True
            match = METRIC_REORDER.search(line)
            if match:
                reorder_top1 = float(match.group(1))
                reorder_top3 = float(match.group(3))
            match = VAL_LINE.search(line)
            if match:
                last_epoch = int(match.group(1))
            match = BEST_EPOCH.search(line)
            if match:
                best_epoch = int(match.group(1))
            match = MACRO_LINE.search(line)
            if match:
                macro_top1 = float(match.group(1))

    if best_epoch is None:
        best_epoch = last_epoch

    metrics: Dict[str, Optional[float]] = {
        "val_reorder_top1": reorder_top1,
        "val_reorder_top3": reorder_top3,
        "val_macro_top1": macro_top1,
        "best_epoch": best_epoch,
    }
    return metrics, saw_oom


def run_attempt(
    args: argparse.Namespace,
    params: SweepParams,
    base_run_id: str,
    attempt_index: int,
    export_dir: Path,
    log_dir: Path,
) -> Tuple[str, Dict[str, Optional[float]], float]:
    """単一試行を実行し、ステータス・指標・経過時間を返す。"""

    attempt_id = base_run_id if attempt_index == 0 else f"{base_run_id}_retry{attempt_index}"
    log_path = log_dir / f"{attempt_id}.log"
    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, params, export_dir)
    use_cuda = command_requests_cuda(cmd)

    if args.dry_run:
        print(f"[DRY-RUN] {attempt_id}\n  " + " ".join(cmd))
        return "ok", {
            "val_reorder_top1": None,
            "val_reorder_top3": None,
            "val_macro_top1": None,
            "best_epoch": None,
        }, 0.0

    env = os.environ.copy()
    if use_cuda and getattr(args, "gpu_ids", None):
        env["TLG_SWEEP_USE_CUDA"] = "1"
        env["TLG_SWEEP_RUN_ID"] = attempt_id
        env["TLG_SWEEP_GPU_IDS"] = ",".join(args.gpu_ids)
        env["TLG_SWEEP_GPU_SLOTS"] = str(getattr(args, "gpu_slots_resolved", 0))
        lock_dir = getattr(args, "gpu_lock_dir", None)
        if lock_dir is not None:
            env["TLG_SWEEP_GPU_LOCK"] = str(lock_dir)
    else:
        env["TLG_SWEEP_USE_CUDA"] = "0"
        env.pop("TLG_SWEEP_RUN_ID", None)
        env.pop("TLG_SWEEP_GPU_IDS", None)
        env.pop("TLG_SWEEP_GPU_SLOTS", None)
        env.pop("TLG_SWEEP_GPU_LOCK", None)

    print(f"[RUN] {attempt_id}: batch={params.batch}")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    wall_time = time.monotonic() - start
    metrics, saw_oom = parse_log(log_path)

    if proc.returncode == 0 and not saw_oom:
        status = "ok"
    elif saw_oom:
        status = "oom"
    else:
        status = "fail"

    print(f"[DONE] {attempt_id}: status={status} wall={wall_time:.1f}s")
    return status, metrics, wall_time


def run_with_retries(
    args: argparse.Namespace,
    params: SweepParams,
    export_root: Path,
    log_dir: Path,
    batch_candidates: Sequence[int],
) -> Dict[str, Any]:
    """OOM 時の再試行を含めた一件分のスイープ結果を返す。"""

    base_run_id = build_run_id(params)
    export_dir = export_root / base_run_id
    wall_total = 0.0
    current_params = params
    status = "fail"
    metrics: Dict[str, Optional[float]] = {
        "val_reorder_top1": None,
        "val_reorder_top3": None,
        "val_macro_top1": None,
        "best_epoch": None,
    }

    for attempt in range(len(set(batch_candidates)) + 1):
        status, metrics, wall = run_attempt(
            args,
            current_params,
            base_run_id,
            attempt,
            export_dir,
            log_dir,
        )
        wall_total += wall
        if status == "oom":
            next_batch = next_lower_batch(current_params.batch, batch_candidates)
            if next_batch is None or next_batch == current_params.batch:
                print(f"[OOM] {base_run_id}: これ以上バッチサイズを縮小できません")
                break
            print(
                f"[OOM] {base_run_id}: batch={current_params.batch} で OOM。"
                f" batch={next_batch} へ縮小して再試行します"
            )
            current_params = current_params.with_batch(next_batch)
            continue
        break

    row: Dict[str, Any] = {
        "run_id": base_run_id,
        "beta": params.beta,
        "alpha": params.alpha,
        "gamma": params.gamma,
        "eps": params.eps,
        "batch": current_params.batch,
        "lr": params.lr,
        "sched": params.sched,
        "seed": params.seed,
        "val_reorder_top1": metrics.get("val_reorder_top1"),
        "val_reorder_top3": metrics.get("val_reorder_top3"),
        "val_macro_top1": metrics.get("val_macro_top1"),
        "best_epoch": metrics.get("best_epoch"),
        "status": status,
        "wall_time_sec": wall_total,
    }
    return row


def read_existing(csv_path: Path) -> List[Dict[str, Any]]:
    """既存の CSV から結果を読み込む。"""

    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def merge_rows(
    existing: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """既存行と新規行を run_id 単位でマージする。"""

    merged: Dict[str, Dict[str, Any]] = {}
    for row in existing:
        merged[row["run_id"]] = row
    for row in new_rows:
        merged[row["run_id"]] = row
    return list(merged.values())


def normalise_value(value: Any) -> Any:
    """CSV 書き出し前に None を空欄へ揃える。"""

    if value is None:
        return ""
    return value


def write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """結果 CSV を保存する。"""

    rows_sorted = sorted(rows, key=lambda r: r["run_id"])
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({key: normalise_value(row.get(key)) for key in CSV_COLUMNS})


def format_percent(value: Any) -> str:
    """Markdown 用にパーセント表記へ整形する。"""

    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def get_git_commit() -> Optional[str]:
    """現在の git commit ハッシュを取得する。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        return None
    return None


def format_reproduce_command(args: argparse.Namespace, row: Dict[str, Any]) -> str:
    """ベスト設定の再現コマンドを生成する。"""

    params = SweepParams(
        beta=float(row["beta"]),
        alpha=float(row["alpha"]),
        gamma=float(row["gamma"]),
        eps=float(row["eps"]),
        batch=int(row["batch"]),
        lr=float(row["lr"]),
        sched=str(row["sched"]),
        seed=int(row["seed"]),
    )
    export_root = Path("runs") / "sweeps" / f"reorder_{args.tag}"
    export_dir = export_root / row["run_id"]
    cmd = build_command(args, params, export_dir)
    return " ".join(cmd)


def write_markdown(
    md_path: Path,
    grid: Dict[str, List[Any]],
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """結果サマリ Markdown を出力する。"""

    commit = get_git_commit()
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    total = len(rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    oom = sum(1 for r in rows if r.get("status") == "oom")
    fail = sum(1 for r in rows if r.get("status") == "fail")

    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("val_reorder_top1") not in (None, "")
    ]

    def to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, int]:
        top1 = to_float(row.get("val_reorder_top1"), -1e9)
        top3 = to_float(row.get("val_reorder_top3"), -1e9)
        epoch_val = to_int(row.get("best_epoch"), 1_000_000)
        return (-top1, -top3, epoch_val)

    ok_rows.sort(key=sort_key)
    top_rows = ok_rows[: min(10, len(ok_rows))]

    lines: List[str] = []
    lines.append(f"# reorder sweep v2 ({now})")
    lines.append("")
    lines.append(f"- タグ: `{args.tag}`")
    if commit:
        lines.append(f"- commit: `{commit}`")
    lines.append(
        "- グリッド: "
        + ", ".join(f"{key}={grid[key]}" for key in ["beta", "alpha", "gamma", "eps", "batch", "sched", "lr", "seed"])
    )
    lines.append(
        "- 共通設定: "
        f"epochs={EPOCHS}, patience={PATIENCE}, lr∈{grid['lr']}, weight_decay={WEIGHT_DECAY}, dropout={DROPOUT}"
    )
    lines.append(f"- 実行ステータス: ok={ok}, oom={oom}, fail={fail}, total={total}")
    lines.append("")

    if top_rows:
        lines.append("## Top 10 (val_reorder_top1)")
        lines.append(
            "| # | run_id | beta | alpha | gamma | eps | batch | lr | sched | seed | top1 | top3 | epoch | status |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|")
        for idx, row in enumerate(top_rows, start=1):
            lines.append(
                "| {idx} | {run_id} | {beta:.4f} | {alpha:.2f} | {gamma:.2f} | {eps:.2f} | {batch} | {lr:.6g} | {sched} | {seed} | {top1} | {top3} | {epoch} | {status} |".format(
                    idx=idx,
                    run_id=row["run_id"],
                    beta=float(row["beta"]),
                    alpha=float(row["alpha"]),
                    gamma=float(row["gamma"]),
                    eps=float(row["eps"]),
                    batch=int(row["batch"]),
                    lr=float(row["lr"]),
                    sched=row["sched"],
                    seed=int(row["seed"]),
                    top1=format_percent(row.get("val_reorder_top1")),
                    top3=format_percent(row.get("val_reorder_top3")),
                    epoch=row.get("best_epoch", "-") if row.get("best_epoch") not in (None, "") else "-",
                    status=row.get("status", "-"),
                )
            )
        lines.append("")

    best_row = ok_rows[0] if ok_rows else None
    if best_row:
        reproduce = format_reproduce_command(args, best_row)
        lines.append("## Reproduce Best Run")
        lines.append("```bash")
        lines.append(reproduce)
        lines.append("```")
        lines.append("")

    lines.append("## すべての結果要約")
    lines.append(
        "| run_id | status | beta | alpha | gamma | eps | batch | lr | sched | seed | top1 | top3 | macro | epoch | wall[s] |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    for row in sorted(rows, key=sort_key):
        lines.append(
            "| {run_id} | {status} | {beta:.4f} | {alpha:.2f} | {gamma:.2f} | {eps:.2f} | {batch} | {lr:.6g} | {sched} | {seed} | {top1} | {top3} | {macro} | {epoch} | {wall:.1f} |".format(
                run_id=row["run_id"],
                status=row.get("status", "-"),
                beta=float(row["beta"]),
                alpha=float(row["alpha"]),
                gamma=float(row["gamma"]),
                eps=float(row["eps"]),
                batch=int(row["batch"]),
                lr=float(row["lr"]),
                sched=row["sched"],
                seed=int(row["seed"]),
                top1=format_percent(row.get("val_reorder_top1")),
                top3=format_percent(row.get("val_reorder_top3")),
                macro=format_percent(row.get("val_macro_top1")),
                epoch=row.get("best_epoch", "-") if row.get("best_epoch") not in (None, "") else "-",
                wall=float(row.get("wall_time_sec") or 0.0),
            )
        )

    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main() -> None:
    """スイープのエントリーポイント。"""

    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency は 1 以上を指定してください")

    grid = {key: list(values) for key, values in GRID.items()}
    if args.grid:
        apply_grid_overrides(grid, args.grid)

    batch_candidates = grid["batch"]
    params_list = generate_params(grid)

    base_dir = Path("runs") / "sweeps" / f"reorder_{args.tag}"
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = base_dir / "gpu_lock"

    initialise_gpu_context(args, lock_dir)
    if getattr(args, "gpu_ids", None):
        print(
            f"[INFO] GPU ロック: devices={','.join(args.gpu_ids)} "
            f"slots={getattr(args, 'gpu_slots_resolved', 0)}"
        )
    else:
        print("[INFO] GPU ロック: 無効 (CUDA を使用しません)")

    csv_path = base_dir / "results_v2.csv"
    md_path = base_dir / "results_v2.md"

    existing_rows = read_existing(csv_path)
    if args.resume and existing_rows:
        completed = {
            row["run_id"]
            for row in existing_rows
            if row.get("status") in COMPLETED_STATUSES
        }
        params_list = [p for p in params_list if build_run_id(p) not in completed]

    if args.max_runs is not None:
        params_list = params_list[: args.max_runs]

    if not params_list and not args.dry_run:
        print("実行対象のジョブはありません (--resume で全件完了済みか指定ミスの可能性)")

    new_rows: List[Dict[str, Any]] = []

    if args.concurrency == 1 or args.dry_run:
        for params in params_list:
            row = run_with_retries(
                args,
                params,
                base_dir,
                log_dir,
                batch_candidates,
            )
            new_rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for params in params_list:
                futures.append(
                    executor.submit(
                        run_with_retries,
                        args,
                        params,
                        base_dir,
                        log_dir,
                        batch_candidates,
                    )
                )
            for future in as_completed(futures):
                new_rows.append(future.result())

    merged_rows = merge_rows(existing_rows, new_rows)

    if not args.dry_run:
        write_csv(csv_path, merged_rows)
        write_markdown(md_path, grid, merged_rows, args)
        print(f"[INFO] CSV を {csv_path} に保存しました")
        print(f"[INFO] Markdown を {md_path} に保存しました")


if __name__ == "__main__":
    main()
