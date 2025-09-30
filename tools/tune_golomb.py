#!/usr/bin/env python3

import argparse
import os
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_GOLOMB_TABLE: List[List[int]] = [
    [0, 4, 4, 7, 24, 89, 270, 489, 137],
    [2, 2, 5, 13, 67, 98, 230, 476, 131],
    [3, 2, 5, 15, 77, 92, 238, 462, 130],
    [2, 2, 4, 10, 51, 108, 237, 482, 128],
    [2, 3, 7, 33, 74, 81, 237, 450, 137],
    [3, 1, 5, 28, 66, 92, 246, 452, 131],
]
GOLOMB_ROW_SUM = 1024
GOLOMB_ROWS = len(DEFAULT_GOLOMB_TABLE)
GOLOMB_COLS = len(DEFAULT_GOLOMB_TABLE[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune TLG8 Golomb table with a genetic search loop")
    parser.add_argument(
        "--binary",
        default="build/tlgconv",
        help="Path to tlgconv binary (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel attempts per iteration (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Optional maximum number of iterations (default: run until interrupted)",
    )
    parser.add_argument(
        "--seed-table",
        type=Path,
        help="Initial Golomb table file to seed the search (default: built-in table or --best-table if it exists)",
    )
    parser.add_argument(
        "--best-table",
        type=Path,
        default=Path("analysis") / "golomb_best.txt",
        help="Output path to store the best Golomb table (default: %(default)s)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("analysis") / "golomb_ga_work",
        help="Directory to store temporary evaluation artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the random generator",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable --tlg8-fast when invoking the encoder",
    )
    return parser.parse_args()


def resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path)


def load_table(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].split(";", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != GOLOMB_COLS:
                raise ValueError(f"{path}: line {line_no}: expected {GOLOMB_COLS} integers, got {len(parts)}")
            try:
                values = [int(p) for p in parts]
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}: line {line_no}: invalid integer") from exc
            if any(v < 0 for v in values):
                raise ValueError(f"{path}: line {line_no}: negative values are not allowed")
            if sum(values) != GOLOMB_ROW_SUM:
                raise ValueError(f"{path}: line {line_no}: row sum must be {GOLOMB_ROW_SUM}")
            rows.append(values)

    if not rows:
        raise ValueError(f"{path}: no rows found")

    if len(rows) == GOLOMB_ROWS:
        return rows

    if GOLOMB_ROWS % len(rows) == 0:
        repeat = GOLOMB_ROWS // len(rows)
        expanded: List[List[int]] = []
        for _ in range(repeat):
            expanded.extend([row[:] for row in rows])
        return expanded

    raise ValueError(
        f"{path}: expected {GOLOMB_ROWS} rows or a divisor of that count, found {len(rows)}"
    )


def write_table(path: Path, table: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in table:
            handle.write(" ".join(str(v) for v in row))
            handle.write("\n")


def mutate_table(table: Sequence[Sequence[int]], rng: random.Random) -> List[List[int]]:
    original = [list(row) for row in table]
    attempt = 0
    while True:
        candidate = [row[:] for row in original]
        change_count = rng.randint(1, GOLOMB_ROWS)
        changed = False
        for _ in range(change_count):
            row_idx = rng.randrange(GOLOMB_ROWS)
            row = candidate[row_idx]
            positive_indices = [idx for idx, value in enumerate(row) if value > 0]
            if not positive_indices:
                continue
            src_idx = rng.choice(positive_indices)
            dst_idx = rng.randrange(GOLOMB_COLS - 1)
            if dst_idx >= src_idx:
                dst_idx += 1
            row[src_idx] -= 1
            row[dst_idx] += 1
            changed = True
        if changed and candidate != original:
            return candidate
        attempt += 1
        if attempt >= 100:
            # Force a minimal change to guarantee progress.
            forced = [row[:] for row in original]
            row = forced[attempt % GOLOMB_ROWS]
            donors = [idx for idx, value in enumerate(row) if value > 0]
            if not donors:
                attempt += 1
                continue
            src_idx = rng.choice(donors)
            dst_idx = (src_idx + 1) % GOLOMB_COLS
            row[src_idx] -= 1
            row[dst_idx] += 1
            return forced


def evaluate_table(
    binary: Path,
    images: Sequence[Path],
    table: Sequence[Sequence[int]],
    work_dir: Path,
    fast_mode: bool,
) -> int:
    with tempfile.TemporaryDirectory(prefix="golomb_eval_", dir=work_dir) as tmp_root:
        tmp_path = Path(tmp_root)
        table_path = tmp_path / "golomb_table.txt"
        write_table(table_path, table)
        total_bits = 0
        for image in images:
            output_path = tmp_path / (image.stem + ".tlg8")
            cmd = [
                str(binary),
                str(image),
                str(output_path),
                "--tlg-version=8",
                f"--tlg8-golomb-table={table_path}",
                "--print-entropy-bits",
            ]
            if fast_mode:
                cmd.append("--tlg8-fast")
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                stdout = result.stdout.decode(errors="ignore").strip()
                stderr = result.stderr.decode(errors="ignore").strip()
                raise RuntimeError(
                    f"Encoding failed for {image.name} with return code {result.returncode}:\n"
                    f"STDOUT: {stdout}\nSTDERR: {stderr}"
                )
            stdout = result.stdout.decode(errors="ignore")
            entropy_bits = None
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("entropy_bits="):
                    try:
                        entropy_bits = int(line.split("=", 1)[1])
                    except ValueError as exc:  # pragma: no cover - defensive
                        raise RuntimeError(f"Failed to parse entropy bits for {image.name}: {line}") from exc
                    break
            if entropy_bits is None:
                raise RuntimeError(
                    f"Encoder output for {image.name} did not contain entropy_bits information:\n{stdout.strip()}"
                )
            total_bits += entropy_bits
        return total_bits


def ensure_images_exist(images: Iterable[Path]) -> List[Path]:
    resolved = sorted(images)
    if not resolved:
        raise FileNotFoundError("No BMP files found under test/images")
    return resolved


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    binary_path = resolve_path(project_root, Path(args.binary))
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    work_dir = resolve_path(project_root, args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    best_table_path = resolve_path(project_root, args.best_table)
    best_table_path.parent.mkdir(parents=True, exist_ok=True)

    seed_table: List[List[int]]
    if args.seed_table:
        seed_path = resolve_path(project_root, args.seed_table)
        seed_table = load_table(seed_path)
    elif best_table_path.exists():
        seed_table = load_table(best_table_path)
    else:
        seed_table = [row[:] for row in DEFAULT_GOLOMB_TABLE]

    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("iterations must be positive when specified")

    rng = random.Random(args.seed)

    image_dir = project_root / "test" / "images"
    images = ensure_images_exist(image_dir.glob("*.bmp"))

    best_table = [row[:] for row in seed_table]
    best_bits = evaluate_table(binary_path, images, best_table, work_dir, args.fast_mode)
    search_table = [row[:] for row in best_table]
    bits_initial = best_bits
    write_table(best_table_path, best_table)

    print(f"Initial total entropy bits: {bits_initial}")
    print(f"Best table saved to {best_table_path}")

    iteration = 0
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            while args.iterations is None or iteration < args.iterations:
                iteration += 1
                futures = []
                for _ in range(args.threads):
                    candidate_table = mutate_table(search_table, rng)
                    futures.append((executor.submit(
                        evaluate_table,
                        binary_path,
                        images,
                        [row[:] for row in candidate_table],
                        work_dir,
                        args.fast_mode,
                    ), candidate_table))

                evaluated = []
                for future, candidate_table in futures:
                    try:
                        candidate_bits = future.result()
                    except Exception as exc:
                        print(f"[iter {iteration}] candidate failed: {exc}")
                        continue
                    evaluated.append((candidate_bits, candidate_table))

                if not evaluated:
                    raise RuntimeError("All candidate evaluations failed in this iteration")

                evaluated.sort(key=lambda item: item[0])
                best_attempt_bits, best_attempt_table = evaluated[0]

                improvements = [entry for entry in evaluated if entry[0] < best_bits]
                non_worse = [entry for entry in evaluated if entry[0] <= best_bits]

                if improvements:
                    candidate_bits, candidate_table = min(improvements, key=lambda item: item[0])
                    best_bits = candidate_bits
                    best_table = [row[:] for row in candidate_table]
                    search_table = [row[:] for row in candidate_table]
                    write_table(best_table_path, best_table)
                    print(
                        f"[iter {iteration}] improvement: best_bits={best_bits} "
                        f"(delta={best_bits - bits_initial})"
                    )
                else:
                    print(
                        f"[iter {iteration}] no improvement (best_candidate_bits={best_attempt_bits}, best_bits={best_bits})"
                    )
                    if non_worse:
                        candidate_bits, candidate_table = rng.choice(non_worse)
                        best_table = [row[:] for row in candidate_table]
                        search_table = [row[:] for row in candidate_table]
                        write_table(best_table_path, best_table)
                        print(
                            f"[iter {iteration}] adopting equal-or-better candidate as new seed (bits={candidate_bits})"
                        )
                    else:
                        search_table = [row[:] for row in best_table]

                print(
                    f"[iter {iteration}] bits_initial={bits_initial}, current_best_bits={best_bits}"
                )
    except KeyboardInterrupt:
        print("Interrupted by user; exiting.")


if __name__ == "__main__":
    main()
