#!/usr/bin/env python3

import argparse
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
        "--image-dir",
        type=Path,
        default=Path("test") / "images",
        help="Directory containing BMP images to evaluate (default: %(default)s)",
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
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Number of random BMP images to evaluate per round (default: %(default)s)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=6,
        help="Number of mutation attempts per round (default: %(default)s)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Maximum generations per attempt before giving up (default: %(default)s)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of tlgconv processes to run in parallel (default: %(default)s)",
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
    max_parallel: int,
) -> int:
    with tempfile.TemporaryDirectory(prefix="golomb_eval_", dir=work_dir) as tmp_root:
        tmp_path = Path(tmp_root)
        table_path = tmp_path / "golomb_table.txt"
        write_table(table_path, table)
        total_bits = 0

        def run_encoder(image: Path) -> int:
            output_path = tmp_path / (image.stem + ".tlg8")
            cmd = [
                str(binary),
                str(image),
                str(output_path),
                "--tlg-version=8",
                f"--tlg8-golomb-table={table_path}",
                "--tlg8-golomb-adaptive-update=off",
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
            return entropy_bits

        if max_parallel <= 1:
            for image in images:
                total_bits += run_encoder(image)
            return total_bits

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(run_encoder, image) for image in images]
            for future in as_completed(futures):
                total_bits += future.result()
        return total_bits


def ensure_images_exist(images: Iterable[Path]) -> List[Path]:
    resolved = sorted(images)
    if not resolved:
        raise FileNotFoundError("No BMP files found in the specified directory")
    return resolved


def choose_initial_images(all_images: Sequence[Path], count: int, rng: random.Random) -> List[Path]:
    pool = list(all_images)
    if count >= len(pool):
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, count)


def replace_image_subset(
    all_images: Sequence[Path],
    current: List[Path],
    rng: random.Random,
) -> List[Tuple[Path, Path]]:
    if not current:
        return []
    replace_count = max(1, len(current) // 2)
    indices = sorted(rng.sample(range(len(current)), min(replace_count, len(current))))
    remaining = [image for idx, image in enumerate(current) if idx not in indices]
    available_pool = [image for image in all_images if image not in remaining]
    replacements = []
    for idx in indices:
        if not available_pool:
            available_pool = list(all_images)
        old_image = current[idx]
        candidates = [image for image in available_pool if image != old_image]
        if not candidates:
            candidates = [image for image in all_images if image != old_image]
        if not candidates:
            candidates = list(all_images)
        new_image = rng.choice(candidates)
        if new_image in available_pool:
            available_pool.remove(new_image)
        current[idx] = new_image
        replacements.append((old_image, new_image))
    return replacements


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

    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("iterations must be positive when specified")
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    if args.attempts <= 0:
        raise ValueError("attempts must be positive")
    if args.max_generations <= 0:
        raise ValueError("max-generations must be positive")
    if args.max_parallel <= 0:
        raise ValueError("max-parallel must be positive")

    rng = random.Random(args.seed)

    image_dir = resolve_path(project_root, args.image_dir)
    images = ensure_images_exist(image_dir.glob("*.bmp"))
    active_images = choose_initial_images(images, args.subset_size, rng)

    best_table = [row[:] for row in seed_table]
    best_bits = evaluate_table(
        binary_path,
        active_images,
        best_table,
        work_dir,
        args.fast_mode,
        args.max_parallel,
    )
    write_table(best_table_path, best_table)

    print("Initial selection: " + ", ".join(image.name for image in active_images))
    print(f"Initial total entropy bits: {best_bits}")
    print(f"Best table saved to {best_table_path}")

    iteration = 0
    try:
        while args.iterations is None or iteration < args.iterations:
            iteration += 1
            print(f"[round {iteration}] evaluating {len(active_images)} images")
            for attempt in range(1, args.attempts + 1):
                print(f"  [attempt {attempt}] start")
                search_table = [row[:] for row in best_table]
                search_bits = best_bits
                success = False
                for generation in range(1, args.max_generations + 1):
                    candidate_table = mutate_table(search_table, rng)
                    candidate_bits = evaluate_table(
                        binary_path,
                        active_images,
                        candidate_table,
                        work_dir,
                        args.fast_mode,
                        args.max_parallel,
                    )
                    if candidate_bits < best_bits:
                        best_bits = candidate_bits
                        best_table = [row[:] for row in candidate_table]
                        search_table = [row[:] for row in candidate_table]
                        write_table(best_table_path, best_table)
                        print(
                            "    [generation {gen}] improvement: best_bits={bits}".format(
                                gen=generation,
                                bits=best_bits,
                            )
                        )
                        success = True
                        break
                    if candidate_bits < search_bits:
                        search_table = [row[:] for row in candidate_table]
                        search_bits = candidate_bits
                        print(
                            "    [generation {gen}] new search seed (bits={bits})".format(
                                gen=generation,
                                bits=candidate_bits,
                            )
                        )
                if not success:
                    print(
                        "  [attempt {idx}] no improvement after {gen} generations".format(
                            idx=attempt,
                            gen=args.max_generations,
                        )
                    )

            replacements = replace_image_subset(images, active_images, rng)
            if replacements:
                print("[round {idx}] replaced images:".format(idx=iteration))
                for old_image, new_image in replacements:
                    print(
                        "    {old} -> {new}".format(
                            old=old_image.name,
                            new=new_image.name,
                        )
                    )
            else:
                print(f"[round {iteration}] no images available for replacement")

            print(
                "[round {idx}] current selection: {names}".format(
                    idx=iteration,
                    names=", ".join(image.name for image in active_images),
                )
            )
            best_bits = evaluate_table(
                binary_path,
                active_images,
                best_table,
                work_dir,
                args.fast_mode,
                args.max_parallel,
            )
            print(f"[round {iteration}] baseline bits with current selection: {best_bits}")
    except KeyboardInterrupt:
        print("Interrupted by user; exiting.")


if __name__ == "__main__":
    main()
