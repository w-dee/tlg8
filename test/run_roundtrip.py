#!/usr/bin/env python3
"""
Round-trip tests for tlgconv using the sample BMP images in test/images.

For each *.bmp file, the script converts it to PNG, TLG6 and TLG8 into a temporary
folder, converts back to BMP, and checks the reconstructed BMP matches the
original via the `cmp` command.

Usage:
  python3 test/run_roundtrip.py [--tlgconv /path/to/tlgconv] [--images DIR]
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple, Sequence


def run_command(cmd, **kwargs):
    """Run a command, raising SystemExit on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        cmd_str = " ".join(str(x) for x in cmd)
        raise SystemExit(f"Command failed ({result.returncode}): {cmd_str}")
    return result


class FormatSpec(NamedTuple):
    name: str
    extension: str
    encode_args: Sequence[str]


def run_timed_command(cmd: Sequence[str], **kwargs) -> float:
    start = time.perf_counter()
    run_command(cmd, **kwargs)
    return time.perf_counter() - start


def roundtrip(
    bmp_path: Path,
    tlgconv: Path,
    temp_dir: Path,
    spec: FormatSpec,
    stats: dict[str, dict[str, float]],
) -> tuple[bool, str | None]:
    compressed_path = temp_dir / f"{bmp_path.stem}{spec.extension}"
    recon_bmp = temp_dir / f"{bmp_path.stem}.roundtrip.{spec.name}.bmp"

    encode_cmd = [str(tlgconv), str(bmp_path), str(compressed_path), *spec.encode_args]
    encode_time = run_timed_command(encode_cmd)

    original_size = bmp_path.stat().st_size
    compressed_size = compressed_path.stat().st_size
    compression_ratio = (compressed_size / original_size * 100) if original_size else 0.0

    decode_cmd = [str(tlgconv), str(compressed_path), str(recon_bmp)]
    decode_time = run_timed_command(decode_cmd)

    print(
        f"    {spec.name}: pre={original_size} bytes, post={compressed_size} bytes, "
        f"ratio={compression_ratio:.2f}%, enc={encode_time:.3f}s, dec={decode_time:.3f}s"
    )

    stat = stats[spec.name]
    stat["original"] += original_size
    stat["compressed"] += compressed_size
    stat["compress_time"] += encode_time
    stat["decompress_time"] += decode_time

    try:
        cmp_proc = subprocess.run(
            ["compare", str(bmp_path), str(recon_bmp), "null:"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise SystemExit("ImageMagick 'compare' command not found: " + str(exc)) from exc

    if cmp_proc.returncode != 0:
        return (
            False,
            f"Mismatch detected for {bmp_path.name} ({spec.name} round-trip)",
        )

    return True, None



FORMATS: tuple[FormatSpec, ...] = (
    FormatSpec("png", ".png", ()),
    FormatSpec("tlg6", ".tlg6", ("--tlg-version=6",)),
    FormatSpec("tlg8", ".tlg8", ("--tlg-version=8",)),
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Round-trip test for tlgconv")
    parser.add_argument(
        "--tlgconv",
        type=Path,
        default=Path("build") / "tlgconv",
        help="Path to the tlgconv executable (default: build/tlgconv)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("test") / "images",
        help="Directory containing BMP images to test",
    )
    args = parser.parse_args(argv)

    if not args.tlgconv.exists() or not args.tlgconv.is_file():
        raise SystemExit(f"tlgconv not found at {args.tlgconv}")
    if not args.images.exists() or not args.images.is_dir():
        raise SystemExit(f"Image directory not found: {args.images}")

    bmp_files = sorted(args.images.glob("*.bmp"))
    if not bmp_files:
        raise SystemExit(f"No BMP files found in {args.images}")

    mismatches: list[str] = []
    stats = {
        spec.name: {
            "original": 0.0,
            "compressed": 0.0,
            "compress_time": 0.0,
            "decompress_time": 0.0,
        }
        for spec in FORMATS
    }

    format_names = "/".join(spec.name for spec in FORMATS)

    with tempfile.TemporaryDirectory(prefix="tlgconv_roundtrip_") as tmp:
        tmp_path = Path(tmp)
        for bmp in bmp_files:
            print(f"[INFO] Testing {bmp.name} -> {format_names} round-trip")
            for spec in FORMATS:
                success, message = roundtrip(bmp, args.tlgconv, tmp_path, spec, stats)
                if not success and message:
                    print(f"[ERROR] {message}")
                    mismatches.append(message)

    print("[SUMMARY] Round-trip totals by format:")
    for spec in FORMATS:
        stat = stats[spec.name]
        original_total = stat["original"]
        compressed_total = stat["compressed"]
        ratio = (compressed_total / original_total * 100) if original_total else 0.0
        print(
            f"    {spec.name}: bmp_total={int(original_total)} bytes, "
            f"compressed_total={int(compressed_total)} bytes, ratio={ratio:.2f}%, "
            f"enc_total={stat['compress_time']:.3f}s, dec_total={stat['decompress_time']:.3f}s"
        )

    if mismatches:
        print("[FAILURE] Round-trip mismatches detected:")
        for message in mismatches:
            print(f"    - {message}")
        return 1

    print("[SUCCESS] All round-trip tests passed")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit as exc:
        if exc.code and exc.code != 0:
            print(exc, file=sys.stderr)
        raise
