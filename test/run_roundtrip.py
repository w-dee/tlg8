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
import filecmp
import shutil
import struct
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


COMPARE_CMD = shutil.which("compare")


def load_bmp_pixels(path: Path) -> tuple[int, int, int, bytes]:
    with path.open("rb") as fp:
        data = fp.read()

    if len(data) < 54 or data[0:2] != b"BM":
        raise ValueError("unsupported bmp format")

    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40:
        raise ValueError("unsupported bmp header")

    width = struct.unpack_from("<i", data, 18)[0]
    height_signed = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    pixel_offset = struct.unpack_from("<I", data, 10)[0]

    if planes != 1 or compression != 0:
        raise ValueError("unsupported bmp features")
    if bpp not in (24, 32):
        raise ValueError("unsupported bmp depth")

    abs_height = abs(height_signed)
    if width <= 0 or abs_height == 0:
        raise ValueError("invalid bmp dimensions")

    bytes_per_pixel = bpp // 8
    row_stride = ((width * bytes_per_pixel + 3) // 4) * 4
    bytes_per_row = width * bytes_per_pixel

    if len(data) < pixel_offset + row_stride * abs_height:
        raise ValueError("bmp payload truncated")

    pixels = bytearray(bytes_per_row * abs_height)
    bottom_up = height_signed > 0

    for row in range(abs_height):
        src_row = abs_height - 1 - row if bottom_up else row
        src_offset = pixel_offset + src_row * row_stride
        row_data = data[src_offset : src_offset + bytes_per_row]
        if len(row_data) < bytes_per_row:
            raise ValueError("bmp payload truncated")
        dest_offset = row * bytes_per_row
        pixels[dest_offset : dest_offset + bytes_per_row] = row_data

    return width, abs_height, bpp, bytes(pixels)


def bmp_pixels_match(a: Path, b: Path) -> bool:
    try:
        width_a, height_a, depth_a, pixels_a = load_bmp_pixels(a)
        width_b, height_b, depth_b, pixels_b = load_bmp_pixels(b)
    except ValueError:
        return False

    return (
        width_a == width_b
        and height_a == height_b
        and depth_a == depth_b
        and pixels_a == pixels_b
    )


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

    mismatch_message = f"Mismatch detected for {bmp_path.name} ({spec.name} round-trip)"

    if COMPARE_CMD:
        cmp_proc = subprocess.run(
            [COMPARE_CMD, str(bmp_path), str(recon_bmp), "null:"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if cmp_proc.returncode != 0:
            return False, mismatch_message
    else:
        if bmp_path.stat().st_size == recon_bmp.stat().st_size and filecmp.cmp(bmp_path, recon_bmp, shallow=False):
            return True, None
        if not bmp_pixels_match(bmp_path, recon_bmp):
            return False, mismatch_message

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
