#!/usr/bin/env python3
"""
Round-trip tests for tlgconv using the sample BMP images in test/images.

For each *.bmp file, the script converts it to TLG5, TLG6 and TLG7 into a temporary
folder, converts back to BMP, and checks the reconstructed BMP matches the
original via the `cmp` command.

Usage:
  python3 test/run_roundtrip.py [--tlgconv /path/to/tlgconv] [--images DIR]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, **kwargs):
    """Run a command, raising SystemExit on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        cmd_str = " ".join(str(x) for x in cmd)
        raise SystemExit(f"Command failed ({result.returncode}): {cmd_str}")
    return result


def roundtrip(bmp_path: Path, tlgconv: Path, temp_dir: Path, version: str) -> None:
    tlg_suffix = f".tlg{version}" if version in {"5", "6", "7"} else ".tlg"
    tlg_path = temp_dir / f"{bmp_path.stem}{tlg_suffix}"
    recon_bmp = temp_dir / f"{bmp_path.stem}.roundtrip.v{version}.bmp"

    run_command([
        str(tlgconv),
        str(bmp_path),
        str(tlg_path),
        f"--tlg-version={version}",
    ])

    original_size = bmp_path.stat().st_size
    compressed_size = tlg_path.stat().st_size
    compression_ratio = (compressed_size / original_size * 100) if original_size else 0.0
    print(
        f"    tlg{version}: pre={original_size} bytes, post={compressed_size} bytes, "
        f"ratio={compression_ratio:.2f}%"
    )

    run_command([
        str(tlgconv),
        str(tlg_path),
        str(recon_bmp),
    ])

    cmp_proc = subprocess.run(["cmp", "-s", str(bmp_path), str(recon_bmp)])
    if cmp_proc.returncode != 0:
        raise SystemExit(
            f"Mismatch detected for {bmp_path.name} (tlg{version} round-trip)"
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

    with tempfile.TemporaryDirectory(prefix="tlgconv_roundtrip_") as tmp:
        tmp_path = Path(tmp)
        for bmp in bmp_files:
            print(f"[INFO] Testing {bmp.name} -> tlg5/tlg6/tlg7 round-trip")
            for version in ("5", "6", "7"):
                roundtrip(bmp, args.tlgconv, tmp_path, version)
        print("[SUCCESS] All round-trip tests passed")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit as exc:
        if exc.code and exc.code != 0:
            print(exc, file=sys.stderr)
        raise
