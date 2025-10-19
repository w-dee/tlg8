"""低メモリでのファイル結合や検証を行うヘルパー群。"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
from pathlib import Path
from typing import List

RECORD_SIZE = 128
MAGIC = 0x4C424C38
VERSION = 1


def append_jsonl_and_count(dst: Path, src: Path, buf_size: int = 1 << 20) -> int:
    """JSONL を追記しつつ行数を数える。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with src.open("rb") as s, dst.open("ab") as d:
        while True:
            chunk = s.read(buf_size)
            if not chunk:
                break
            d.write(chunk)
            lines += chunk.count(b"\n")
    return lines


def count_lines(path: Path, buf_size: int = 1 << 20) -> int:
    """追記せずに JSONL の行数を得る。"""
    lines = 0
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(buf_size), b""):
            lines += chunk.count(b"\n")
    return lines


def validate_label_part(bin_path: Path, expected_records: int) -> None:
    """ラベルキャッシュ断片のサイズとヘッダーを検証する。"""
    size = bin_path.stat().st_size
    expected_size = expected_records * RECORD_SIZE
    if size != expected_size:
        raise RuntimeError(f"{bin_path}: size {size} != {expected_records}*{RECORD_SIZE}")
    with bin_path.open("rb") as fp:
        head = fp.read(8)
    if len(head) != 8:
        raise RuntimeError(f"{bin_path}: ヘッダーを読み取れませんでした")
    magic, ver, res = struct.unpack("<IHH", head)
    if magic != MAGIC or ver != VERSION:
        raise RuntimeError(
            f"{bin_path}: bad magic/version ({hex(magic)}, {ver})"
        )
    if res != 0:
        raise RuntimeError(f"{bin_path}: reserved フィールドが 0 ではありません: {res}")


def append_file(dst: Path, src: Path, buf_size: int = 1 << 20) -> None:
    """任意のバイナリファイルを追記する。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as s, dst.open("ab") as d:
        shutil.copyfileobj(s, d, length=buf_size)


def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    """SHA-256 を逐次的に計算する。"""
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(buf_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dataset_hash(inputs: List[dict]) -> str:
    """入力情報のリストからデータセットハッシュを求める。"""
    hasher = hashlib.sha256()
    for item in inputs:
        sha_hex = item.get("sha256")
        size = item.get("size")
        path = item.get("path")
        if not isinstance(sha_hex, str) or not isinstance(size, int) or not isinstance(path, str):
            raise RuntimeError("データセットハッシュ計算用の入力情報が不正です")
        hasher.update(bytes.fromhex(sha_hex))
        hasher.update(int(size).to_bytes(8, "little", signed=False))
        hasher.update(os.fsencode(os.path.abspath(path)))
    return hasher.hexdigest()
