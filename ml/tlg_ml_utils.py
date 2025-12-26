#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LABEL_RECORD_SIZE = 128
LABEL_RECORD_STRUCT = struct.Struct("<IHH12hI92s")
LABEL_MAGIC = 0x4C424C38  # "LBL8"
LABEL_VERSION = 1
LABEL_FIELDS = {
    "best_predictor": 0,
    "best_filter_perm": 1,
    "best_filter_primary": 2,
    "best_filter_secondary": 3,
    "best_reorder": 4,
    "best_interleave": 5,
    "second_predictor": 6,
    "second_filter_perm": 7,
    "second_filter_primary": 8,
    "second_filter_secondary": 9,
    "second_reorder": 10,
    "second_interleave": 11,
}
LABEL_RANGES = {
    "predictor": (0, 7),
    "filter_perm": (0, 5),
    "filter_primary": (0, 3),
    "filter_secondary": (0, 3),
    "reorder": (0, 7),
    "interleave": (0, 1),
}

BLOCK_W = 8
BLOCK_H = 8
RAW_RGB_DIMS = BLOCK_W * BLOCK_H * 3

HEADS = {
    "predictor": 8,
    "cf_perm": 6,
    "cf_primary": 4,
    "cf_secondary": 4,
    "reorder": 8,
    "interleave": 2,
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_label_value(name: str, value: int, allow_minus1: bool) -> None:
    if allow_minus1 and value == -1:
        return
    min_v, max_v = LABEL_RANGES[name]
    if not (min_v <= value <= max_v):
        raise ValueError(f"{name} out of range: {value} expected=[{min_v}..{max_v}]")


def read_label_record(fp) -> dict[str, int]:
    chunk = fp.read(LABEL_RECORD_SIZE)
    if len(chunk) != LABEL_RECORD_SIZE:
        raise ValueError("labels.all.bin ended early")
    magic, version, reserved, *rest = LABEL_RECORD_STRUCT.unpack(chunk)
    labels = rest[:12]
    if magic != LABEL_MAGIC:
        raise ValueError(f"LabelRecord magic mismatch: 0x{magic:08x}")
    if version != LABEL_VERSION:
        raise ValueError(f"LabelRecord version unsupported: {version}")
    if reserved != 0:
        raise ValueError("LabelRecord reserved field is non-zero")

    out: dict[str, int] = {}
    for name, idx in LABEL_FIELDS.items():
        out[name] = int(labels[idx])

    validate_label_value("predictor", out["best_predictor"], allow_minus1=False)
    validate_label_value("filter_perm", out["best_filter_perm"], allow_minus1=False)
    validate_label_value("filter_primary", out["best_filter_primary"], allow_minus1=False)
    validate_label_value("filter_secondary", out["best_filter_secondary"], allow_minus1=False)
    validate_label_value("reorder", out["best_reorder"], allow_minus1=False)
    validate_label_value("interleave", out["best_interleave"], allow_minus1=False)

    validate_label_value("predictor", out["second_predictor"], allow_minus1=True)
    validate_label_value("filter_perm", out["second_filter_perm"], allow_minus1=True)
    validate_label_value("filter_primary", out["second_filter_primary"], allow_minus1=True)
    validate_label_value("filter_secondary", out["second_filter_secondary"], allow_minus1=True)
    validate_label_value("reorder", out["second_reorder"], allow_minus1=True)
    validate_label_value("interleave", out["second_interleave"], allow_minus1=True)
    return out


def split_filter(code: int) -> tuple[int, int, int]:
    perm = ((code >> 4) & 0x7) % 6
    primary = ((code >> 2) & 0x3) % 4
    secondary = (code & 0x3) % 4
    return perm, primary, secondary


def join_filter(perm: int, primary: int, secondary: int) -> int:
    return (perm << 4) | (primary << 2) | secondary


def build_filter_code_map() -> dict[tuple[int, int, int], int]:
    mapping: dict[tuple[int, int, int], int] = {}
    for code in range(96):
        key = split_filter(code)
        if key not in mapping:
            mapping[key] = code
    return mapping


def extract_pixels_rgb(row: dict[str, Any]) -> np.ndarray:
    block_w = int(row["block_size"][0])
    block_h = int(row["block_size"][1])
    if block_w != BLOCK_W or block_h != BLOCK_H:
        raise ValueError(f"block_size must be [8,8], got {(block_w, block_h)}")
    components = int(row["components"])
    if components not in (3, 4):
        raise ValueError(f"components must be 3 or 4, got {components}")
    pixels = row["pixels"]
    expected_len = block_w * block_h * components
    if len(pixels) != expected_len:
        raise ValueError(f"pixels length mismatch: {len(pixels)} expected={expected_len}")

    out = np.empty(RAW_RGB_DIMS, dtype=np.float32)
    inv255 = 1.0 / 255.0
    idx = 0
    for yy in range(block_h):
        for xx in range(block_w):
            base = (yy * block_w + xx) * components
            if components == 3:
                r = float(pixels[base + 0]) * inv255
                g = float(pixels[base + 1]) * inv255
                b = float(pixels[base + 2]) * inv255
            else:
                r = float(pixels[base + 1]) * inv255
                g = float(pixels[base + 2]) * inv255
                b = float(pixels[base + 3]) * inv255
            out[idx] = r
            out[idx + 1] = g
            out[idx + 2] = b
            idx += 3
    return out


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_numeric_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(is_numeric(v) for v in value)


def collect_numeric_schema(
    row: dict[str, Any],
    schema: dict[str, int],
    invalid_keys: set[str],
    ignore_keys: Iterable[str],
) -> None:
    for key, value in row.items():
        if key in ignore_keys or key in invalid_keys:
            continue
        if is_numeric(value):
            if key not in schema:
                schema[key] = 1
            elif schema[key] != 1:
                invalid_keys.add(key)
                schema.pop(key, None)
        elif is_numeric_list(value):
            length = len(value)
            if key not in schema:
                schema[key] = length
            elif schema[key] != length:
                invalid_keys.add(key)
                schema.pop(key, None)


def build_raw_feature_names(schema: dict[str, int]) -> list[str]:
    names: list[str] = []
    for key in sorted(schema.keys()):
        length = schema[key]
        if length == 1:
            names.append(key)
        else:
            for idx in range(length):
                names.append(f"{key}[{idx}]")
    return names


def extract_numeric_row(row: dict[str, Any], schema: dict[str, int]) -> np.ndarray:
    values: list[float] = []
    for key in sorted(schema.keys()):
        length = schema[key]
        value = row.get(key)
        if length == 1:
            if is_numeric(value):
                values.append(float(value))
            else:
                values.append(0.0)
        else:
            if not is_numeric_list(value) or len(value) != length:
                values.extend([0.0] * length)
            else:
                values.extend(float(v) for v in value)
    return np.asarray(values, dtype=np.float32)


def load_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def topk_with_tiebreak(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    entries = [(idx, float(score)) for idx, score in enumerate(scores)]
    entries.sort(key=lambda x: (-x[1], x[0]))
    return entries[:k]


def soft_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    max_logits = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    log_probs = logits - max_logits - np.log(exp.sum(axis=1, keepdims=True))
    loss = -(targets * log_probs).sum(axis=1).mean()
    return float(loss)


def clamp_positive(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, None)


def apply_transform(x: np.ndarray, kind: str, *, clip_min: float | None = None, clip_max: float | None = None) -> np.ndarray:
    if kind == "log1p":
        return np.log1p(clamp_positive(x))
    if kind == "sqrt":
        return np.sqrt(clamp_positive(x))
    if kind == "square":
        return np.square(x)
    if kind == "abs":
        return np.abs(x)
    if kind == "clip":
        lo = -math.inf if clip_min is None else clip_min
        hi = math.inf if clip_max is None else clip_max
        return np.clip(x, lo, hi)
    raise ValueError(f"unknown transform: {kind}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
