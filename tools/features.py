"""特徴量抽出用の補助関数群。"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Sobel フィルタ係数（水平・垂直）
_SOBEL_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)
_SOBEL_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=np.float32)


def _sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sobel フィルタで勾配を算出する。"""

    if image.ndim != 2:
        raise ValueError("image は 2 次元配列である必要があります")
    h, w = image.shape
    if h == 0 or w == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    padded = np.pad(image, 1, mode="edge")
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            weight_x = _SOBEL_X[ky, kx]
            weight_y = _SOBEL_Y[ky, kx]
            if weight_x == 0.0 and weight_y == 0.0:
                continue
            region = padded[ky : ky + h, kx : kx + w]
            if weight_x != 0.0:
                gx += weight_x * region
            if weight_y != 0.0:
                gy += weight_y * region
    return gx, gy


def compute_orientation_features(tile: np.ndarray) -> np.ndarray:
    """平均化タイルから方位統計特徴量 (+13D) を生成する。"""

    if tile.ndim != 3:
        raise ValueError("tile は (components, H, W) 形状である必要があります")
    components, h, w = tile.shape
    if components <= 0 or h <= 0 or w <= 0:
        return np.zeros((13,), dtype=np.float32)
    mean_plane = tile[:components, :h, :w].mean(axis=0, dtype=np.float32)
    gx, gy = _sobel_gradients(mean_plane)
    if gx.size == 0 or gy.size == 0:
        return np.zeros((13,), dtype=np.float32)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    flat_mag = mag.reshape(-1)
    flat_theta = theta.reshape(-1)
    bins = np.clip(
        np.floor((flat_theta + math.pi) * (8.0 / (2.0 * math.pi))).astype(np.int64),
        0,
        7,
    )
    hist = np.bincount(bins, weights=flat_mag.astype(np.float64), minlength=8).astype(np.float32)
    total_hist = float(hist.sum())
    if total_hist > 0.0:
        hist /= total_hist
    mean_mag = float(flat_mag.mean()) if flat_mag.size else 0.0
    if flat_mag.size:
        sorted_mag = np.sort(flat_mag)
        idx = int(math.floor(0.95 * max(flat_mag.size - 1, 0)))
        idx = min(max(idx, 0), sorted_mag.size - 1)
        p95 = float(sorted_mag[idx])
        threshold = mean_mag * 1.5 if mean_mag > 0.0 else float("inf")
        if math.isfinite(threshold):
            edge_ratio = float(np.count_nonzero(flat_mag > threshold) / float(flat_mag.size))
        else:
            edge_ratio = 0.0
        quart = max(1, flat_mag.size // 4)
        low_mean = float(sorted_mag[:quart].mean()) if quart > 0 else 0.0
        high_mean = float(sorted_mag[-quart:].mean()) if quart > 0 else 0.0
        low_high_ratio = low_mean / (high_mean + 1e-6)
    else:
        p95 = 0.0
        edge_ratio = 0.0
        low_high_ratio = 0.0
    abs_gx = float(np.abs(gx).sum())
    abs_gy = float(np.abs(gy).sum())
    hv_balance = (abs_gx - abs_gy) / (abs_gx + abs_gy + 1e-6)
    features = np.concatenate(
        [
            hist,
            np.array(
                [
                    mean_mag,
                    p95,
                    edge_ratio,
                    hv_balance,
                    low_high_ratio,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return features.astype(np.float32, copy=False)
