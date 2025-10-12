#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build allocation vectors (length=9, sum=1024) per context r=0..7 from residual dump CSV (r,a,k,m).

Pipeline (per r):
  1) Parse robustly (regex) rows "r,a,k,m".
  2) For each observed a, choose best integer k in [0..15] that minimizes (truncated) Rice code length.
  3) Fill k[a] to full 0..1023 using a hybrid scheme:
     - Outside observed range: use default allocation expanded to default_k[a].
     - Between observed anchors: stepwise monotone interpolation (staircase; |Δk|≤1 per step).
  4) Optional light smoothing: Lipschitz(|Δk| ≤ L).
  5) Clip to [0..8], count occurrences of 0..8 → allocation vector (sum=1024).
  6) Fallback: if a context has no/sparse data, emit DEFAULT_ALLOC as-is.

Notes:
  - Output is NOT a histogram of k itself; it's the compressed representation of monotone k[a].
  - Expanding allocation c0..c8 as [0]*c0 + [1]*c1 + ... yields the final k[a] table (length 1024).
"""

import sys
import argparse
import re
import numpy as np

# ---------- Global config ----------
LUT_SIZE     = 1024
R_CLASSES    = 8
K_MIN, K_MAX = 0, 15

# Truncated Rice settings (match encoder side if enabled there)
USE_TRUNC_RICE = True
TRUNC_L        = 7
ESC_BITS       = 9

# Prior (disabled by default; turn on if you want small-a bias)
LAMBDA0   = 0.0
GAMMA     = 0.35
A0        = 64
BETA      = 1.0

# Smoothing
ENFORCE_ISOTONIC = False   # keep off to preserve anchors; hybrid fill already enforces monotonicity
LIPSCHITZ        = 1       # gentle step limiting

# Default allocation (sum must be 1024). Expands to the default monotone k[a].
DEFAULT_ALLOC = [0, 4, 8, 16, 32, 96, 256, 512, 100]
assert sum(DEFAULT_ALLOC) == LUT_SIZE, "DEFAULT_ALLOC must sum to 1024"

# ---------- Robust parser ----------
_RX = re.compile(r'[-+]?\d+')
def parse_line_to_int4(line: str):
    xs = _RX.findall(line)
    if len(xs) < 4:
        return None
    try:
        r = int(xs[0]); a = int(xs[1]); kobs = int(xs[2]); m = int(xs[3])
    except ValueError:
        return None
    return r, a, kobs, m

# ---------- Prior helpers ----------
def prior_weight(a: int) -> float:
    if LAMBDA0 <= 0.0: return 0.0
    return LAMBDA0 * (1.0 + BETA * max(0, A0 - a) / max(1, A0))

def penalty_len(a: int, k_vals: np.ndarray) -> np.ndarray:
    if LAMBDA0 <= 0.0:
        return np.zeros_like(k_vals, dtype=np.float64)
    return prior_weight(a) * (GAMMA * k_vals.astype(np.float64))

# ---------- Rice cost models ----------
def trunc_rice_costs_for_m(m_arr: np.ndarray, ks: np.ndarray, L: int, esc_bits: int) -> np.ndarray:
    # Sum over samples: (q+1+k) if q<=L else (L+1+esc_bits)
    q = (m_arr[:, None] >> ks[None, :]).astype(np.int64)
    mask = (q <= L)
    base_ok  = q + 1 + ks[None, :]
    base_esc = (L + 1 + esc_bits)
    return np.where(mask, base_ok, base_esc).sum(axis=0)

def pure_rice_costs_for_m(m_arr: np.ndarray, ks: np.ndarray) -> np.ndarray:
    q = (m_arr[:, None] >> ks[None, :]).astype(np.int64)
    N = m_arr.shape[0]
    return q.sum(axis=0) + N * (1 + ks)

def best_k_for_bin_vec(m_arr: np.ndarray, a_bin: int) -> int:
    if m_arr.size == 0:
        return 0
    ks = np.arange(K_MIN, K_MAX + 1, dtype=np.int64)
    base = (trunc_rice_costs_for_m(m_arr, ks, TRUNC_L, ESC_BITS)
            if USE_TRUNC_RICE else pure_rice_costs_for_m(m_arr, ks))
    cost = base.astype(np.float64) + penalty_len(a_bin, ks)
    return int(ks[np.argmin(cost)])

# ---------- Isotonic & Lipschitz ----------
def isotonic_non_decreasing_pav(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isfinite(v), v, 0.0)
    n = v.size
    level=[]; weight=[]; end=[]
    for j in range(n):
        level.append(v[j]); weight.append(1.0); end.append(j)
        while len(level) >= 2 and level[-2] > level[-1]:
            L2, L1 = level[-2], level[-1]
            W2, W1 = weight[-2], weight[-1]
            e1 = end[-1]
            lv = (L2*W2 + L1*W1)/(W2+W1)
            level[-2] = lv; weight[-2] = W2+W1; end[-2] = e1
            level.pop(); weight.pop(); end.pop()
    out = np.empty(n, dtype=np.float64); s = 0
    for lv, ed in zip(level, end):
        out[s:ed+1] = lv; s = ed+1
    return out

def enforce_lipschitz(k_vals: np.ndarray, L: int) -> np.ndarray:
    out = k_vals.astype(np.float64).copy()
    if L > 0:
        for i in range(1, out.size):
            if out[i] > out[i-1] + L:
                out[i] = out[i-1] + L
        for i in range(out.size-2, -1, -1):
            if out[i] > out[i+1] + L:
                out[i] = out[i+1] + L
    out = np.rint(out).astype(np.int32)
    return np.clip(out, 0, 8)

# ---------- Default expansion & hybrid fill ----------
def expand_default_to_k(default_alloc, size=1024):
    """Expand DEFAULT_ALLOC(9) to a length=size monotone array default_k[a]."""
    default_alloc = list(default_alloc)
    assert sum(default_alloc) == size
    k = np.empty(size, dtype=np.int32)
    idx = 0
    for val, cnt in enumerate(default_alloc):
        if cnt > 0:
            k[idx:idx+cnt] = val
            idx += cnt
    return k

def fill_gaps_hybrid_with_default(k_partial: dict,
                                  default_alloc=DEFAULT_ALLOC,
                                  size=1024) -> np.ndarray:
    """
    Given anchors {a: k}, build full k[a] for a=0..size-1:
      - Outside observed range: keep default_k[a] (avoid stretching end values).
      - Inside observed range: stepwise monotone interpolation between anchors (staircase with unit steps).
    """
    default_k = expand_default_to_k(default_alloc, size=size)
    if not k_partial:
        return default_k.copy()

    a_sorted = np.array(sorted(k_partial.keys()), dtype=np.int32)
    k_sorted = np.array([int(k_partial[a]) for a in a_sorted], dtype=np.int32)

    out = default_k.copy()
    amin, amax = int(a_sorted[0]), int(a_sorted[-1])

    # Force anchors
    out[amin] = k_sorted[0]
    out[amax] = k_sorted[-1]

    # Interpolate between anchors as a staircase with unit steps
    for i in range(len(a_sorted) - 1):
        a0, a1 = int(a_sorted[i]), int(a_sorted[i+1])
        k0, k1 = int(k_sorted[i]), int(k_sorted[i+1])
        if a1 <= a0:
            continue
        n = a1 - a0
        if k0 == k1:
            out[a0:a1+1] = k0
        else:
            delta = k1 - k0
            # Positions where the step increases/decreases; equally spaced
            step_positions = np.linspace(0, n, num=abs(delta)+1, dtype=np.int32)
            cur = k0
            start = a0
            for sp in step_positions[1:]:
                end = a0 + sp
                out[start:end] = cur
                cur += 1 if delta > 0 else -1
                start = end
            out[start:a1+1] = cur

    # Re-apply anchors (safety)
    for a, kv in k_partial.items():
        if 0 <= a < size:
            out[a] = int(kv)

    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to residual dump CSV (r,a,k,m)")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="fallback to DEFAULT_ALLOC if total rows per r < this")
    ap.add_argument("--min-bins", type=int, default=8,
                    help="fallback to DEFAULT_ALLOC if distinct 'a' bins per r < this")
    args = ap.parse_args()

    # Gather m by (r,a)
    m_by_r_a = [{} for _ in range(R_CLASSES)]
    total_samples = [0]*R_CLASSES

    with open(args.path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_line_to_int4(line)
            if not parsed: continue
            r, a, _kobs, m = parsed
            if not (0 <= r < R_CLASSES): continue
            if not (0 <= a < LUT_SIZE):  continue
            if m < 0:                    continue
            m_by_r_a[r].setdefault(a, []).append(m)
            total_samples[r] += 1

    # Process per r
    for r in range(R_CLASSES):
        a_keys = sorted(m_by_r_a[r].keys())
        n_bins = len(a_keys)
        n_samp = total_samples[r]
        sys.stderr.write(f"[r={r}] bins={n_bins}, samples={n_samp}\n")

        # Fallback if no/sparse data
        if n_bins == 0 or n_samp < args.min_samples or n_bins < args.min_bins:
            print(" ".join(str(x) for x in DEFAULT_ALLOC))
            continue

        # Optimize k on observed a
        k_partial = {}
        for a, m_list in m_by_r_a[r].items():
            m_arr = np.fromiter(m_list, dtype=np.int64)
            k_partial[a] = best_k_for_bin_vec(m_arr, a)

        # Hybrid fill to 0..1023 (default outside, stair inside), then gentle Lipschitz
        k_full = fill_gaps_hybrid_with_default(k_partial, DEFAULT_ALLOC, LUT_SIZE)
        if ENFORCE_ISOTONIC:
            k_full = isotonic_non_decreasing_pav(k_full)   # usually unnecessary; hybrid already monotone
        k_full = enforce_lipschitz(k_full, LIPSCHITZ)

        # Allocation vector (counts of 0..8), guarantee sum=1024
        k_int = np.clip(k_full, 0, 8)
        alloc = np.bincount(k_int, minlength=9).astype(int)
        s = int(alloc.sum())
        if s != LUT_SIZE:
            alloc[-1] += (LUT_SIZE - s)

        print(" ".join(str(int(x)) for x in alloc.tolist()))

if __name__ == "__main__":
    main()
