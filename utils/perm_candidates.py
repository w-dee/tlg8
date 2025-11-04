"""カラー順序 perm 候補を相関ヒューリスティックで生成するユーティリティ。"""

from __future__ import annotations

"""perm 候補生成用ヒューリスティック集。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# perm ID ごとのチャンネル並び。TLG8 では 3! = 6 通りの並び替えのみを考慮する。
PERMUTATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


@dataclass(frozen=True)
class PermCandidate:
    """生成された perm 候補とスコアを保持するデータクラス。"""

    perm_id: int
    score: float


def _extract_covariance(stats: Mapping[str, object] | None) -> np.ndarray | None:
    """stats から 3x3 共分散 / 相関行列を取得する。"""

    if stats is None:
        return None
    if "channel_corr" in stats:
        arr = np.asarray(stats["channel_corr"], dtype=np.float64)
    elif "channel_cov" in stats:
        arr = np.asarray(stats["channel_cov"], dtype=np.float64)
        if arr.shape == (3, 3):
            diag = np.sqrt(np.clip(np.diag(arr), 1e-8, None))
            denom = np.outer(diag, diag)
            with np.errstate(invalid="ignore", divide="ignore"):
                arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
    else:
        return None
    if arr.shape != (3, 3):
        return None
    return arr


def _score_permutation(corr: np.ndarray, perm: Sequence[int]) -> float:
    """与えられた perm 並びでのオフ対角相関絶対値総和を負値で返す。"""

    permuted = corr[np.ix_(perm, perm)]
    mask = ~np.eye(permuted.shape[0], dtype=bool)
    penalty = np.sum(np.abs(permuted[mask]))
    return -float(penalty)


def gen_perm_candidates(
    image_stats: Mapping[str, object] | None,
    k_max: int = 4,
    *,
    fallback: Iterable[int] | None = None,
) -> List[int]:
    """ヒューリスティックに perm 候補 ID を上位順に返す。"""

    if k_max <= 0:
        return []
    corr = _extract_covariance(image_stats)
    if corr is None:
        base = list(range(len(PERMUTATIONS)))
    else:
        scored: List[PermCandidate] = []
        for perm_id, perm in enumerate(PERMUTATIONS):
            score = _score_permutation(corr, perm)
            scored.append(PermCandidate(perm_id=perm_id, score=score))
        scored.sort(key=lambda item: (-item.score, item.perm_id))
        base = [item.perm_id for item in scored]
    if fallback is not None:
        base = list(dict.fromkeys(list(base) + list(fallback)))
    return base[: min(k_max, len(base))]


__all__ = ["gen_perm_candidates", "PERMUTATIONS"]
