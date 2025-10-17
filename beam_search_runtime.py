"""マルチタスクモデル出力を用いた段階的ビームサーチ推論。"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from multitask_model import build_filter_candidates, log_softmax, MultiTaskModel


def _ensure_vector(arr: np.ndarray) -> np.ndarray:
    """1 次元ロジット配列を取得するヘルパー。"""

    vec = np.asarray(arr, dtype=np.float64)
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec[0]
    if vec.ndim != 1:
        raise ValueError("単一サンプルのロジットが必要です")
    return vec


def _top_indices(vec: np.ndarray, k: int) -> np.ndarray:
    """ロジット値から上位 k 個のインデックスを返す。"""

    order = np.argsort(vec)[::-1]
    return order[: min(k, order.shape[0])]


def _candidate_scores(vec: np.ndarray, indices: np.ndarray) -> Dict[int, float]:
    """指定されたインデックスに対する log-softmax スコアを返す。"""

    log_probs = log_softmax(vec[None, :])[0]
    return {int(idx): float(log_probs[idx]) for idx in indices}


def _apply_margin(vec: np.ndarray, k: int, margin: float) -> Tuple[List[int], float]:
    """上位候補を取得し、信頼度マージンによって 1 件に絞るか判断する。"""

    indices = _top_indices(vec, k)
    if indices.size == 0:
        return [], float("inf")
    margin_value = float("inf")
    if indices.size >= 2:
        margin_value = float(vec[indices[0]] - vec[indices[1]])
    width = 1 if indices.size >= 2 and margin_value > margin else indices.size
    return [int(idx) for idx in indices[:width]], margin_value


def _enumerate_candidates(
    logits: Dict[str, np.ndarray],
    max_trials: int,
    margin_predictor: float,
    margin_filter: float,
    margin_reorder: float,
    margin_interleave: float,
) -> List[Tuple[Dict[str, int], float]]:
    """モデルロジットから最大 max_trials 件の候補構成を生成する。"""

    predictor_vec = _ensure_vector(logits["predictor"])
    reorder_vec = _ensure_vector(logits["reorder"])
    interleave_vec = _ensure_vector(logits["interleave"])
    perm_vec = _ensure_vector(logits["filter_perm"])
    primary_vec = _ensure_vector(logits["filter_primary"])
    secondary_vec = _ensure_vector(logits["filter_secondary"])

    predictor_candidates, predictor_margin = _apply_margin(
        predictor_vec, 2, margin_predictor
    )
    if not predictor_candidates:
        predictor_candidates = [int(_top_indices(predictor_vec, 1)[0])]
    reorder_candidates, reorder_margin = _apply_margin(reorder_vec, 2, margin_reorder)
    if not reorder_candidates:
        reorder_candidates = [int(_top_indices(reorder_vec, 1)[0])]
    interleave_candidates, interleave_margin = _apply_margin(
        interleave_vec, 2, margin_interleave
    )
    if not interleave_candidates:
        interleave_candidates = [int(_top_indices(interleave_vec, 1)[0])]

    perm_top = _top_indices(perm_vec, 1)
    primary_top = _top_indices(primary_vec, 2)
    secondary_top = _top_indices(secondary_vec, 2)

    primary_margin = (
        float(primary_vec[primary_top[0]] - primary_vec[primary_top[1]])
        if primary_top.size >= 2
        else float("inf")
    )
    secondary_margin = (
        float(secondary_vec[secondary_top[0]] - secondary_vec[secondary_top[1]])
        if secondary_top.size >= 2
        else float("inf")
    )
    filter_margin = min(primary_margin, secondary_margin)

    perm_scores = _candidate_scores(perm_vec, perm_top)
    primary_scores = _candidate_scores(primary_vec, primary_top)
    secondary_scores = _candidate_scores(secondary_vec, secondary_top)

    filter_candidates = build_filter_candidates(
        [(int(idx), perm_scores[int(idx)]) for idx in perm_top],
        [(int(idx), primary_scores[int(idx)]) for idx in primary_top],
        [(int(idx), secondary_scores[int(idx)]) for idx in secondary_top],
        max_candidates=4,
    )
    if not filter_candidates:
        best_code = int(
            ((int(perm_top[0]) % 6) << 4)
            | ((int(primary_top[0]) % 4) << 2)
            | (int(secondary_top[0]) % 4)
        )
        filter_candidates = [(best_code, 0.0)]

    if (
        filter_margin > margin_filter
        and primary_top.size >= 2
        and secondary_top.size >= 2
        and len(filter_candidates) > 2
    ):
        filter_candidates = filter_candidates[:2]

    reorder_collapsed = len(reorder_candidates) == 1
    interleave_collapsed = len(interleave_candidates) == 1

    total_combo = (
        len(predictor_candidates)
        * len(filter_candidates)
        * len(reorder_candidates)
        * len(interleave_candidates)
    )
    if total_combo > max_trials and not (reorder_collapsed or interleave_collapsed):
        if filter_margin < predictor_margin and len(filter_candidates) > 2:
            filter_candidates = filter_candidates[:2]
        elif len(predictor_candidates) > 1:
            predictor_candidates = predictor_candidates[:1]
        elif len(filter_candidates) > 2:
            filter_candidates = filter_candidates[:2]
        total_combo = (
            len(predictor_candidates)
            * len(filter_candidates)
            * len(reorder_candidates)
            * len(interleave_candidates)
        )
    if total_combo > max_trials and len(filter_candidates) > 2:
        filter_candidates = filter_candidates[:2]
        total_combo = (
            len(predictor_candidates)
            * len(filter_candidates)
            * len(reorder_candidates)
            * len(interleave_candidates)
        )
    if total_combo > max_trials and len(predictor_candidates) > 1:
        predictor_candidates = predictor_candidates[:1]
        total_combo = (
            len(predictor_candidates)
            * len(filter_candidates)
            * len(reorder_candidates)
            * len(interleave_candidates)
        )

    predictor_scores = _candidate_scores(predictor_vec, np.array(predictor_candidates))
    reorder_scores = _candidate_scores(reorder_vec, np.array(reorder_candidates))
    interleave_scores = _candidate_scores(interleave_vec, np.array(interleave_candidates))

    candidates: List[Tuple[Dict[str, int], float]] = []
    for reorder_id in reorder_candidates:
        for interleave_id in interleave_candidates:
            for filter_code, filter_score in filter_candidates:
                for predictor_id in predictor_candidates:
                    score = (
                        predictor_scores[predictor_id]
                        + filter_score
                        + reorder_scores[reorder_id]
                        + interleave_scores[interleave_id]
                    )
                    config = {
                        "predictor": int(predictor_id),
                        "filter": int(filter_code),
                        "reorder": int(reorder_id),
                        "interleave": int(interleave_id),
                    }
                    candidates.append((config, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > max_trials:
        candidates = candidates[:max_trials]
    return candidates


def select_block_config(
    features: np.ndarray,
    model: MultiTaskModel,
    try_encode_block: Callable[[Dict[str, int]], int] | None,
    *,
    margin_predictor: float = 0.5,
    margin_filter: float = 0.3,
    margin_reorder: float = 0.5,
    margin_interleave: float = 0.5,
    max_trials: int = 16,
):
    """段階的ビームサーチにより最終ブロック構成を決定する。"""

    logits = model.predict_logits(features)
    candidates = _enumerate_candidates(
        logits,
        max_trials,
        margin_predictor,
        margin_filter,
        margin_reorder,
        margin_interleave,
    )

    if try_encode_block is None:
        return candidates

    best_config = None
    best_bits = None
    eval_count = 0
    for config, _score in candidates:
        bits = int(try_encode_block(config))
        eval_count += 1
        if best_bits is None or bits < best_bits:
            best_bits = bits
            best_config = config
    if eval_count > max_trials:
        raise RuntimeError("最大試行回数を超過しました")
    return best_config, best_bits
