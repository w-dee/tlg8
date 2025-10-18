"""マルチタスクモデル出力を用いた段階的ビームサーチ推論。"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from multitask_model import (
    MultiTaskModel,
    TorchMultiTask,
    build_filter_candidates,
    log_softmax,
    predict_logits_batched,
)


LOGGER = logging.getLogger(__name__)

def _ensure_vector(arr: np.ndarray) -> np.ndarray:
    """1 次元ロジット配列を取得するヘルパー。"""

    vec = np.asarray(arr, dtype=np.float64)
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec[0]
    if vec.ndim != 1:
        raise ValueError("単一サンプルのロジットが必要です")
    return vec


def _deterministic_top_indices(vec: np.ndarray, k: int) -> np.ndarray:
    """(-logit, class_id) の順で上位 k インデックスを返す。"""

    if k <= 0:
        raise ValueError("k は正の整数である必要があります")
    order = np.lexsort((np.arange(vec.shape[0]), -vec))
    return order[: min(k, order.shape[0])]


def _candidate_scores(vec: np.ndarray, indices: Sequence[int]) -> Dict[int, float]:
    """指定インデックスの log-softmax 値を取得する。"""

    if not indices:
        return {}
    log_probs = log_softmax(vec[None, :])[0]
    return {int(idx): float(log_probs[int(idx)]) for idx in indices}


def _select_with_margin(
    vec: np.ndarray, max_k: int, margin: float
) -> Tuple[List[int], float, float]:
    """マージン条件に基づき上位候補リストを返す。"""

    top_indices = _deterministic_top_indices(vec, max_k)
    if top_indices.size == 0:
        return [], float("inf"), float("inf")
    scores = vec[top_indices]
    margin12 = float("inf")
    margin23 = float("inf")
    if top_indices.size >= 2:
        margin12 = float(scores[0] - scores[1])
    if top_indices.size >= 3:
        margin23 = float(scores[1] - scores[2])

    if top_indices.size >= 2 and margin12 > margin:
        selected = top_indices[:1]
    elif top_indices.size >= 3:
        if margin12 <= margin and margin23 <= margin:
            selected = top_indices[:3]
        else:
            selected = top_indices[:2]
    else:
        selected = top_indices
    return [int(idx) for idx in selected], margin12, margin23


def _enumerate_candidates(
    logits: Dict[str, np.ndarray],
    max_trials: int,
    margin_reorder: float,
    margin_interleave: float,
    margin_filter_perm: float,
    margin_predictor: float,
    temperature: float,
) -> List[Tuple[Dict[str, int], float]]:
    """モデルロジットから最大 max_trials 件の候補構成を生成する。"""

    temp = max(float(temperature), 1e-6)
    predictor_vec = _ensure_vector(logits["predictor"]) / temp
    reorder_vec = _ensure_vector(logits["reorder"]) / temp
    interleave_vec = _ensure_vector(logits["interleave"]) / temp
    perm_vec = _ensure_vector(logits["filter_perm"]) / temp
    primary_vec = _ensure_vector(logits["filter_primary"]) / temp
    secondary_vec = _ensure_vector(logits["filter_secondary"]) / temp

    predictor_candidates, predictor_margin, predictor_margin23 = _select_with_margin(
        predictor_vec, 3, margin_predictor
    )
    reorder_candidates, reorder_margin, reorder_margin23 = _select_with_margin(
        reorder_vec, 3, margin_reorder
    )
    interleave_candidates, interleave_margin, _ = _select_with_margin(
        interleave_vec, 3, margin_interleave
    )
    perm_candidates, perm_margin, _ = _select_with_margin(
        perm_vec, 2, margin_filter_perm
    )
    primary_indices = _deterministic_top_indices(primary_vec, 3)
    secondary_indices = _deterministic_top_indices(secondary_vec, 3)

    LOGGER.info(
        "ヘッド predictor: k=%d margin12=%.3f margin23=%.3f 候補=%s",
        len(predictor_candidates),
        predictor_margin,
        predictor_margin23,
        predictor_candidates,
    )
    LOGGER.info(
        "ヘッド reorder: k=%d margin12=%.3f margin23=%.3f 候補=%s",
        len(reorder_candidates),
        reorder_margin,
        reorder_margin23,
        reorder_candidates,
    )
    LOGGER.info(
        "ヘッド interleave: k=%d margin12=%.3f 候補=%s",
        len(interleave_candidates),
        interleave_margin,
        interleave_candidates,
    )
    LOGGER.info(
        "ヘッド filter_perm: k=%d margin12=%.3f 候補=%s",
        len(perm_candidates),
        perm_margin,
        perm_candidates,
    )

    perm_scores = _candidate_scores(perm_vec, perm_candidates)
    primary_scores = _candidate_scores(primary_vec, primary_indices)
    secondary_scores = _candidate_scores(secondary_vec, secondary_indices)

    base_product = (
        max(1, len(reorder_candidates))
        * max(1, len(interleave_candidates))
        * max(1, len(predictor_candidates))
    )
    allow_six = base_product * 6 <= max_trials
    filter_max = 6 if allow_six else 4

    filter_candidates = build_filter_candidates(
        [(pid, perm_scores[pid]) for pid in perm_candidates],
        [(int(idx), primary_scores[int(idx)]) for idx in primary_indices],
        [(int(idx), secondary_scores[int(idx)]) for idx in secondary_indices],
        max_candidates=filter_max,
    )

    if not filter_candidates:
        perm_fallback = perm_candidates[0] if perm_candidates else 0
        primary_fallback = int(primary_indices[0]) if primary_indices.size else 0
        secondary_fallback = int(secondary_indices[0]) if secondary_indices.size else 0
        code = ((perm_fallback % 6) << 4) | ((primary_fallback % 4) << 2) | (secondary_fallback % 4)
        filter_candidates = [(code, 0.0)]

    def total_combos() -> int:
        return (
            len(reorder_candidates)
            * len(interleave_candidates)
            * len(filter_candidates)
            * len(predictor_candidates)
        )

    total = total_combos()
    for target in (6, 4, 3, 2):
        if total <= max_trials:
            break
        if len(filter_candidates) > target:
            filter_candidates = filter_candidates[:target]
            total = total_combos()
            LOGGER.info(
                "フィルター候補を %d 件に削減 (総組合せ=%d)",
                len(filter_candidates),
                total,
            )

    if total > max_trials and len(predictor_candidates) > 1:
        new_len = 2 if len(predictor_candidates) > 2 else 1
        predictor_candidates = predictor_candidates[:new_len]
        total = total_combos()
        LOGGER.info(
            "predictor 候補を %d 件に削減 (margin12=%.3f 総組合せ=%d)",
            len(predictor_candidates),
            predictor_margin,
            total,
        )

    if total > max_trials and len(reorder_candidates) > 1:
        new_len = 2 if len(reorder_candidates) > 2 else 1
        reorder_candidates = reorder_candidates[:new_len]
        total = total_combos()
        LOGGER.info(
            "reorder 候補を %d 件に削減 (margin12=%.3f 総組合せ=%d)",
            len(reorder_candidates),
            reorder_margin,
            total,
        )

    if total > max_trials and len(interleave_candidates) > 1:
        interleave_candidates = interleave_candidates[:1]
        total = total_combos()
        LOGGER.info(
            "interleave 候補を %d 件に削減 (margin12=%.3f 総組合せ=%d)",
            len(interleave_candidates),
            interleave_margin,
            total,
        )

    total = total_combos()
    if total > max_trials:
        LOGGER.warning("組合せ削減で上限 %d 件を下回れませんでした (総組合せ=%d)", max_trials, total)

    predictor_scores = _candidate_scores(predictor_vec, predictor_candidates)
    reorder_scores = _candidate_scores(reorder_vec, reorder_candidates)
    interleave_scores = _candidate_scores(interleave_vec, interleave_candidates)

    LOGGER.info(
        "最終候補数 reorder=%d interleave=%d filter=%d predictor=%d 総組合せ=%d",
        len(reorder_candidates),
        len(interleave_candidates),
        len(filter_candidates),
        len(predictor_candidates),
        total_combos(),
    )

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
    margin_reorder: float = 0.5,
    margin_interleave: float = 0.5,
    margin_filter_perm: float = 0.4,
    margin_predictor: float = 0.4,
    max_trials: int = 16,
    temperature: float | None = None,
):
    """段階的ビームサーチにより最終ブロック構成を決定する。"""

    logits = model.predict_logits(features)
    candidates = _enumerate_candidates(
        logits,
        max_trials,
        margin_reorder,
        margin_interleave,
        margin_filter_perm,
        margin_predictor,
        temperature if temperature is not None else model.inference_temperature,
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
    LOGGER.info("実際に評価した組合せ=%d", eval_count)
    return best_config, best_bits


def select_block_config_batched(
    features_batch: np.ndarray,
    model: TorchMultiTask,
    *,
    device,
    batch_size: int = 8192,
    amp: str = "bf16",
    margin_reorder: float = 0.5,
    margin_interleave: float = 0.5,
    margin_filter_perm: float = 0.4,
    margin_predictor: float = 0.4,
    max_trials: int = 16,
    temperature: float | None = None,
    try_encode_block: Callable[[Dict[str, int]], int] | None = None,
):
    """複数ブロックの特徴量をまとめてビームサーチ推論する。"""

    if features_batch.ndim == 1:
        features_batch = features_batch[None, :]
    logits_batch = predict_logits_batched(
        model,
        features_batch,
        device,
        batch_size=batch_size,
        amp=amp,
    )
    sample_count = next(iter(logits_batch.values())).shape[0]
    results: List[object] = []
    for idx in range(sample_count):
        single_logits = {name: arr[idx : idx + 1] for name, arr in logits_batch.items()}
        candidates = _enumerate_candidates(
            single_logits,
            max_trials,
            margin_reorder,
            margin_interleave,
            margin_filter_perm,
            margin_predictor,
            temperature if temperature is not None else model.inference_temperature,
        )
        if try_encode_block is None:
            results.append(candidates)
            continue
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
        LOGGER.info("実際に評価した組合せ=%d", eval_count)
        results.append((best_config, best_bits))
    return results
