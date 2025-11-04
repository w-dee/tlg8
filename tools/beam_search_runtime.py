"""マルチタスクモデル出力を用いたビームサーチ推論ユーティリティ。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multitask_model import (
    MultiTaskModel,
    TorchMultiTask,
    log_softmax,
    predict_logits_batched,
)
from utils.perm_candidates import gen_perm_candidates

LOGGER = logging.getLogger(__name__)

# DNN でスコアリングする軸の展開順序。
AXIS_ORDER: Tuple[str, ...] = (
    "reorder",
    "interleave",
    "predictor",
    "filter_primary",
    "filter_secondary",
)

# 代理コスト（ビット・時間）テーブル。現状は最小構成のプレースホルダ。
AXIS_BITS_COST: Dict[str, Dict[int, float]] = {}
AXIS_TIME_COST: Dict[str, Dict[int, float]] = {}


@dataclass(frozen=True)
class AxisCandidate:
    """単一軸における候補値と対数確率。"""

    value: int
    log_prob: float


@dataclass(order=True)
class BeamNode:
    """ビームサーチ中の部分構成を保持するノード。"""

    priority: float
    axis_index: int = field(compare=False)
    log_prob: float = field(compare=False)
    bits_cost: float = field(compare=False)
    time_cost: float = field(compare=False)
    choices: Dict[str, int] = field(compare=False, default_factory=dict)


@dataclass(frozen=True)
class BeamCandidate:
    """最終的な構成候補とその代理コスト。"""

    config: Dict[str, int]
    cost: float
    log_prob: float
    bits_cost: float
    time_cost: float


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


def _build_axis_candidates(logits: Mapping[str, np.ndarray], axis: str, topk: int) -> List[AxisCandidate]:
    """指定軸の top-k 候補を log-softmax 値と共に生成する。"""

    if axis not in logits:
        raise KeyError(f"logits に軸 {axis} が存在しません")
    vec = _ensure_vector(logits[axis])
    indices = _deterministic_top_indices(vec, max(1, topk))
    log_probs = log_softmax(vec[None, :])[0]
    candidates = [AxisCandidate(int(idx), float(log_probs[int(idx)])) for idx in indices]
    if not candidates:
        candidates = [AxisCandidate(0, float(log_probs.max()))]
    return candidates


def _axis_cost(table: Mapping[str, Mapping[int, float]], axis: str, value: int) -> float:
    """軸ごとの代理コストを返す。"""

    mapping = table.get(axis)
    if mapping is None:
        return 0.0
    return float(mapping.get(int(value), 0.0))


def _pack_filter_code(perm: int, primary: int, secondary: int) -> int:
    """perm / primary / secondary から 6*4*4 = 96 種のフィルターコードを生成する。"""

    perm_mod = int(perm) % 6
    primary_mod = int(primary) % 4
    secondary_mod = int(secondary) % 4
    return (perm_mod << 4) | (primary_mod << 2) | secondary_mod


def _perm_prior(rank: int) -> float:
    """ヒューリスティック perm の順位に基づく擬似対数事前確率。"""

    return -0.2 * float(rank)


def _generate_beam_candidates(
    logits: Mapping[str, np.ndarray],
    image_stats: Mapping[str, object] | None,
    *,
    dnn_topk: int,
    beam_width: int,
    beam_slack: float,
    perm_candidate_mode: str,
    perm_kmax: int,
    time_penalty_lambda: float,
    axis_bits_cost: Mapping[str, Mapping[int, float]] | None = None,
    axis_time_cost: Mapping[str, Mapping[int, float]] | None = None,
) -> List[BeamCandidate]:
    """DNN ロジットと統計からビーム候補を生成する。"""

    axis_bits_cost = axis_bits_cost or AXIS_BITS_COST
    axis_time_cost = axis_time_cost or AXIS_TIME_COST
    axis_candidates: Dict[str, List[AxisCandidate]] = {}
    for axis in AXIS_ORDER:
        axis_candidates[axis] = _build_axis_candidates(logits, axis, dnn_topk)
    if perm_candidate_mode == "rule":
        perm_ids = gen_perm_candidates(image_stats, k_max=perm_kmax)
    else:
        perm_ids = list(range(min(perm_kmax, 6)))
    if not perm_ids:
        perm_ids = [0]

    beam: List[BeamNode] = [
        BeamNode(priority=0.0, axis_index=0, log_prob=0.0, bits_cost=0.0, time_cost=0.0, choices={})
    ]
    best_cost = math.inf

    for axis_idx, axis in enumerate(AXIS_ORDER):
        next_nodes: List[BeamNode] = []
        candidates = axis_candidates[axis]
        for node in beam:
            for candidate in candidates:
                choices = dict(node.choices)
                choices[axis] = candidate.value
                bits_cost = node.bits_cost + _axis_cost(axis_bits_cost, axis, candidate.value)
                time_cost = node.time_cost + _axis_cost(axis_time_cost, axis, candidate.value)
                log_prob = node.log_prob + candidate.log_prob
                priority = bits_cost + time_penalty_lambda * time_cost - log_prob
                if best_cost < math.inf and priority > best_cost * (1.0 + beam_slack):
                    continue
                next_nodes.append(
                    BeamNode(
                        priority=priority,
                        axis_index=axis_idx + 1,
                        log_prob=log_prob,
                        bits_cost=bits_cost,
                        time_cost=time_cost,
                        choices=choices,
                    )
                )
        if not next_nodes:
            LOGGER.warning("軸 %s の候補が空のためフォールバックします", axis)
            next_nodes = beam
        next_nodes.sort()
        beam = next_nodes[: max(1, beam_width)]

    final_candidates: List[BeamCandidate] = []
    for node in beam:
        primary = node.choices.get("filter_primary", 0)
        secondary = node.choices.get("filter_secondary", 0)
        for rank, perm_id in enumerate(perm_ids):
            perm_log_prior = _perm_prior(rank)
            bits_cost = node.bits_cost + _axis_cost(axis_bits_cost, "perm", perm_id)
            time_cost = node.time_cost + _axis_cost(axis_time_cost, "perm", perm_id)
            log_prob = node.log_prob + perm_log_prior
            cost = bits_cost + time_penalty_lambda * time_cost - log_prob
            config = {
                "predictor": node.choices.get("predictor", 0),
                "filter": _pack_filter_code(perm_id, primary, secondary),
                "reorder": node.choices.get("reorder", 0),
                "interleave": node.choices.get("interleave", 0),
            }
            final_candidates.append(
                BeamCandidate(
                    config=config,
                    cost=cost,
                    log_prob=log_prob,
                    bits_cost=bits_cost,
                    time_cost=time_cost,
                )
            )
            best_cost = min(best_cost, cost)
    final_candidates.sort(key=lambda item: (item.cost, item.config["filter"]))
    return final_candidates


def select_block_config(
    features: np.ndarray,
    model: MultiTaskModel,
    try_encode_block: Callable[[Dict[str, int]], int] | None,
    *,
    beam_width: int = 4,
    beam_slack: float = 0.02,
    perm_candidate_mode: str = "rule",
    perm_kmax: int = 4,
    dnn_topk: int = 3,
    time_penalty_lambda: float = 0.0,
    image_stats: Mapping[str, object] | None = None,
) -> object:
    """ビームサーチで最適ブロック構成を選択する。"""

    logits = model.predict_logits(features)
    candidates = _generate_beam_candidates(
        logits,
        image_stats,
        dnn_topk=dnn_topk,
        beam_width=beam_width,
        beam_slack=float(max(0.0, beam_slack)),
        perm_candidate_mode=perm_candidate_mode,
        perm_kmax=perm_kmax,
        time_penalty_lambda=time_penalty_lambda,
    )
    if try_encode_block is None:
        return [(cand.config, cand.cost) for cand in candidates]

    best_config: Optional[Dict[str, int]] = None
    best_bits: Optional[int] = None
    best_cost: Optional[float] = None
    evaluated = 0
    for idx, cand in enumerate(candidates):
        if best_cost is not None and cand.cost > best_cost * (1.0 + beam_slack):
            LOGGER.debug(
                "スラック基準で候補展開を停止 idx=%d cost=%.4f best=%.4f slack=%.3f",
                idx,
                cand.cost,
                best_cost,
                beam_slack,
            )
            break
        bits = int(try_encode_block(cand.config))
        evaluated += 1
        if best_bits is None or bits < best_bits:
            best_bits = bits
            best_config = cand.config
            best_cost = cand.cost
    LOGGER.info(
        "ビーム候補数=%d 評価=%d 最良コスト=%.4f",
        len(candidates),
        evaluated,
        best_cost if best_cost is not None else float("nan"),
    )
    stats = {
        "heuristic_candidates": len(candidates),
        "evaluated": evaluated,
        "best_cost": best_cost,
        "beam_width": beam_width,
        "dnn_topk": dnn_topk,
        "perm_kmax": perm_kmax,
    }
    return best_config, best_bits, stats


def select_block_config_batched(
    features_batch: np.ndarray,
    model: TorchMultiTask,
    *,
    device,
    batch_size: int = 8192,
    amp: str = "bf16",
    beam_width: int = 4,
    beam_slack: float = 0.02,
    perm_candidate_mode: str = "rule",
    perm_kmax: int = 4,
    dnn_topk: int = 3,
    time_penalty_lambda: float = 0.0,
    image_stats_batch: Sequence[Mapping[str, object]] | None = None,
    try_encode_block: Callable[[Dict[str, int]], int] | None = None,
) -> List[object]:
    """複数ブロックをまとめてビームサーチ推論する。"""

    if features_batch.ndim == 1:
        features_batch = features_batch[None, :]
    logits_batch = predict_logits_batched(
        model,
        features_batch,
        device,
        batch_size=batch_size,
        amp=amp,
        allow_cpu_transfer=True,
    )
    sample_count = next(iter(logits_batch.values())).shape[0]
    results: List[object] = []
    for idx in range(sample_count):
        single_logits = {name: arr[idx : idx + 1] for name, arr in logits_batch.items()}
        stats_map = image_stats_batch[idx] if image_stats_batch and idx < len(image_stats_batch) else None
        candidates = _generate_beam_candidates(
            single_logits,
            stats_map,
            dnn_topk=dnn_topk,
            beam_width=beam_width,
            beam_slack=float(max(0.0, beam_slack)),
            perm_candidate_mode=perm_candidate_mode,
            perm_kmax=perm_kmax,
            time_penalty_lambda=time_penalty_lambda,
        )
        if try_encode_block is None:
            results.append([(cand.config, cand.cost) for cand in candidates])
            continue
        best_config: Optional[Dict[str, int]] = None
        best_bits: Optional[int] = None
        best_cost: Optional[float] = None
        evaluated = 0
        for cand in candidates:
            if best_cost is not None and cand.cost > best_cost * (1.0 + beam_slack):
                break
            bits = int(try_encode_block(cand.config))
            evaluated += 1
            if best_bits is None or bits < best_bits:
                best_bits = bits
                best_config = cand.config
                best_cost = cand.cost
        results.append((best_config, best_bits, {"evaluated": evaluated, "best_cost": best_cost}))
    return results
