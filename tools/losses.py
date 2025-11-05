"""学習用損失関数の拡張。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """マルチクラス分類向け Focal Loss。"""

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction は mean/sum/none のいずれかである必要があります")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """ロジットと正解ラベルから損失を算出する。"""

        if logits.ndim != 2:
            raise ValueError("logits は (B, C) 形状である必要があります")
        if target.ndim != 1:
            raise ValueError("target は (B,) 形状である必要があります")
        logpt = F.log_softmax(logits, dim=1)
        gather = logpt.gather(1, target.view(-1, 1))
        log_probs = gather.squeeze(1)
        pt = torch.exp(log_probs)
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = -focal_factor * log_probs
        if self.alpha is not None:
            loss = loss * self.alpha
        if weight is not None:
            loss = weight[target] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
