"""Loss functions for fall classification, selectable via config."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weighting.

    Focal loss (Lin et al., 2017) down-weights easy examples so training
    focuses on hard ones::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    At ``gamma == 0`` this reduces to (optionally weighted) cross-entropy, up
    to the weight-normalization convention: focal uses a plain mean over the
    batch (the standard focal convention), whereas ``nn.CrossEntropyLoss``
    normalizes weighted loss by the sum of the per-sample weights.

    Args:
        gamma: Focusing parameter (>= 0). Higher means more focus on hard
            examples. Typical values 1.0-2.0.
        alpha: Optional per-class weight tensor of shape ``(num_classes,)``.
            Reuse inverse-frequency class weights here to address imbalance.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction {reduction!r}")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()
        focal = (1.0 - pt) ** self.gamma * (-log_pt)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(
    config: dict[str, Any],
    class_weights: Tensor | None,
) -> nn.Module:
    """Construct the training loss from the ``training`` config section.

    Supported ``training.loss`` values:
        - ``"cross_entropy"`` (default): weighted CrossEntropyLoss.
        - ``"focal"``: FocalLoss with ``focal_gamma`` and ``focal_alpha``.

    ``focal_alpha`` may be ``"auto"`` (reuse ``class_weights``), ``null``/``none``
    (no per-class weighting), or a list of per-class floats.
    """
    training_cfg = config.get("training", {})
    loss_name = str(training_cfg.get("loss", "cross_entropy")).lower()

    if loss_name in ("cross_entropy", "ce", "crossentropy"):
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == "focal":
        gamma = float(training_cfg.get("focal_gamma", 1.5))
        alpha_cfg = training_cfg.get("focal_alpha", "auto")

        if alpha_cfg is None or (isinstance(alpha_cfg, str) and alpha_cfg.lower() == "none"):
            alpha = None
        elif isinstance(alpha_cfg, str) and alpha_cfg.lower() == "auto":
            alpha = class_weights
        elif isinstance(alpha_cfg, (list, tuple)):
            alpha = torch.tensor([float(v) for v in alpha_cfg], dtype=torch.float32)
        else:
            raise ValueError(f"focal_alpha must be 'auto', null, or a list; got {alpha_cfg!r}")

        return FocalLoss(gamma=gamma, alpha=alpha)

    raise ValueError(f"Unknown training.loss {loss_name!r}; expected 'cross_entropy' or 'focal'")
