"""Hierarchical dual-stream BiGRU with temporal attention pooling."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.constants import BOTTOM_KPT_DIM, FEAT_DIM, NUM_CLASSES, TOP_KPT_DIM
from src.models.base import BaseFallModel, register_model


class TemporalAttentionPooling(nn.Module):
    """Masked softmax attention over sequence outputs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        """Pool ``(B, T, D)`` into ``(B, D)`` using a validity mask ``(B, T)``."""
        valid = mask.to(dtype=torch.bool)
        scores = self.score(sequence).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = weights * valid.to(dtype=weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights / denom
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)


class BidirectionalGRUBranch(nn.Module):
    """Bidirectional GRU followed by masked temporal attention pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = TemporalAttentionPooling(hidden_dim * 2)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        outputs, _ = self.gru(sequence)
        return self.pool(outputs, mask)


@register_model("bigru_hierarchical")
class HierarchicalDualStreamBiGRU(BaseFallModel):
    """Three-branch BiGRU model over top/bottom keypoints and engineered features."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", config)

        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.3))
        num_classes = int(model_cfg.get("num_classes", NUM_CLASSES))

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gru_top = BidirectionalGRUBranch(TOP_KPT_DIM, hidden_dim, num_layers, dropout)
        self.gru_bot = BidirectionalGRUBranch(BOTTOM_KPT_DIM, hidden_dim, num_layers, dropout)
        self.gru_feat = BidirectionalGRUBranch(FEAT_DIM, hidden_dim, num_layers, dropout)

        branch_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(branch_dim * 3, num_classes),
        )

    @property
    def model_name(self) -> str:
        return getattr(self, "_registry_name", "bigru_hierarchical")

    def forward(self, batch: dict[str, Any]) -> Tensor:
        top_kp = batch["top_kp"]
        bot_kp = batch["bot_kp"]
        feat = batch["feat"]
        mask = batch["mask"]

        top_repr = self.gru_top(top_kp, mask)
        bot_repr = self.gru_bot(bot_kp, mask)
        feat_repr = self.gru_feat(feat, mask)

        fused = torch.cat([top_repr, bot_repr, feat_repr], dim=-1)
        return self.classifier(fused)
