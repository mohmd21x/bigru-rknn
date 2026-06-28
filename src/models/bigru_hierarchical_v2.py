"""Hierarchical dual-stream (B)GRU v2 with feature standardization.

Improvements over ``bigru_hierarchical``:

- **Configurable direction**: ``bidirectional`` (default ``True``). Set to
  ``False`` for a causal/unidirectional GRU suited to low-latency streaming.
- **Per-feature standardization**: the engineered ``feat`` stream is z-scored
  using train-set statistics stored as buffers (populated once before
  training via :meth:`set_feature_stats`, saved in the checkpoint, and reused
  at inference). Keypoint streams are already bbox-normalized.
- **Stronger regularization**: optional ``LayerNorm`` on pooled branch
  representations plus dropout, to curb the heavy overfitting seen in v1.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.constants import BOTTOM_KPT_DIM, FEAT_DIM, NUM_CLASSES, TOP_KPT_DIM
from src.models.base import BaseFallModel, register_model
from src.models.bigru_hierarchical import TemporalAttentionPooling


class GRUBranch(nn.Module):
    """(Optionally bidirectional) GRU + masked temporal attention pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = TemporalAttentionPooling(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim) if use_layernorm else nn.Identity()

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        outputs, _ = self.gru(sequence)
        pooled = self.pool(outputs, mask)
        return self.norm(pooled)


@register_model("bigru_hierarchical_v2")
class HierarchicalDualStreamBiGRUv2(BaseFallModel):
    """Three-branch (B)GRU over top/bottom keypoints and engineered features."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", config)

        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.4))
        num_classes = int(model_cfg.get("num_classes", NUM_CLASSES))
        bidirectional = bool(model_cfg.get("bidirectional", True))
        use_layernorm = bool(model_cfg.get("use_layernorm", True))
        self.standardize_features = bool(model_cfg.get("standardize_features", True))

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Standardization buffers for the engineered feature stream. Saved in
        # the checkpoint state_dict and reused at inference. Defaults are a
        # no-op (mean 0 / std 1) until set_feature_stats() populates them.
        self.register_buffer("feat_mean", torch.zeros(FEAT_DIM))
        self.register_buffer("feat_std", torch.ones(FEAT_DIM))
        self.register_buffer("stats_initialized", torch.zeros(1))

        self.gru_top = GRUBranch(TOP_KPT_DIM, hidden_dim, num_layers, dropout, bidirectional, use_layernorm)
        self.gru_bot = GRUBranch(BOTTOM_KPT_DIM, hidden_dim, num_layers, dropout, bidirectional, use_layernorm)
        self.gru_feat = GRUBranch(FEAT_DIM, hidden_dim, num_layers, dropout, bidirectional, use_layernorm)

        branch_dim = self.gru_top.output_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(branch_dim * 3, num_classes),
        )

    @property
    def model_name(self) -> str:
        return getattr(self, "_registry_name", "bigru_hierarchical_v2")

    @torch.no_grad()
    def set_feature_stats(self, mean: Tensor, std: Tensor, eps: float = 1e-6) -> None:
        """Populate feature standardization buffers from train statistics."""
        mean = torch.as_tensor(mean, dtype=self.feat_mean.dtype, device=self.feat_mean.device)
        std = torch.as_tensor(std, dtype=self.feat_std.dtype, device=self.feat_std.device)
        self.feat_mean.copy_(mean.reshape(-1))
        self.feat_std.copy_(std.reshape(-1).clamp_min(eps))
        self.stats_initialized.fill_(1.0)

    @property
    def feature_stats_ready(self) -> bool:
        return bool(self.stats_initialized.item() > 0)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        top_kp = batch["top_kp"]
        bot_kp = batch["bot_kp"]
        feat = batch["feat"]
        mask = batch["mask"]

        if self.standardize_features:
            feat = (feat - self.feat_mean) / self.feat_std
            # Keep padded/invalid frames at zero so attention masking is clean.
            feat = feat * mask.unsqueeze(-1)

        top_repr = self.gru_top(top_kp, mask)
        bot_repr = self.gru_bot(bot_kp, mask)
        feat_repr = self.gru_feat(feat, mask)

        fused = torch.cat([top_repr, bot_repr, feat_repr], dim=-1)
        return self.classifier(fused)
