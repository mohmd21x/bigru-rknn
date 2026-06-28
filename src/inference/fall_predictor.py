"""Load BiGRU checkpoint and run window-level fall prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.constants import CLASS_NAMES, ID_TO_LABEL
from src.models.base import build_model


class FallPredictor:
    """Realtime fall / not_fall classifier from a trained checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        config_path: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config")
        if not isinstance(config, dict) and config_path is not None:
            with Path(config_path).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Checkpoint does not contain config; pass --config")

        self.config = config
        model_name = checkpoint.get("model_name") or config.get("model", {}).get("name", "bigru_hierarchical")
        self.model = build_model(model_name, config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.min_valid_frame_ratio = float(config.get("data", {}).get("min_valid_frame_ratio", 0.5))

    @torch.no_grad()
    def predict(self, batch: dict[str, np.ndarray], valid_ratio: float) -> dict[str, Any]:
        """Return label, class probabilities, and fall probability."""
        if valid_ratio < self.min_valid_frame_ratio:
            return {
                "label": "not_fall",
                "label_id": 0,
                "confidence": 0.0,
                "probs": np.array([1.0, 0.0], dtype=np.float32),
                "fall_prob": 0.0,
                "ready": False,
            }

        tensor_batch = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in batch.items()
            if key in {"top_kp", "bot_kp", "feat", "mask"}
        }
        logits = self.model(tensor_batch)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        label_id = int(np.argmax(probs))
        label = ID_TO_LABEL[label_id]
        return {
            "label": label,
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probs": probs,
            "fall_prob": float(probs[CLASS_NAMES.index("fall")]),
            "ready": True,
        }
