#!/usr/bin/env python3
"""Export a trained BiGRU checkpoint to ONNX for RKNN conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import BOTTOM_KPT_DIM, FEAT_DIM, TOP_KPT_DIM
from src.models.base import build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a BiGRU fall-classifier checkpoint to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14).",
    )
    return parser.parse_args(argv)


class BiGRUOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        top_kp: torch.Tensor,
        bot_kp: torch.Tensor,
        feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            {
                "top_kp": top_kp,
                "bot_kp": bot_kp,
                "feat": feat,
                "mask": mask,
            }
        )


def export_bigru_onnx(checkpoint_path: Path, onnx_path: Path, opset: int = 14) -> int:
    checkpoint_path = checkpoint_path.resolve()
    onnx_path = onnx_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {checkpoint_path}")

    model_name = ckpt.get("model_name", config.get("model", {}).get("name", "bigru_hierarchical"))
    window_size = int(config["data"]["window_size"])

    model = build_model(model_name, config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    top_kp = torch.zeros(1, window_size, TOP_KPT_DIM)
    bot_kp = torch.zeros(1, window_size, BOTTOM_KPT_DIM)
    feat = torch.zeros(1, window_size, FEAT_DIM)
    mask = torch.ones(1, window_size)

    wrapper = BiGRUOnnxWrapper(model)
    torch.onnx.export(
        wrapper,
        (top_kp, bot_kp, feat, mask),
        str(onnx_path),
        input_names=["top_kp", "bot_kp", "feat", "mask"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={
            "top_kp": {0: "batch"},
            "bot_kp": {0: "batch"},
            "feat": {0: "batch"},
            "mask": {0: "batch"},
        },
    )

    print(f"Exported ONNX: {onnx_path}")
    print(f"  model      : {model_name}")
    print(f"  window_size: {window_size}")
    print(f"  inputs     : top_kp (1,{window_size},{TOP_KPT_DIM}), "
          f"bot_kp (1,{window_size},{BOTTOM_KPT_DIM}), "
          f"feat (1,{window_size},{FEAT_DIM}), mask (1,{window_size})")
    print(f"  output     : logits (1, 2)")
    return window_size


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    export_bigru_onnx(args.checkpoint, args.onnx, opset=args.opset)


if __name__ == "__main__":
    main()
