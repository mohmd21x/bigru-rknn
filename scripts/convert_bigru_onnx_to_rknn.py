#!/usr/bin/env python3
"""Convert a BiGRU ONNX model to RKNN for Rockchip NPU deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import BOTTOM_KPT_DIM, FEAT_DIM, TOP_KPT_DIM

INPUT_NAMES = ("top_kp", "bot_kp", "feat", "mask")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BiGRU ONNX to RKNN (requires rknn-toolkit2).",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--rknn",
        type=Path,
        required=True,
        help="Output RKNN model path.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="rk3588",
        help="Rockchip target platform (default: rk3588).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Sliding-window length T (default: infer from ONNX or checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional .pt checkpoint used to read window_size when ONNX has dynamic T.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (requires --dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Calibration dataset file for INT8 quantization.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RKNN logging.",
    )
    return parser.parse_args(argv)


def infer_window_size_from_onnx(onnx_path: Path) -> int | None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to infer window_size from ONNX") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for inp in session.get_inputs():
        if inp.name != "top_kp":
            continue
        shape = inp.shape
        if len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])
    return None


def infer_window_size_from_checkpoint(checkpoint_path: Path) -> int:
    import torch

    ckpt = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {checkpoint_path}")
    return int(config["data"]["window_size"])


def resolve_window_size(
    onnx_path: Path,
    window_size: int | None,
    checkpoint_path: Path | None,
) -> int:
    if window_size is not None:
        return window_size

    inferred = infer_window_size_from_onnx(onnx_path)
    if inferred is not None:
        return inferred

    if checkpoint_path is not None:
        return infer_window_size_from_checkpoint(checkpoint_path)

    raise ValueError(
        "Could not infer window_size from ONNX. Pass --window-size or --checkpoint."
    )


def convert_bigru_onnx_to_rknn(
    onnx_path: Path,
    rknn_path: Path,
    *,
    platform: str = "rk3588",
    window_size: int | None = None,
    checkpoint_path: Path | None = None,
    quantize: bool = False,
    dataset_path: Path | None = None,
    verbose: bool = False,
) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is not installed. Install it on your Rockchip host/board, then retry."
        ) from exc

    onnx_path = onnx_path.resolve()
    rknn_path = rknn_path.resolve()
    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if quantize and dataset_path is None:
        raise ValueError("--quantize requires --dataset")

    t = resolve_window_size(onnx_path, window_size, checkpoint_path)
    input_size_list = [
        [1, t, TOP_KPT_DIM],
        [1, t, BOTTOM_KPT_DIM],
        [1, t, FEAT_DIM],
        [1, t],
    ]

    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(
            target_platform=platform,
            quantized_dtype="asymmetric_quantized-8",
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=list(INPUT_NAMES),
            input_size_list=input_size_list,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        build_kwargs: dict = {"do_quantization": quantize}
        if quantize:
            build_kwargs["dataset"] = str(dataset_path.resolve())

        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()

    print(f"Converted RKNN: {rknn_path}")
    print(f"  source onnx : {onnx_path}")
    print(f"  platform    : {platform}")
    print(f"  window_size : {t}")
    print(f"  quantize    : {quantize}")
    print("  inputs      : "
          f"top_kp {input_size_list[0]}, "
          f"bot_kp {input_size_list[1]}, "
          f"feat {input_size_list[2]}, "
          f"mask {input_size_list[3]}")
    print("  output      : logits (1, 2)")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    convert_bigru_onnx_to_rknn(
        args.onnx,
        args.rknn,
        platform=args.platform,
        window_size=args.window_size,
        checkpoint_path=args.checkpoint,
        quantize=args.quantize,
        dataset_path=args.dataset,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
