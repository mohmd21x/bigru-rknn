"""Unified pose backend interface for YOLOv8 and RTMO-S."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np

from src.inference.rtmo_pose import (
    DEFAULT_ONNX,
    create_ort_session,
    infer_rtmo_s,
    pick_rtmo_person_keypoints,
    resolve_rtmo_onnx,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YOLO_CONF = 0.35
DEFAULT_RTMO_CONF = 0.5


class PoseBackend(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return COCO-17 keypoints as (17, 3) float32 [x, y, conf], or None."""
        ...


def resolve_yolo_weights(path: Path) -> Path:
    """Return an existing YOLO pose weights path, with common fallbacks."""
    candidates = [
        path,
        path.with_name("yolov8n-pose.pt"),
        REPO_ROOT / "weights/yolov8n-pose.pt",
        REPO_ROOT / "weights/yolo8n.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "YOLO pose weights not found. Place the pose model at "
        f"{path} (expected a YOLOv8 *pose* checkpoint, e.g. yolov8n-pose.pt renamed to yolo8n.pt)."
    )


def pick_person_keypoints(result, conf_threshold: float) -> np.ndarray | None:
    """Select the highest-confidence person detection from a YOLO pose result."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None

    confs = result.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    if float(confs[best_idx]) < conf_threshold:
        return None

    kpts = result.keypoints.data.detach().cpu().numpy()
    if best_idx >= len(kpts):
        return None
    return np.asarray(kpts[best_idx], dtype=np.float32).reshape(17, 3)


class YoloPoseBackend:
    def __init__(self, weights: Path, device: str, conf: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for the YOLO pose backend. Install with: pip install ultralytics"
            ) from exc

        self._conf = conf
        self._device = device
        self._yolo = YOLO(str(weights))
        self.weights = weights

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        results = self._yolo.predict(frame_bgr, verbose=False, device=self._device)
        return pick_person_keypoints(results[0], self._conf)


class RtmoPoseBackend:
    def __init__(self, onnx_path: Path, conf: float) -> None:
        self.onnx_path = resolve_rtmo_onnx(onnx_path)
        self._conf = conf
        self._session = create_ort_session(self.onnx_path)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        poses = infer_rtmo_s(frame_bgr, self._session, score_thr=self._conf)
        return pick_rtmo_person_keypoints(poses, self._conf)


def add_pose_backend_args(parser: argparse.ArgumentParser) -> None:
    """Register shared pose-backend CLI flags on an ArgumentParser."""
    parser.add_argument(
        "--pose-backend",
        choices=("yolo", "rtmo"),
        default="yolo",
        help="Pose estimation backend (default: yolo).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=REPO_ROOT / "weights/yolo8n.pt",
        help="YOLOv8 pose weights (YOLO backend only).",
    )
    parser.add_argument(
        "--rtmo-onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help="RTMO-S ONNX model path (RTMO backend only, default: rtmo/rtmo-s.onnx).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=(
            "Person detection confidence threshold. "
            f"Default: {DEFAULT_YOLO_CONF} for yolo, {DEFAULT_RTMO_CONF} for rtmo."
        ),
    )


def resolve_pose_conf(args: argparse.Namespace) -> float:
    if args.conf is not None:
        return float(args.conf)
    if getattr(args, "pose_backend", "yolo") == "rtmo":
        return DEFAULT_RTMO_CONF
    return DEFAULT_YOLO_CONF


def create_pose_backend(args: argparse.Namespace, device: str | None = None) -> PoseBackend:
    """Build a pose backend from parsed CLI arguments."""
    conf = resolve_pose_conf(args)
    backend = getattr(args, "pose_backend", "yolo")

    if backend == "rtmo":
        return RtmoPoseBackend(
            onnx_path=Path(args.rtmo_onnx).resolve(),
            conf=conf,
        )

    import torch

    resolved_device = device or getattr(args, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weights = resolve_yolo_weights(Path(args.yolo_weights).resolve())
    return YoloPoseBackend(weights=weights, device=resolved_device, conf=conf)
