#!/usr/bin/env python3
"""Benchmark ONNX vs RKNN fall classifiers on shared pose-feature windows.

Pipeline (headless):
  pose backend (YOLO or RTMO) -> PoseFeatureExtractor (python) -> FallFrameBuffer -> ONNX + RKNN

On Rockchip (no ultralytics): use --pose-backend rtmo --rtmo-onnx rtmo/rtmo-s.onnx
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *args, **kwargs: None  # headless OpenCV lacks GUI APIs; ultralytics needs this

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import CLASS_NAMES, DEFAULT_WINDOW_SIZE, ID_TO_LABEL
from src.inference.frame_buffer import FallFrameBuffer
from src.inference.pose_backend import add_pose_backend_args, create_pose_backend
from src.inference.pose_features import PoseFeatureExtractor


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _label_from_fall_prob(fall_prob: float, threshold: float) -> str:
    return "fall" if fall_prob >= threshold else "not_fall"


def _pearson_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class OnnxFallPredictor:
    """Run BiGRU fall classification via ONNX Runtime (CPU)."""

    def __init__(
        self,
        onnx_path: Path | str,
        min_valid_frame_ratio: float = 0.5,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for OnnxFallPredictor. Install with: pip install onnxruntime"
            ) from exc

        self.min_valid_frame_ratio = float(min_valid_frame_ratio)
        self.session = ort.InferenceSession(
            str(Path(onnx_path).resolve()),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def predict(self, batch: dict[str, np.ndarray], valid_ratio: float) -> dict[str, Any]:
        if valid_ratio < self.min_valid_frame_ratio:
            return {
                "label": "not_fall",
                "label_id": 0,
                "confidence": 0.0,
                "probs": np.array([1.0, 0.0], dtype=np.float32),
                "fall_prob": 0.0,
                "ready": False,
            }

        feed = {
            name: np.ascontiguousarray(batch[name], dtype=np.float32)
            for name in self.input_names
            if name in batch
        }
        outputs = self.session.run(None, feed)
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        probs = _softmax(logits)
        label_id = int(np.argmax(probs))
        return {
            "label": ID_TO_LABEL[label_id],
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probs": probs,
            "fall_prob": float(probs[CLASS_NAMES.index("fall")]),
            "ready": True,
        }


class RknnFallPredictor:
    """Run BiGRU fall classification via RKNN Lite (board) or RKNN Toolkit (PC sim)."""

    def __init__(
        self,
        rknn_path: Path | str,
        min_valid_frame_ratio: float = 0.5,
    ) -> None:
        self.min_valid_frame_ratio = float(min_valid_frame_ratio)
        self.rknn_path = Path(rknn_path).resolve()
        self.backend = self._init_runtime()

    def _init_runtime(self) -> str:
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            RKNNLite = None  # type: ignore[misc, assignment]

        if RKNNLite is not None:
            rknn = RKNNLite()
            ret = rknn.load_rknn(str(self.rknn_path))
            if ret == 0:
                ret = rknn.init_runtime()
                if ret == 0:
                    self._rknn = rknn
                    return "rknnlite"

        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise ImportError(
                "Neither rknnlite nor rknn-toolkit2 is installed. "
                "Install rknnlite2 on the board or rknn-toolkit2 on the host."
            ) from exc

        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(str(self.rknn_path))
        if ret != 0:
            raise RuntimeError(f"RKNN.load_rknn failed with code {ret}")
        ret = rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"RKNN.init_runtime failed with code {ret}")
        self._rknn = rknn
        return "rknn-toolkit"

    def release(self) -> None:
        release = getattr(self._rknn, "release", None)
        if callable(release):
            release()

    def predict(self, batch: dict[str, np.ndarray], valid_ratio: float) -> dict[str, Any]:
        if valid_ratio < self.min_valid_frame_ratio:
            return {
                "label": "not_fall",
                "label_id": 0,
                "confidence": 0.0,
                "probs": np.array([1.0, 0.0], dtype=np.float32),
                "fall_prob": 0.0,
                "ready": False,
            }

        inputs = [
            np.ascontiguousarray(batch["top_kp"], dtype=np.float32),
            np.ascontiguousarray(batch["bot_kp"], dtype=np.float32),
            np.ascontiguousarray(batch["feat"], dtype=np.float32),
            np.ascontiguousarray(batch["mask"], dtype=np.float32),
        ]
        outputs = self._rknn.inference(inputs=inputs)
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        probs = _softmax(logits)
        label_id = int(np.argmax(probs))
        return {
            "label": ID_TO_LABEL[label_id],
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probs": probs,
            "fall_prob": float(probs[CLASS_NAMES.index("fall")]),
            "ready": True,
        }


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def probe_source_fps(cap: cv2.VideoCapture, source: str, override: float | None) -> float:
    if override is not None and override > 0:
        return float(override)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 1.0:
        return fps

    if "://" not in source and not source.isdigit():
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if frame_count > 1:
            saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            duration_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
            if duration_ms > 0:
                fps = frame_count / (duration_ms / 1000.0)
                if fps > 1.0:
                    return fps

    return 25.0


class VideoFrameSampler:
    """Map decoded video frames to model frames at ``process_fps`` per video second."""

    def __init__(self, source_fps: float, process_fps: float | None = None) -> None:
        self.source_fps = max(float(source_fps), 1e-3)
        self.process_fps = max(float(process_fps or source_fps), 1e-3)
        self._next_sample_time = 0.0
        self.model_frame_count = 0

    def should_process(self, video_frame_idx: int) -> tuple[bool, float]:
        video_time = video_frame_idx / self.source_fps
        sample_interval = 1.0 / self.process_fps
        frame_mid = video_time + (0.5 / self.source_fps)

        if frame_mid + 1e-9 >= self._next_sample_time:
            timestamp = self._next_sample_time
            self._next_sample_time += sample_interval
            self.model_frame_count += 1
            return True, timestamp
        return False, video_time


@dataclass
class WindowResult:
    win: int
    video_t: float
    onnx_fall_p: float
    rknn_fall_p: float
    onnx_label: str
    rknn_label: str

    @property
    def delta(self) -> float:
        return self.rknn_fall_p - self.onnx_fall_p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX vs RKNN fall classifiers on shared pose windows.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / "weights/bigru_hierarchical.onnx",
        help="ONNX fall classifier path.",
    )
    parser.add_argument(
        "--rknn",
        type=Path,
        default=REPO_ROOT / "weights/bigru_hierarchical.rknn",
        help="RKNN fall classifier path.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input video paths.",
    )
    add_pose_backend_args(parser)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO pose (default: cuda if available).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Temporal window length for fall model.",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=None,
        help="Real frames before predicting (default: window-size).",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run fall models every N processed frames once the window is full.",
    )
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.5,
        help="Decision threshold on fall_prob for label assignment.",
    )
    parser.add_argument(
        "--min-valid-frame-ratio",
        type=float,
        default=0.5,
        help="Minimum valid-frame ratio before running inference.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source video FPS when metadata is missing/wrong.",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=None,
        help="Frames per second of video to feed the fall model (default: source FPS).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "test/output",
        help="Directory for per-video CSV exports.",
    )
    return parser.parse_args(argv)


def benchmark_video(
    video_path: Path,
    *,
    pose_backend: Any,
    onnx_predictor: OnnxFallPredictor,
    rknn_predictor: RknnFallPredictor,
    window_size: int,
    clip_frames: int | None,
    infer_every: int,
    fall_threshold: float,
    fps_override: float | None,
    process_fps: float | None,
) -> list[WindowResult]:
    cap = open_capture(str(video_path))
    source_fps = probe_source_fps(cap, str(video_path), fps_override)
    effective_process_fps = process_fps if process_fps and process_fps > 0 else source_fps
    sampler = VideoFrameSampler(source_fps=source_fps, process_fps=effective_process_fps)

    feature_extractor = PoseFeatureExtractor()
    frame_buffer = FallFrameBuffer(window_size=window_size, clip_frames=clip_frames)

    results: list[WindowResult] = []
    video_frame_idx = -1
    model_frame_idx = 0
    window_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_frame_idx += 1
            should_process, feature_timestamp = sampler.should_process(video_frame_idx)
            keypoints = pose_backend.predict(frame)

            if not should_process:
                continue

            model_frame_idx += 1

            if keypoints is not None:
                frame_features = feature_extractor.update(keypoints, feature_timestamp)
                frame_buffer.add(frame_features)
            else:
                feature_extractor.reset()
                frame_buffer.reset()
                continue

            if not frame_buffer.is_ready:
                continue
            if model_frame_idx % max(infer_every, 1) != 0:
                continue

            batch = frame_buffer.as_batch()
            valid_ratio = frame_buffer.valid_ratio()
            onnx_pred = onnx_predictor.predict(batch, valid_ratio)
            rknn_pred = rknn_predictor.predict(batch, valid_ratio)

            if not onnx_pred.get("ready", False) or not rknn_pred.get("ready", False):
                continue

            window_idx += 1
            onnx_fall_p = float(onnx_pred["fall_prob"])
            rknn_fall_p = float(rknn_pred["fall_prob"])
            results.append(
                WindowResult(
                    win=window_idx,
                    video_t=feature_timestamp,
                    onnx_fall_p=onnx_fall_p,
                    rknn_fall_p=rknn_fall_p,
                    onnx_label=_label_from_fall_prob(onnx_fall_p, fall_threshold),
                    rknn_label=_label_from_fall_prob(rknn_fall_p, fall_threshold),
                )
            )
    finally:
        cap.release()

    return results


def print_results_table(video_path: Path, results: list[WindowResult]) -> None:
    print(f"\n=== {video_path.name} ({len(results)} windows) ===")
    header = f"{'win':>4}  {'video_t':>8}  {'onnx_fall_p':>12}  {'rknn_fall_p':>12}  {'delta':>10}  {'onnx_label':>10}  {'rknn_label':>10}"
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row.win:4d}  {row.video_t:8.3f}  {row.onnx_fall_p:12.6f}  {row.rknn_fall_p:12.6f}  "
            f"{row.delta:+10.6f}  {row.onnx_label:>10}  {row.rknn_label:>10}"
        )

    if not results:
        print("(no windows with sufficient valid frames)")
        return

    deltas = [abs(r.delta) for r in results]
    onnx_probs = [r.onnx_fall_p for r in results]
    rknn_probs = [r.rknn_fall_p for r in results]
    mae = float(np.mean(deltas))
    max_delta = float(np.max(deltas))
    corr = _pearson_corr(onnx_probs, rknn_probs)
    label_matches = sum(1 for r in results if r.onnx_label == r.rknn_label)
    match_rate = label_matches / len(results)

    print("-" * len(header))
    print(
        f"summary: MAE={mae:.6f}  max|delta|={max_delta:.6f}  "
        f"pearson={corr:.6f}  label_match={match_rate * 100:.1f}% ({label_matches}/{len(results)})"
    )


def write_results_csv(video_path: Path, results: list[WindowResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"bench_{video_path.stem}_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["win", "video_t", "onnx_fall_p", "rknn_fall_p", "delta", "onnx_label", "rknn_label"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "win": row.win,
                    "video_t": f"{row.video_t:.6f}",
                    "onnx_fall_p": f"{row.onnx_fall_p:.6f}",
                    "rknn_fall_p": f"{row.rknn_fall_p:.6f}",
                    "delta": f"{row.delta:+.6f}",
                    "onnx_label": row.onnx_label,
                    "rknn_label": row.rknn_label,
                }
            )

        if results:
            deltas = [abs(r.delta) for r in results]
            onnx_probs = [r.onnx_fall_p for r in results]
            rknn_probs = [r.rknn_fall_p for r in results]
            label_matches = sum(1 for r in results if r.onnx_label == r.rknn_label)
            writer.writerow(
                {
                    "win": "summary",
                    "video_t": "",
                    "onnx_fall_p": f"mae={np.mean(deltas):.6f}",
                    "rknn_fall_p": f"max_abs_delta={np.max(deltas):.6f}",
                    "delta": f"pearson={_pearson_corr(onnx_probs, rknn_probs):.6f}",
                    "onnx_label": f"label_match={label_matches}/{len(results)}",
                    "rknn_label": f"rate={label_matches / len(results):.4f}",
                }
            )

    return csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.onnx.is_file():
        print(f"Error: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 1
    if not args.rknn.is_file():
        print(f"Error: RKNN model not found: {args.rknn}", file=sys.stderr)
        return 1

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pose_backend = create_pose_backend(args, device=device)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Device: {device}")
    print(f"Pose backend: {args.pose_backend}")
    print(f"ONNX: {args.onnx}")
    print(f"RKNN: {args.rknn}")

    onnx_predictor = OnnxFallPredictor(args.onnx, min_valid_frame_ratio=args.min_valid_frame_ratio)
    rknn_predictor = RknnFallPredictor(args.rknn, min_valid_frame_ratio=args.min_valid_frame_ratio)
    print(f"RKNN runtime: {rknn_predictor.backend}")

    exit_code = 0
    try:
        for video_path in args.video:
            resolved = video_path.resolve()
            if not resolved.is_file():
                print(f"Error: video not found: {resolved}", file=sys.stderr)
                exit_code = 1
                continue

            t0 = time.perf_counter()
            results = benchmark_video(
                resolved,
                pose_backend=pose_backend,
                onnx_predictor=onnx_predictor,
                rknn_predictor=rknn_predictor,
                window_size=args.window_size,
                clip_frames=args.clip_frames,
                infer_every=args.infer_every,
                fall_threshold=args.fall_threshold,
                fps_override=args.fps,
                process_fps=args.process_fps,
            )
            elapsed = time.perf_counter() - t0

            print_results_table(resolved, results)
            csv_path = write_results_csv(resolved, results, args.output_dir)
            print(f"CSV: {csv_path}")
            print(f"Elapsed: {elapsed:.2f}s")
    finally:
        rknn_predictor.release()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
