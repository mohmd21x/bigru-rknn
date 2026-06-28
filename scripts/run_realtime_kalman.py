#!/usr/bin/env python3
"""Production realtime fall detection with causal Kalman upsampling.

Pipeline:
  RTSP/video -> pose backend (~7 fps) -> 2s rolling buffer ->
  causal KF upsample (64 frames @ 30 fps) -> PoseFeatureExtractor -> BiGRU
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kalman_upsample import causal_upsample_measurements  # noqa: E402
from src.constants import (
    BOTTOM_KPT_DIM,
    DEFAULT_WINDOW_SIZE,
    FEAT_DIM,
    TOP_KPT_DIM,
)
from src.inference.fall_predictor import FallPredictor
from src.inference.pose_backend import add_pose_backend_args, create_pose_backend
from src.inference.pose_features import PoseFeatureExtractor
from src.inference.visualize import draw_hud, draw_pose

TARGET_FPS = 30.0
WINDOW_SEC = DEFAULT_WINDOW_SIZE / TARGET_FPS


class SparsePoseBuffer:
    """Rolling buffer of (timestamp, keypoints_17x3) from sparse pose at ~7 fps."""

    def __init__(self, window_sec: float) -> None:
        self.window_sec = float(window_sec)
        self._items: deque[tuple[float, np.ndarray | None]] = deque()

    def push(self, timestamp: float, keypoints: np.ndarray | None) -> None:
        self._items.append((float(timestamp), keypoints))
        t_min = timestamp - self.window_sec
        while self._items and self._items[0][0] < t_min - 1e-9:
            self._items.popleft()

    def snapshot(self, t_end: float) -> list[tuple[float, np.ndarray | None]]:
        """Return measurements in [t_end - window_sec, t_end], sorted by time."""
        t_min = t_end - self.window_sec
        return [(t, kp) for t, kp in self._items if t_min - 1e-9 <= t <= t_end + 1e-9]


class CausalWindowUpsampler:
    """Stateless causal Kalman upsampler — fresh instance per prediction."""

    def __init__(
        self,
        *,
        target_fps: float = TARGET_FPS,
        process_noise: float = 20.0,
        meas_noise_base: float = 10.0,
        max_gap_sec: float = 0.5,
        conf_decay_sec: float = 0.3,
    ) -> None:
        self.target_fps = target_fps
        self.process_noise = process_noise
        self.meas_noise_base = meas_noise_base
        self.max_gap_sec = max_gap_sec
        self.conf_decay_sec = conf_decay_sec

    def upsample(
        self,
        measurements: list[tuple[float, np.ndarray | None]],
        t_start: float,
        t_end: float,
    ) -> list[tuple[float, np.ndarray]]:
        return causal_upsample_measurements(
            measurements,
            t_start,
            t_end,
            target_fps=self.target_fps,
            process_noise=self.process_noise,
            meas_noise_base=self.meas_noise_base,
            max_gap_sec=self.max_gap_sec,
            conf_decay_sec=self.conf_decay_sec,
        )


class VideoFrameSampler:
    """Map decoded video frames to sparse pose samples at ``process_fps``."""

    def __init__(self, source_fps: float, process_fps: float) -> None:
        self.source_fps = max(float(source_fps), 1e-3)
        self.process_fps = max(float(process_fps), 1e-3)
        self._next_sample_time = 0.0
        self.sample_count = 0

    def should_process(self, video_frame_idx: int) -> tuple[bool, float]:
        video_time = video_frame_idx / self.source_fps
        sample_interval = 1.0 / self.process_fps
        frame_mid = video_time + (0.5 / self.source_fps)

        if frame_mid + 1e-9 >= self._next_sample_time:
            timestamp = self._next_sample_time
            self._next_sample_time += sample_interval
            self.sample_count += 1
            return True, timestamp
        return False, video_time


def build_batch(
    feature_frames: list[dict[str, np.ndarray | float | bool]],
    window_size: int,
) -> dict[str, np.ndarray]:
    top = np.stack([frame["top_kp"] for frame in feature_frames], axis=0)
    bot = np.stack([frame["bot_kp"] for frame in feature_frames], axis=0)
    feat = np.stack([frame["feat"] for frame in feature_frames], axis=0)
    mask = np.array([frame["mask"] for frame in feature_frames], dtype=np.float32)
    return {
        "top_kp": top.reshape(1, window_size, TOP_KPT_DIM),
        "bot_kp": bot.reshape(1, window_size, BOTTOM_KPT_DIM),
        "feat": feat.reshape(1, window_size, FEAT_DIM),
        "mask": mask.reshape(1, window_size),
    }


def valid_ratio(
    feature_frames: list[dict[str, np.ndarray | float | bool]],
    window_size: int,
) -> float:
    if not feature_frames:
        return 0.0
    valid = sum(float(frame["mask"]) for frame in feature_frames)
    return valid / window_size


def resolve_export_path(
    export: bool | str | Path | None,
    source: str,
) -> Path | None:
    if export is False:
        return None

    stem = Path(source).stem if not source.isdigit() and "://" not in source else "capture"
    default_path = REPO_ROOT / "reports" / "realtime_kalman" / f"{stem}_labeled.mp4"

    if export is True:
        return default_path

    chosen = Path(export)
    if chosen.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        return chosen
    return chosen / f"{stem}_labeled.mp4"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime fall detection with causal Kalman upsampling (RTSP/video).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or RTSP URL.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bigru_hierarchical.yaml",
        help="Training config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/bigru_hierarchical/best.pt",
        help="Fall classifier checkpoint.",
    )
    add_pose_backend_args(parser)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO/fall model (default: cuda if available).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Temporal window length for fall model (default: 64).",
    )
    parser.add_argument(
        "--pose-fps",
        type=float,
        default=7.0,
        help="Sparse pose sampling rate in video seconds (default: 7).",
    )
    parser.add_argument(
        "--predict-stride-sec",
        type=float,
        default=1.0,
        help="Seconds between BiGRU predictions (default: 1.0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source video FPS when metadata is missing/wrong.",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=20.0,
        help="Kalman process noise (default: 20).",
    )
    parser.add_argument(
        "--meas-noise-base",
        type=float,
        default=10.0,
        help="Kalman measurement noise base (default: 10).",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=0.5,
        help="Past-only absence gap in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--conf-decay-sec",
        type=float,
        default=0.3,
        help="Confidence decay time constant (default: 0.3).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable OpenCV display window.",
    )
    parser.add_argument(
        "--export",
        nargs="?",
        default=False,
        const=True,
        metavar="PATH",
        help="Write labeled output video. Implies --no-show.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_path = resolve_export_path(args.export, args.source)
    if export_path is not None:
        args.no_show = True

    window_size = args.window_size
    window_sec = window_size / TARGET_FPS

    if args.predict_stride_sec <= 0:
        print("Error: --predict-stride-sec must be > 0", file=sys.stderr)
        return 1
    if args.pose_fps <= 0:
        print("Error: --pose-fps must be > 0", file=sys.stderr)
        return 1

    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pose_backend = create_pose_backend(args, device=device)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Device: {device}")
    print(f"Pose backend: {args.pose_backend}")
    print(f"Fall checkpoint: {args.checkpoint}")

    predictor = FallPredictor(args.checkpoint, config_path=args.config, device=device)
    extractor = PoseFeatureExtractor()
    pose_buffer = SparsePoseBuffer(window_sec=window_sec)
    upsampler = CausalWindowUpsampler(
        target_fps=TARGET_FPS,
        process_noise=args.process_noise,
        meas_noise_base=args.meas_noise_base,
        max_gap_sec=args.max_gap_sec,
        conf_decay_sec=args.conf_decay_sec,
    )

    cap = open_capture(args.source)
    source_fps = probe_source_fps(cap, args.source, args.fps)
    sampler = VideoFrameSampler(source_fps=source_fps, process_fps=args.pose_fps)
    print(
        f"Source FPS: {source_fps:.2f} | Pose: {args.pose_fps:.1f} fps | "
        f"Window: {window_size} frames @ {TARGET_FPS:.0f} fps ({window_sec:.2f}s) | "
        f"Predict every {args.predict_stride_sec:.2f}s"
    )

    writer = None
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(export_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        print(f"Exporting labeled video to: {export_path}")

    label = "not_fall"
    confidence = 0.0
    fall_prob = 0.0
    pose_ms = 0.0
    fall_ms = 0.0
    video_frame_idx = -1
    video_time = 0.0
    fps_value = 0.0
    last_tick = time.perf_counter()
    last_predict_time = -float("inf")
    last_kpts: np.ndarray | None = None
    window_fill = 0

    window_name = "Fall Detection (Kalman)"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_frame_idx += 1
            video_time = video_frame_idx / source_fps
            should_run_pose, sample_time = sampler.should_process(video_frame_idx)

            if should_run_pose:
                t0 = time.perf_counter()
                keypoints = pose_backend.predict(frame)
                pose_ms = (time.perf_counter() - t0) * 1000.0
                if keypoints is not None:
                    last_kpts = keypoints
                pose_buffer.push(sample_time, keypoints)

            if video_time - last_predict_time >= args.predict_stride_sec:
                last_predict_time = video_time
                t_end = video_time
                t_start = t_end - window_sec
                measurements = pose_buffer.snapshot(t_end)

                if measurements:
                    frames_30fps = upsampler.upsample(measurements, t_start, t_end)
                    window_fill = len(frames_30fps)

                    if len(frames_30fps) >= window_size:
                        frames_30fps = frames_30fps[-window_size:]
                        extractor.reset()
                        feature_frames = [
                            extractor.update(kp.astype(np.float32), t)
                            for t, kp in frames_30fps
                        ]
                        t1 = time.perf_counter()
                        prediction = predictor.predict(
                            build_batch(feature_frames, window_size),
                            valid_ratio=valid_ratio(feature_frames, window_size),
                        )
                        fall_ms = (time.perf_counter() - t1) * 1000.0
                        label = prediction["label"]
                        confidence = prediction["confidence"]
                        fall_prob = prediction["fall_prob"]

            skeleton_color = (0, 0, 255) if label == "fall" else (0, 220, 0)
            if last_kpts is not None:
                draw_pose(frame, last_kpts, skeleton_color)

            now = time.perf_counter()
            fps_value = (
                0.9 * fps_value + 0.1 * (1.0 / max(now - last_tick, 1e-6))
                if video_frame_idx > 0
                else 0.0
            )
            last_tick = now

            draw_hud(
                frame,
                label=label,
                confidence=confidence,
                fall_prob=fall_prob,
                pose_ms=pose_ms,
                fall_ms=fall_ms,
                fps=fps_value,
                source_fps=TARGET_FPS,
                video_time=video_time,
                window_fill=window_fill,
                window_size=window_size,
                source=args.source,
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(
        f"Processed {video_frame_idx + 1} video frames "
        f"({sampler.sample_count} pose samples @ {args.pose_fps:.1f} fps). "
        f"Last label={label} conf={confidence:.3f} fall_prob={fall_prob:.3f}"
    )
    if export_path is not None:
        print(f"Labeled video saved: {export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
