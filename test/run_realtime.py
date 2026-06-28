#!/usr/bin/env python3
"""Realtime fall detection demo for video files and RTSP streams.

Pipeline:
  pose backend (YOLO or RTMO) -> online pose features -> BiGRU window classifier
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import DEFAULT_WINDOW_SIZE
from src.inference.fall_predictor import FallPredictor
from src.inference.frame_buffer import FallFrameBuffer
from src.inference.kalman_pose_interpolator import (
    DEFAULT_MIN_KPT_CONF,
    DEFAULT_MIN_VALID_KEYPOINTS,
    KalmanPoseInterpolator,
    is_valid_pose_detection,
)
from src.inference.pose_backend import add_pose_backend_args, create_pose_backend
from src.inference.pose_features import PoseFeatureExtractor
from src.inference.visualize import draw_hud, draw_pose
from src.constants import BOTTOM_JOINT_INDICES, ENGINEERED_FEATURE_COLUMNS, TOP_JOINT_INDICES

DEFAULT_CPP_EXTRACTOR = REPO_ROOT / "pose_features/build/extract_pose_features_from_csv"


def _parse_csv_float(value: str) -> float:
    text = (value or "").strip().lower()
    if not text or text == "nan":
        return float("nan")
    return float(text)


def _build_subprocess_env(lib_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if lib_dir is None:
        return env
    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = str(lib_dir)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


class CppPoseFeatureExtractor:
    """Realtime wrapper around the C++ CSV feature extractor."""

    def __init__(self, extractor: Path, extractor_lib_dir: Path | None = None) -> None:
        self._extractor = extractor
        self._env = _build_subprocess_env(extractor_lib_dir)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="realtime_pose_features_")
        self._input_csv = Path(self._tmpdir.name) / "pose_rows.csv"
        self._output_csv = Path(self._tmpdir.name) / "pose_features.csv"
        self._frame_index = 0
        self._write_header()

    def _write_header(self) -> None:
        headers = ["video_name", "frame_index", "timestamp", "person_id"]
        for i in range(17):
            headers.extend([f"kpt{i}_x", f"kpt{i}_y", f"kpt{i}_conf"])
        with self._input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)

    def _append_row(self, keypoints: np.ndarray, timestamp: float) -> None:
        row: list[float | int | str] = ["realtime", self._frame_index, float(timestamp), 0]
        for i in range(17):
            row.extend(
                [
                    float(keypoints[i, 0]),
                    float(keypoints[i, 1]),
                    float(keypoints[i, 2]),
                ]
            )
        with self._input_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
        self._frame_index += 1

    def _read_last_feature_row(self) -> dict[str, str]:
        last_row: dict[str, str] | None = None
        with self._output_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                last_row = row
        if last_row is None:
            raise RuntimeError("C++ extractor produced no output rows")
        return last_row

    def _run_extractor(self) -> None:
        result = subprocess.run(
            [str(self._extractor), str(self._input_csv), str(self._output_csv)],
            capture_output=True,
            text=True,
            env=self._env,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or (result.stdout or "").strip()
            raise RuntimeError(f"C++ feature extractor failed: {detail or result.returncode}")

    def reset(self) -> None:
        self._frame_index = 0
        self._write_header()

    def update(self, keypoints: np.ndarray, timestamp: float) -> dict[str, np.ndarray | float | bool]:
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        self._append_row(keypoints, timestamp)
        self._run_extractor()
        row = self._read_last_feature_row()

        top_pairs = []
        for idx in TOP_JOINT_INDICES:
            top_pairs.extend(
                [
                    _parse_csv_float(row.get(f"norm_kpt{idx}_x", "nan")),
                    _parse_csv_float(row.get(f"norm_kpt{idx}_y", "nan")),
                ]
            )
        bot_pairs = []
        for idx in BOTTOM_JOINT_INDICES:
            bot_pairs.extend(
                [
                    _parse_csv_float(row.get(f"norm_kpt{idx}_x", "nan")),
                    _parse_csv_float(row.get(f"norm_kpt{idx}_y", "nan")),
                ]
            )

        feat_values = np.zeros(len(ENGINEERED_FEATURE_COLUMNS), dtype=np.float32)
        for i, name in enumerate(ENGINEERED_FEATURE_COLUMNS):
            val = _parse_csv_float(row.get(name, "nan"))
            if np.isfinite(val):
                feat_values[i] = float(val)

        valid_pose = int(float(row.get("valid_pose", "0") or 0.0)) == 1
        return {
            "top_kp": np.asarray(top_pairs, dtype=np.float32),
            "bot_kp": np.asarray(bot_pairs, dtype=np.float32),
            "feat": feat_values,
            "mask": 1.0 if valid_pose else 0.0,
            "valid_pose": valid_pose,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime fall detection on video or RTSP.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or RTSP URL (e.g. rtsp://user:pass@ip:554/stream).",
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
        help="Temporal window length for fall model.",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Real frames to accumulate before predicting (default: window-size). "
            "Set this to the typical training-clip length for models trained on "
            "short padded clips, e.g. --clip-frames 14 for the 7.5 FPS model "
            "(original clips ~56 frames @ 30 FPS → ~14 frames @ 7.5 FPS). "
            "The buffer will hold N real frames and zero-pad to window-size, "
            "matching the training-time window distribution."
        ),
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run fall model every N processed frames once the window is full.",
    )
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold for classifying fall from BiGRU fall probability "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source video FPS when metadata is missing/wrong (e.g. RTSP).",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=None,
        help=(
            "Frames per second of video to feed the fall model. "
            "Defaults to source FPS (10 fps video -> 10 model frames per second)."
        ),
    )
    parser.add_argument(
        "--feature-backend",
        choices=("cpp", "python", "kalman"),
        default="cpp",
        help="Feature extractor backend (cpp matches training extractor).",
    )
    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=20.0,
        help="Kalman process noise sigma (higher = more responsive). Default 20.0.",
    )
    parser.add_argument(
        "--kalman-meas-noise",
        type=float,
        default=10.0,
        help="Kalman measurement noise sigma (lower = more trust in YOLO). Default 10.0.",
    )
    parser.add_argument(
        "--kalman-miss-tolerance",
        type=int,
        default=2,
        help=(
            "Consecutive missed YOLO frames tolerated as occlusion before treating "
            "the person as absent (resets state, zeroes output). At 7fps: 2 ≈ 0.28s. "
            "Default 2."
        ),
    )
    parser.add_argument(
        "--kalman-kpt-conf",
        type=float,
        default=DEFAULT_MIN_KPT_CONF,
        help="Minimum per-keypoint confidence for valid pose (default: 0.3).",
    )
    parser.add_argument(
        "--kalman-min-valid-keypoints",
        type=int,
        default=DEFAULT_MIN_VALID_KEYPOINTS,
        help=(
            "Minimum keypoints at or above --kalman-kpt-conf to accept a detection "
            "(rejects ghost boxes). Default 3."
        ),
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=DEFAULT_CPP_EXTRACTOR,
        help="Path to C++ extract_pose_features_from_csv binary (used with --feature-backend cpp).",
    )
    parser.add_argument(
        "--extractor-lib-dir",
        type=Path,
        default=None,
        help="Optional library directory to prepend to LD_LIBRARY_PATH for the C++ extractor.",
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
        help=(
            "Write labeled output video (pose + fall/not_fall HUD). "
            "Optional path or directory; bare --export writes to "
            "test/output/<source>_labeled.mp4. Implies --no-show."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def resolve_export_path(
    export: bool | str | Path | None,
    save: Path | None,
    source: str,
) -> Path | None:
    """Resolve --export / legacy --save to a concrete output file path."""
    if export is False:
        return save

    stem = Path(source).stem if not source.isdigit() and "://" not in source else "capture"
    default_path = REPO_ROOT / "test/output" / f"{stem}_labeled.mp4"

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
    """Read FPS from capture metadata, with duration-based fallback for files."""
    if override is not None and override > 0:
        return float(override)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 1.0:
        return fps

    # Fallback: frame_count / duration for local video files.
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

    @property
    def window_seconds(self) -> float:
        return self.model_frame_count / self.process_fps if self.model_frame_count else 0.0

    def should_process(self, video_frame_idx: int) -> tuple[bool, float]:
        """Return whether to add this decoded frame to the model and its timestamp."""
        video_time = video_frame_idx / self.source_fps
        sample_interval = 1.0 / self.process_fps
        frame_mid = video_time + (0.5 / self.source_fps)

        if frame_mid + 1e-9 >= self._next_sample_time:
            timestamp = self._next_sample_time
            self._next_sample_time += sample_interval
            self.model_frame_count += 1
            return True, timestamp
        return False, video_time


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_path = resolve_export_path(args.export, args.save, args.source)
    if export_path is not None:
        args.no_show = True

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
    kalman_interpolator: KalmanPoseInterpolator | None = None
    if args.feature_backend == "cpp":
        extractor_path = args.extractor.resolve()
        if not extractor_path.is_file():
            print(f"Error: extractor binary not found: {extractor_path}", file=sys.stderr)
            print(
                "Build it with: cd pose_features && mkdir -p build && cd build && cmake .. && cmake --build .",
                file=sys.stderr,
            )
            return 1
        feature_extractor = CppPoseFeatureExtractor(
            extractor_path,
            extractor_lib_dir=args.extractor_lib_dir.resolve()
            if args.extractor_lib_dir
            else None,
        )
        print(f"Feature backend: cpp ({extractor_path})")
    elif args.feature_backend == "kalman":
        feature_extractor = None
        kalman_interpolator = KalmanPoseInterpolator(
            target_fps=30.0,
            process_noise=args.kalman_process_noise,
            meas_noise_base=args.kalman_meas_noise,
            miss_tolerance=args.kalman_miss_tolerance,
            min_kpt_conf=args.kalman_kpt_conf,
            min_valid_keypoints=args.kalman_min_valid_keypoints,
        )
        print("Feature backend: kalman (30fps upsample from sparse YOLO)")
    else:
        feature_extractor = PoseFeatureExtractor()
        print("Feature backend: python")
    frame_buffer = FallFrameBuffer(window_size=args.window_size, clip_frames=args.clip_frames)

    cap = open_capture(args.source)
    source_fps = probe_source_fps(cap, args.source, args.fps)
    process_fps = args.process_fps if args.process_fps and args.process_fps > 0 else source_fps
    sampler = VideoFrameSampler(source_fps=source_fps, process_fps=process_fps)
    window_seconds = args.window_size / process_fps
    print(
        f"Source FPS: {source_fps:.2f} | Model sampling: {process_fps:.2f} frames/s "
        f"| {args.window_size}-frame window ~= {window_seconds:.2f}s of video"
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
    model_frame_idx = 0
    video_time = 0.0
    fps_value = 0.0
    last_tick = time.perf_counter()

    window_name = "Fall Detection"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_frame_idx += 1
            video_time = video_frame_idx / source_fps
            should_process, feature_timestamp = sampler.should_process(video_frame_idx)

            t0 = time.perf_counter()
            keypoints = pose_backend.predict(frame)
            pose_ms = (time.perf_counter() - t0) * 1000.0
            skeleton_color = (0, 0, 255) if label == "fall" else (0, 220, 0)

            if keypoints is not None:
                draw_pose(frame, keypoints, skeleton_color)

            if should_process:
                model_frame_idx += 1
                if args.feature_backend == "kalman":
                    assert kalman_interpolator is not None
                    kalman_kpts = keypoints
                    if kalman_kpts is not None and not is_valid_pose_detection(
                        kalman_kpts,
                        min_kpt_conf=args.kalman_kpt_conf,
                        min_valid_keypoints=args.kalman_min_valid_keypoints,
                    ):
                        kalman_kpts = None
                    result = kalman_interpolator.push(kalman_kpts, feature_timestamp)
                    if result.reset_occurred:
                        frame_buffer.reset()
                    for feat in result.frames:
                        frame_buffer.add(feat)
                else:
                    assert feature_extractor is not None
                    if keypoints is not None:
                        frame_features = feature_extractor.update(keypoints, feature_timestamp)
                        frame_buffer.add(frame_features)
                    else:
                        feature_extractor.reset()
                        frame_buffer.reset()

                if frame_buffer.is_ready and model_frame_idx % max(args.infer_every, 1) == 0:
                    t1 = time.perf_counter()
                    prediction = predictor.predict(
                        frame_buffer.as_batch(),
                        valid_ratio=frame_buffer.valid_ratio(),
                    )
                    fall_ms = (time.perf_counter() - t1) * 1000.0
                    fall_prob = prediction["fall_prob"]
                    if prediction.get("ready", False):
                        label = "fall" if fall_prob >= args.fall_threshold else "not_fall"
                        confidence = fall_prob if label == "fall" else (1.0 - fall_prob)
                    else:
                        label = "not_fall"
                        confidence = 0.0
                    skeleton_color = (0, 0, 255) if label == "fall" else (0, 220, 0)
                    if keypoints is not None:
                        draw_pose(frame, keypoints, skeleton_color)

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
                source_fps=process_fps,
                video_time=video_time,
                window_fill=len(frame_buffer),
                window_size=frame_buffer.clip_frames,
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
        f"({sampler.model_frame_count} model frames @ {process_fps:.2f} fps). "
        f"Last label={label} conf={confidence:.3f} fall_prob={fall_prob:.3f}"
    )
    if export_path is not None:
        print(f"Labeled video saved: {export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
