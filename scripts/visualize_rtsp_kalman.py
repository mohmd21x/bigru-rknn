#!/usr/bin/env python3
"""Live 3-window viewer: raw stream, ~7fps pose, 30fps Kalman pose.

Opens three separate OpenCV windows:
  1. Stream — raw video / RTSP
  2. Pose ~7fps — sparse pose samples (held between detections)
  3. Kalman 30fps — causal upsampled pose between sparse samples

Example:
  python scripts/visualize_rtsp_kalman.py --source rtsp://user:pass@host:554/stream
  python scripts/visualize_rtsp_kalman.py --source Videos/danial-fall-slow-cam-3-2-1.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from bisect import bisect_right
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "test"))
from run_realtime import (  # noqa: E402
    VideoFrameSampler,
    open_capture,
    probe_source_fps,
)

from src.inference.kalman_pose_interpolator import (  # noqa: E402
    DEFAULT_MIN_KPT_CONF,
    DEFAULT_MIN_VALID_KEYPOINTS,
    KalmanPoseInterpolator,
    is_valid_pose_detection,
)
from src.inference.pose_backend import add_pose_backend_args, create_pose_backend  # noqa: E402
from src.inference.visualize import draw_pose  # noqa: E402

POSE_COLOR = (0, 220, 0)
POSE_FLASH_COLOR = (0, 255, 255)
KALMAN_COLOR = (0, 200, 255)
ABSENT_COLOR = (0, 0, 255)
TEXT_COLOR = (240, 240, 240)

WINDOW_STREAM = "1 - Stream"
WINDOW_POSE = "2 - Pose ~7fps"
WINDOW_KALMAN = "3 - Kalman 30fps"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-window RTSP/video viewer: stream, sparse pose 7fps, Kalman 30fps.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or RTSP URL.",
    )
    parser.add_argument(
        "--pose-fps",
        type=float,
        default=7.0,
        help="Sparse pose sampling rate (default: 7).",
    )
    add_pose_backend_args(parser)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO backend (default: cuda if available).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source FPS when metadata is missing (RTSP).",
    )
    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=20.0,
        help="Kalman process noise (default: 20).",
    )
    parser.add_argument(
        "--kalman-meas-noise",
        type=float,
        default=10.0,
        help="Kalman measurement noise base (default: 10).",
    )
    parser.add_argument(
        "--kalman-miss-tolerance",
        type=int,
        default=2,
        help="Consecutive missed pose frames before person absent (default: 2).",
    )
    parser.add_argument(
        "--kalman-kpt-conf",
        type=float,
        default=DEFAULT_MIN_KPT_CONF,
        help="Minimum keypoint confidence for valid pose (default: 0.3).",
    )
    parser.add_argument(
        "--kalman-min-valid-keypoints",
        type=int,
        default=DEFAULT_MIN_VALID_KEYPOINTS,
        help="Minimum valid keypoints to accept detection (default: 3).",
    )
    parser.add_argument(
        "--buffer-sec",
        type=float,
        default=10.0,
        help="Seconds of Kalman keypoints to keep for lookup (default: 10).",
    )
    return parser.parse_args(argv)


def draw_banner(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    *,
    absent: bool = False,
) -> None:
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        subtitle,
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    if absent:
        cv2.putText(
            frame,
            "PERSON ABSENT",
            (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            ABSENT_COLOR,
            2,
            cv2.LINE_AA,
        )


def lookup_keypoints_at_time(
    buffer: deque[tuple[float, np.ndarray]],
    t: float,
) -> np.ndarray | None:
    """Return Kalman keypoints closest to time ``t`` (at or before ``t``)."""
    if not buffer:
        return None
    times = [entry[0] for entry in buffer]
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return None
    return buffer[idx][1]


def prune_buffer(buffer: deque[tuple[float, np.ndarray]], min_time: float) -> None:
    while buffer and buffer[0][0] < min_time:
        buffer.popleft()


def filter_sparse_keypoints(
    keypoints: np.ndarray | None,
    *,
    min_kpt_conf: float,
    min_valid_keypoints: int,
) -> np.ndarray | None:
    if keypoints is None:
        return None
    if not is_valid_pose_detection(
        keypoints,
        min_kpt_conf=min_kpt_conf,
        min_valid_keypoints=min_valid_keypoints,
    ):
        return None
    return keypoints


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.pose_fps <= 0:
        print("Error: --pose-fps must be > 0", file=sys.stderr)
        return 1
    if args.kalman_min_valid_keypoints <= 0:
        print("Error: --kalman-min-valid-keypoints must be > 0", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pose_backend = create_pose_backend(args, device=device)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    interpolator = KalmanPoseInterpolator(
        target_fps=30.0,
        process_noise=args.kalman_process_noise,
        meas_noise_base=args.kalman_meas_noise,
        miss_tolerance=args.kalman_miss_tolerance,
        min_kpt_conf=args.kalman_kpt_conf,
        min_valid_keypoints=args.kalman_min_valid_keypoints,
    )

    cap = open_capture(args.source)
    source_fps = probe_source_fps(cap, args.source, args.fps)
    sampler = VideoFrameSampler(source_fps=source_fps, process_fps=args.pose_fps)

    cv2.namedWindow(WINDOW_STREAM, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_POSE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_KALMAN, cv2.WINDOW_NORMAL)

    kalman_buffer: deque[tuple[float, np.ndarray]] = deque()
    last_sparse_kpts: np.ndarray | None = None
    pose_sample_flash = 0
    person_absent = False

    video_frame_idx = -1
    pose_ms = 0.0
    wall_fps = 0.0
    last_tick = time.perf_counter()

    print(f"Source: {args.source}")
    print(f"Device: {device} | Pose backend: {args.pose_backend} | Source FPS: {source_fps:.2f} | Pose: {args.pose_fps:.1f} fps")
    print("Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if "://" in args.source:
                    # RTSP reconnect attempt
                    cap.release()
                    time.sleep(0.5)
                    cap = open_capture(args.source)
                    continue
                break

            video_frame_idx += 1
            video_time = video_frame_idx / source_fps
            should_sample, sample_time = sampler.should_process(video_frame_idx)

            stream_view = frame.copy()

            if should_sample:
                t0 = time.perf_counter()
                raw_kpts = pose_backend.predict(frame)
                pose_ms = (time.perf_counter() - t0) * 1000.0

                sparse_kpts = filter_sparse_keypoints(
                    raw_kpts,
                    min_kpt_conf=args.kalman_kpt_conf,
                    min_valid_keypoints=args.kalman_min_valid_keypoints,
                )
                last_sparse_kpts = sparse_kpts
                pose_sample_flash = 8

                result = interpolator.push(sparse_kpts, sample_time)
                if result.reset_occurred:
                    kalman_buffer.clear()
                    last_sparse_kpts = None
                    person_absent = True
                for t_kp, kp in zip(result.timestamps, result.keypoint_frames):
                    kalman_buffer.append((t_kp, kp.copy()))
                if sparse_kpts is not None:
                    person_absent = False
                elif result.frames:
                    person_absent = not is_valid_pose_detection(
                        result.keypoint_frames[-1],
                        min_kpt_conf=args.kalman_kpt_conf,
                        min_valid_keypoints=args.kalman_min_valid_keypoints,
                    )

            prune_buffer(kalman_buffer, video_time - args.buffer_sec)

            pose_view = frame.copy()
            kalman_view = frame.copy()

            if last_sparse_kpts is not None:
                color = POSE_FLASH_COLOR if pose_sample_flash > 0 else POSE_COLOR
                draw_pose(pose_view, last_sparse_kpts, color, conf_threshold=args.kalman_kpt_conf)
                draw_banner(
                    pose_view,
                    WINDOW_POSE,
                    f"Pose @ {args.pose_fps:.1f} fps | pose={pose_ms:.0f}ms",
                )
            else:
                draw_banner(
                    pose_view,
                    WINDOW_POSE,
                    f"t={video_time:.2f}s | Pose @ {args.pose_fps:.1f} fps",
                    absent=True,
                )

            kalman_kpts = lookup_keypoints_at_time(kalman_buffer, video_time)
            kalman_absent = (
                kalman_kpts is None
                or not is_valid_pose_detection(
                    kalman_kpts,
                    min_kpt_conf=args.kalman_kpt_conf,
                    min_valid_keypoints=args.kalman_min_valid_keypoints,
                )
            )
            if kalman_absent or person_absent:
                draw_banner(
                    kalman_view,
                    WINDOW_KALMAN,
                    f"t={video_time:.2f}s | upsampled 30fps",
                    absent=True,
                )
            else:
                draw_pose(
                    kalman_view,
                    kalman_kpts,
                    KALMAN_COLOR,
                    conf_threshold=args.kalman_kpt_conf,
                )
                draw_banner(
                    kalman_view,
                    WINDOW_KALMAN,
                    f"t={video_time:.2f}s | upsampled 30fps",
                )

            draw_banner(stream_view, WINDOW_STREAM, f"t={video_time:.2f}s | {source_fps:.1f} fps")

            now = time.perf_counter()
            wall_fps = 0.9 * wall_fps + 0.1 * (1.0 / max(now - last_tick, 1e-6))
            last_tick = now
            cv2.putText(
                stream_view,
                f"wall_fps={wall_fps:.1f}",
                (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_STREAM, stream_view)
            cv2.imshow(WINDOW_POSE, pose_view)
            cv2.imshow(WINDOW_KALMAN, kalman_view)

            if pose_sample_flash > 0:
                pose_sample_flash -= 1

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
