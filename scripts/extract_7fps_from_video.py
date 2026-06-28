#!/usr/bin/env python3
"""Extract sparse (~7 FPS) pose keypoints from a video via a pose backend.

Samples decoded frames at approximately ``--target-fps`` while recording
wall-clock timestamps from the source video (``video_frame_index / source_fps``),
not a uniform 1/target-fps grid. Optional ``--jitter-ms`` adds timestamp noise to
model irregular real-world sampling.

Output CSV matches ``dataset/outputs/*.csv`` (COCO-17, 55 columns).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

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
    is_valid_pose_detection,
)
from src.inference.pose_backend import (  # noqa: E402
    PoseBackend,
    add_pose_backend_args,
    create_pose_backend,
)
from src.inference.rtmo_pose import DEFAULT_ONNX  # noqa: E402

CSV_HEADER = ["video_name", "frame_index", "timestamp", "person_id"]
for _i in range(17):
    CSV_HEADER.extend([f"kpt{_i}_x", f"kpt{_i}_y", f"kpt{_i}_conf"])


def keypoints_to_row(
    video_name: str,
    frame_index: int,
    timestamp: float,
    keypoints: np.ndarray | None,
) -> list[float | int | str]:
    """Build one CSV row from COCO-17 keypoints (or zeros when missing)."""
    row: list[float | int | str] = [video_name, frame_index, float(timestamp), 0]
    if keypoints is None:
        for _ in range(17):
            row.extend([0.0, 0.0, 0.0])
        return row

    kpts = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
    for i in range(17):
        row.extend([float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])])
    return row


def apply_timestamp_jitter(timestamp: float, jitter_ms: float, rng: random.Random) -> float:
    if jitter_ms <= 0:
        return timestamp
    return timestamp + rng.uniform(-jitter_ms, jitter_ms) / 1000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sparse pose keypoints from video at ~target FPS.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input video file path.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=7.0,
        help="Approximate sampling rate for pose extraction (default: 7).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output keypoint CSV path.",
    )
    add_pose_backend_args(parser)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for YOLO backend (default: cuda if available).",
    )
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=DEFAULT_MIN_KPT_CONF,
        help="Minimum per-keypoint confidence for a valid pose (default: 0.3).",
    )
    parser.add_argument(
        "--min-valid-keypoints",
        type=int,
        default=DEFAULT_MIN_VALID_KEYPOINTS,
        help=(
            "Minimum keypoints at or above --kpt-conf to keep a detection; "
            "otherwise write an absent row (default: 3)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source video FPS when metadata is missing or wrong.",
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Uniform random timestamp jitter in milliseconds (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --jitter-ms (default: nondeterministic).",
    )
    return parser.parse_args(argv)


def extract_video_keypoints(
    source: Path | str,
    output_csv: Path,
    *,
    target_fps: float = 7.0,
    pose_backend: PoseBackend | None = None,
    device: str | None = None,
    conf: float | None = None,
    kpt_conf: float = DEFAULT_MIN_KPT_CONF,
    min_valid_keypoints: int = DEFAULT_MIN_VALID_KEYPOINTS,
    fps: float | None = None,
    jitter_ms: float = 0.0,
    seed: int | None = None,
    # Legacy kwargs for callers that still pass yolo_weights / pose-backend args.
    yolo_weights: Path | None = None,
    pose_backend_name: str | None = None,
    rtmo_onnx: Path | None = None,
) -> Path:
    """Extract sparse pose keypoints from a video and write a keypoint CSV."""
    if target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if min_valid_keypoints <= 0:
        raise ValueError("--min-valid-keypoints must be > 0")

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"source video not found: {source_path}")

    if pose_backend is None:
        backend_args = argparse.Namespace(
            pose_backend=pose_backend_name or "yolo",
            yolo_weights=yolo_weights or (REPO_ROOT / "weights/yolo8n.pt"),
            rtmo_onnx=rtmo_onnx or DEFAULT_ONNX,
            conf=conf,
        )
        device_resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pose_backend = create_pose_backend(backend_args, device=device_resolved)

    video_name = source_path.stem
    rng = random.Random(seed)

    cap = open_capture(str(source_path))
    source_fps = probe_source_fps(cap, str(source_path), fps)
    sampler = VideoFrameSampler(source_fps=source_fps, process_fps=target_fps)

    output_path = output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_frame_idx = -1
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_frame_idx += 1
            should_sample, _ = sampler.should_process(video_frame_idx)
            if not should_sample:
                continue

            keypoints = pose_backend.predict(frame)
            if keypoints is not None and not is_valid_pose_detection(
                keypoints,
                min_kpt_conf=kpt_conf,
                min_valid_keypoints=min_valid_keypoints,
            ):
                keypoints = None
            timestamp = apply_timestamp_jitter(
                video_frame_idx / source_fps,
                jitter_ms,
                rng,
            )
            writer.writerow(
                keypoints_to_row(video_name, video_frame_idx, timestamp, keypoints)
            )

    cap.release()
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.target_fps <= 0:
        print("Error: --target-fps must be > 0", file=sys.stderr)
        return 1
    if args.min_valid_keypoints <= 0:
        print("Error: --min-valid-keypoints must be > 0", file=sys.stderr)
        return 1

    source_path = Path(args.source)
    if not source_path.is_file():
        print(f"Error: source video not found: {source_path}", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_csv = args.output_csv.resolve()

    try:
        pose_backend = create_pose_backend(args, device=device)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Source: {source_path}")
    print(f"Device: {device}")
    print(f"Pose backend: {args.pose_backend}")
    print(f"Target sampling: {args.target_fps:.3f} fps")
    if args.jitter_ms > 0:
        print(f"Timestamp jitter: +/- {args.jitter_ms:.1f} ms")

    try:
        extract_video_keypoints(
            source_path,
            output_csv,
            target_fps=args.target_fps,
            pose_backend=pose_backend,
            kpt_conf=args.kpt_conf,
            min_valid_keypoints=args.min_valid_keypoints,
            fps=args.fps,
            jitter_ms=args.jitter_ms,
            seed=args.seed,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sample_count = sum(1 for _ in output_csv.open()) - 1
    print(f"Wrote {sample_count} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
