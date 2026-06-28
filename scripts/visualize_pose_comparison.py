#!/usr/bin/env python3
"""Side-by-side visualization of sparse vs Kalman-upsampled pose keypoints.

Accepts either:
  - ``--source`` video (runs YOLO ~7fps extract + Kalman upsample), or
  - ``--low-fps`` + ``--high-fps`` keypoint CSVs

Produces:
  - OpenCV MP4 with 7fps (left) vs 30fps (right) skeleton panels at playback FPS
  - Matplotlib PNG of hip-center Y trajectory over time
"""

from __future__ import annotations

import argparse
import csv
import sys
from bisect import bisect_right
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.visualize import draw_pose

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from kalman_video_pipeline import prepare_keypoints_from_video  # noqa: E402

NUM_KEYPOINTS = 17
CSV_HEADER = ["video_name", "frame_index", "timestamp", "person_id"]
for _i in range(NUM_KEYPOINTS):
    CSV_HEADER.extend([f"kpt{_i}_x", f"kpt{_i}_y", f"kpt{_i}_conf"])

DIM_COLOR = (0, 120, 0)
BRIGHT_COLOR = (0, 220, 0)
HIGH_FPS_COLOR = (0, 200, 255)
PANEL_BG = (24, 24, 24)
TEXT_COLOR = (240, 240, 240)
FLASH_COLOR = (0, 255, 255)
ABSENT_COLOR = (0, 0, 255)
MIN_CONF = 1e-3


def _parse_float(value: str) -> float:
    text = (value or "").strip().lower()
    if not text or text == "nan":
        return float("nan")
    return float(text)


def row_to_keypoints(row: dict[str, str]) -> np.ndarray:
    kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for i in range(NUM_KEYPOINTS):
        kpts[i, 0] = _parse_float(row.get(f"kpt{i}_x", "nan"))
        kpts[i, 1] = _parse_float(row.get(f"kpt{i}_y", "nan"))
        kpts[i, 2] = _parse_float(row.get(f"kpt{i}_conf", "nan"))
        if not np.isfinite(kpts[i, 0]):
            kpts[i, 0] = 0.0
        if not np.isfinite(kpts[i, 1]):
            kpts[i, 1] = 0.0
        if not np.isfinite(kpts[i, 2]):
            kpts[i, 2] = 0.0
    return kpts


def read_keypoint_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [col for col in CSV_HEADER if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        rows = list(reader)
    rows.sort(key=lambda row: float(row["timestamp"]))
    return rows


def primary_track_key(rows: list[dict[str, str]]) -> tuple[str, int] | None:
    """Return the (video_name, person_id) key with the most rows."""
    if not rows:
        return None
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["video_name"], int(float(row["person_id"])))
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def filter_track(rows: list[dict[str, str]], key: tuple[str, int]) -> list[dict[str, str]]:
    video_name, person_id = key
    filtered = [
        row
        for row in rows
        if row["video_name"] == video_name and int(float(row["person_id"])) == person_id
    ]
    filtered.sort(key=lambda row: float(row["timestamp"]))
    return filtered


def lookup_high_fps_index(timestamps: list[float], t: float) -> int:
    """Return the nearest high-fps row index for playback time ``t``."""
    if not timestamps:
        return 0
    idx = bisect_right(timestamps, t) - 1
    if idx < 0:
        return 0
    if idx + 1 < len(timestamps):
        prev_delta = abs(timestamps[idx] - t)
        next_delta = abs(timestamps[idx + 1] - t)
        if next_delta < prev_delta:
            return idx + 1
    return min(idx, len(timestamps) - 1)


def is_person_absent(keypoints: np.ndarray) -> bool:
    """Return True when every keypoint has zero (or near-zero) confidence."""
    return bool(np.all(keypoints[:, 2] <= MIN_CONF))


def collect_absent_regions(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    """Return contiguous [start, end] time intervals with no detected person."""
    if not rows:
        return []

    regions: list[tuple[float, float]] = []
    in_absent = False
    region_start = 0.0

    for row in rows:
        t = float(row["timestamp"])
        absent = is_person_absent(row_to_keypoints(row))
        if absent and not in_absent:
            region_start = t
            in_absent = True
        elif not absent and in_absent:
            regions.append((region_start, t))
            in_absent = False

    if in_absent:
        regions.append((region_start, float(rows[-1]["timestamp"])))

    return regions


def draw_absent_overlay(panel: np.ndarray) -> None:
    text = "PERSON ABSENT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (panel.shape[1] - text_w) // 2
    y = (panel.shape[0] + text_h) // 2
    cv2.putText(panel, text, (x, y), font, scale, ABSENT_COLOR, thickness, cv2.LINE_AA)


def hip_center_y(keypoints: np.ndarray, conf_min: float = 0.3) -> float | None:
    left = keypoints[11]
    right = keypoints[12]
    left_ok = left[2] >= conf_min
    right_ok = right[2] >= conf_min
    if not left_ok and not right_ok:
        return None
    if left_ok and right_ok:
        return float((left[1] + right[1]) * 0.5)
    return float(left[1] if left_ok else right[1])


def collect_bounds(rows_list: list[list[dict[str, str]]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for rows in rows_list:
        for row in rows:
            kpts = row_to_keypoints(row)
            valid = kpts[kpts[:, 2] >= 0.3]
            if len(valid):
                xs.extend(valid[:, 0].tolist())
                ys.extend(valid[:, 1].tolist())
    if not xs:
        return 0.0, 640.0, 0.0, 480.0
    margin = 40.0
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def normalize_keypoints(
    keypoints: np.ndarray,
    bounds: tuple[float, float, float, float],
    panel_w: int,
    panel_h: int,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)
    scale = min((panel_w - 40) / span_x, (panel_h - 60) / span_y)

    out = keypoints.copy()
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    for i in range(len(out)):
        out[i, 0] = (out[i, 0] - cx) * scale + panel_w * 0.5
        out[i, 1] = (out[i, 1] - cy) * scale + panel_h * 0.5
    return out


def find_sample_at_time(
    rows: list[dict[str, str]],
    timestamps: list[float],
    t: float,
    tolerance: float,
) -> tuple[np.ndarray | None, bool]:
    """Return keypoints at ``t`` if a sample exists within tolerance."""
    if not rows:
        return None, False
    idx = bisect_right(timestamps, t) - 1
    if idx < 0:
        return None, False
    if abs(timestamps[idx] - t) <= tolerance:
        return row_to_keypoints(rows[idx]), True
    if idx + 1 < len(timestamps) and abs(timestamps[idx + 1] - t) <= tolerance:
        return row_to_keypoints(rows[idx + 1]), True
    return row_to_keypoints(rows[idx]), False


def draw_panel_header(
    panel: np.ndarray,
    title: str,
    timestamp: float,
    *,
    flash: bool = False,
) -> None:
    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"t={timestamp:.3f}s",
        (12, panel.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    if flash:
        cv2.putText(
            panel,
            "NEW SAMPLE",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            FLASH_COLOR,
            2,
            cv2.LINE_AA,
        )


def save_trajectory_plot(
    low_rows: list[dict[str, str]],
    high_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    low_t: list[float] = []
    low_y: list[float] = []
    for row in low_rows:
        y = hip_center_y(row_to_keypoints(row))
        if y is not None:
            low_t.append(float(row["timestamp"]))
            low_y.append(y)

    high_t: list[float] = []
    high_y: list[float] = []
    for row in high_rows:
        y = hip_center_y(row_to_keypoints(row))
        if y is not None:
            high_t.append(float(row["timestamp"]))
            high_y.append(y)

    fig, ax = plt.subplots(figsize=(10, 4))
    for start, end in collect_absent_regions(high_rows):
        ax.axvspan(start, end, color="#888888", alpha=0.25, zorder=0)
    if high_t:
        ax.plot(high_t, high_y, color="#00c8ff", linewidth=1.5, label="30 FPS (Kalman)")
    if low_t:
        ax.scatter(low_t, low_y, color="#00dc00", s=28, zorder=3, label="7 FPS (raw)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip center Y (pixels)")
    ax.set_title("Hip-center vertical position: sparse vs upsampled")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_comparison_video(
    low_rows: list[dict[str, str]],
    high_rows: list[dict[str, str]],
    output_path: Path,
    *,
    playback_fps: float,
    frame_width: int,
    panel_height: int,
) -> int:
    if not high_rows:
        raise ValueError("High-fps CSV has no rows")

    bounds = collect_bounds([low_rows, high_rows])
    low_timestamps = [float(row["timestamp"]) for row in low_rows]
    high_timestamps = [float(row["timestamp"]) for row in high_rows]

    t_start = high_timestamps[0]
    t_end = high_timestamps[-1]
    dt = 1.0 / playback_fps
    sample_tolerance = max(dt * 0.5, 1.0 / 120.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        playback_fps,
        (frame_width * 2, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    frame_count = 0
    t = t_start
    while t <= t_end + dt * 0.5:
        left_panel = np.full((panel_height, frame_width, 3), PANEL_BG, dtype=np.uint8)
        right_panel = np.full((panel_height, frame_width, 3), PANEL_BG, dtype=np.uint8)

        low_kpts, is_sample = find_sample_at_time(
            low_rows, low_timestamps, t, sample_tolerance
        )
        if low_kpts is not None:
            low_norm = normalize_keypoints(low_kpts, bounds, frame_width, panel_height)
            draw_pose(left_panel, low_norm, BRIGHT_COLOR if is_sample else DIM_COLOR)

        high_idx = lookup_high_fps_index(high_timestamps, t)
        high_kpts_raw = row_to_keypoints(high_rows[high_idx])
        if is_person_absent(high_kpts_raw):
            draw_absent_overlay(right_panel)
        else:
            high_kpts = normalize_keypoints(
                high_kpts_raw,
                bounds,
                frame_width,
                panel_height,
            )
            draw_pose(right_panel, high_kpts, HIGH_FPS_COLOR)

        draw_panel_header(
            left_panel,
            "7 FPS - raw samples",
            t,
            flash=is_sample,
        )
        draw_panel_header(right_panel, "30 FPS - Kalman", t)

        combined = np.hstack([left_panel, right_panel])
        writer.write(combined)
        frame_count += 1
        t += dt

    writer.release()
    return frame_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sparse vs Kalman-upsampled pose keypoints.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Input video file (runs YOLO ~7fps extract + Kalman upsample).",
    )
    parser.add_argument(
        "--low-fps",
        type=Path,
        default=None,
        help="Input sparse keypoint CSV (e.g. ~7 FPS).",
    )
    parser.add_argument(
        "--high-fps",
        type=Path,
        default=None,
        help="Input upsampled keypoint CSV (e.g. 30 FPS).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate keypoint CSVs when using --source "
        "(default: reports/kalman_pipeline/<video_stem>/).",
    )
    parser.add_argument(
        "--extract-fps",
        type=float,
        default=7.0,
        help="YOLO pose sampling rate for --source (default: 7).",
    )
    parser.add_argument(
        "--upsample-fps",
        type=float,
        default=30.0,
        help="Kalman upsample target rate for --source (default: 30).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=REPO_ROOT / "weights/yolo8n.pt",
        help="YOLOv8 pose weights for --source.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for --source YOLO (default: cuda if available).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO person confidence threshold for --source.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override source video FPS metadata for --source.",
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Timestamp jitter in ms for --source extraction (default: 0).",
    )
    parser.add_argument(
        "--mode",
        choices=("causal", "rts"),
        default="causal",
        help="Kalman upsample mode for --source: causal (forward-only, production-safe) "
        "or rts (offline smoother with future leakage). Default: causal.",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=None,
        help="Kalman process noise for --source (default: 20 causal, 5 rts).",
    )
    parser.add_argument(
        "--meas-noise-base",
        type=float,
        default=None,
        help="Kalman measurement noise base for --source (default: 10 causal, 20 rts).",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=None,
        help="Absence gap threshold for --source Kalman upsample (default: 0.5 causal, 1.0 rts).",
    )
    parser.add_argument(
        "--conf-decay-sec",
        type=float,
        default=0.3,
        help="Confidence decay time constant for causal mode (default: 0.3).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Output side-by-side comparison MP4 (required unless --source only).",
    )
    parser.add_argument(
        "--output-trajectory",
        type=Path,
        default=None,
        help="Output hip-center trajectory PNG (default: <video_stem>_trajectory.png).",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=30.0,
        help="Output video frame rate (default: 30).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Width of each panel in pixels (default: 640).",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=480,
        help="Height of each panel in pixels (default: 480).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    has_source = args.source is not None
    has_low = args.low_fps is not None
    has_high = args.high_fps is not None

    if has_source and (has_low or has_high):
        print("Error: use either --source or --low-fps/--high-fps, not both", file=sys.stderr)
        return 1
    if not has_source and not (has_low and has_high):
        print(
            "Error: provide --source video or both --low-fps and --high-fps CSVs",
            file=sys.stderr,
        )
        return 1
    if has_source and not has_low and not has_high:
        source_path = args.source.resolve()
        if not source_path.is_file():
            print(f"Error: source video not found: {source_path}", file=sys.stderr)
            return 1
        if args.output_video is None:
            args.output_video = REPO_ROOT / "reports" / "kalman_pipeline" / f"{source_path.stem}_comparison.mp4"
        try:
            artifacts = prepare_keypoints_from_video(
                source_path,
                work_dir=args.work_dir,
                extract_fps=args.extract_fps,
                upsample_fps=args.upsample_fps,
                yolo_weights=args.yolo_weights,
                device=args.device,
                conf=args.conf,
                source_fps=args.source_fps,
                jitter_ms=args.jitter_ms,
                process_noise=args.process_noise,
                meas_noise_base=args.meas_noise_base,
                max_gap_sec=args.max_gap_sec,
                conf_decay_sec=args.conf_decay_sec,
                mode=args.mode,
            )
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        low_path = artifacts.low_fps_csv
        high_path = artifacts.high_fps_csv
    else:
        if args.output_video is None:
            print("Error: --output-video is required when using CSV inputs", file=sys.stderr)
            return 1
        low_path = args.low_fps.resolve()
        high_path = args.high_fps.resolve()
        for label, path in (("low-fps", low_path), ("high-fps", high_path)):
            if not path.is_file():
                print(f"Error: {label} CSV not found: {path}", file=sys.stderr)
                return 1

    try:
        low_rows = read_keypoint_csv(low_path)
        high_rows = read_keypoint_csv(high_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not high_rows:
        print(f"Error: high-fps CSV has no data rows: {high_path}", file=sys.stderr)
        return 1

    track_key = primary_track_key(high_rows) or primary_track_key(low_rows)
    if track_key is not None:
        low_rows = filter_track(low_rows, track_key)
        high_rows = filter_track(high_rows, track_key)

    if args.playback_fps <= 0:
        print("Error: --playback-fps must be > 0", file=sys.stderr)
        return 1
    if args.frame_width <= 0 or args.panel_height <= 0:
        print("Error: --frame-width and --panel-height must be > 0", file=sys.stderr)
        return 1

    traj_path = args.output_trajectory
    if traj_path is None:
        traj_path = args.output_video.with_name(f"{args.output_video.stem}_trajectory.png")

    try:
        frame_count = render_comparison_video(
            low_rows,
            high_rows,
            args.output_video.resolve(),
            playback_fps=args.playback_fps,
            frame_width=args.frame_width,
            panel_height=args.panel_height,
        )
        save_trajectory_plot(low_rows, high_rows, traj_path.resolve())
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Video: {args.output_video.resolve()} ({frame_count} frames @ {args.playback_fps:.1f} fps)")
    print(f"Trajectory: {traj_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
