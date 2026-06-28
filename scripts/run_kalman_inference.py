#!/usr/bin/env python3
"""Offline fall inference on upsampled 30fps keypoint CSV or input video.

Pipeline (video mode):
  video  ->  ~7fps YOLO pose  ->  Kalman upsample  ->  BiGRU inference

Pipeline (CSV mode):
  30fps CSV  ->  PoseFeatureExtractor.update(kp, t)
            ->  sliding windows (window=64, stride=32)
            ->  FallPredictor.predict()
            ->  per-window JSON + clip-level summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import CLASS_NAMES, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STRIDE
from src.data.windowing import slide_windows, valid_frame_ratio
from src.inference.fall_predictor import FallPredictor
from src.inference.pose_backend import add_pose_backend_args
from src.inference.pose_features import PoseFeatureExtractor

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from kalman_video_pipeline import prepare_keypoints_from_video  # noqa: E402

NUM_KEYPOINTS = 17
CSV_HEADER = ["video_name", "frame_index", "timestamp", "person_id"]
for _i in range(NUM_KEYPOINTS):
    CSV_HEADER.extend([f"kpt{_i}_x", f"kpt{_i}_y", f"kpt{_i}_conf"])


def _parse_float(value: str) -> float:
    text = (value or "").strip().lower()
    if not text or text == "nan":
        return float("nan")
    return float(text)


def row_to_keypoints(row: dict[str, str]) -> np.ndarray:
    """Return COCO-17 keypoints as ``(17, 3)`` float32 array."""
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
        return list(reader)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["video_name"], int(float(row["person_id"])))
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: float(row["timestamp"]))
    return groups


def extract_frame_features(
    rows: list[dict[str, str]],
    extractor: PoseFeatureExtractor,
) -> tuple[list[dict], list[float]]:
    """Run PoseFeatureExtractor over sorted keypoint rows."""
    frames: list[dict] = []
    timestamps: list[float] = []
    for row in rows:
        timestamp = float(row["timestamp"])
        keypoints = row_to_keypoints(row)
        frames.append(extractor.update(keypoints, timestamp))
        timestamps.append(timestamp)
    return frames, timestamps


def window_to_batch(window) -> dict[str, np.ndarray]:
    return {
        "top_kp": window.top_kp.reshape(1, window.top_kp.shape[0], -1),
        "bot_kp": window.bot_kp.reshape(1, window.bot_kp.shape[0], -1),
        "feat": window.feat.reshape(1, window.feat.shape[0], -1),
        "mask": window.mask.reshape(1, window.mask.shape[0]),
    }


def infer_track(
    track_id: str,
    rows: list[dict[str, str]],
    predictor: FallPredictor,
    *,
    window_size: int,
    stride: int,
) -> tuple[list[dict], dict]:
    """Run inference on one (video_name, person_id) track."""
    extractor = PoseFeatureExtractor()
    frames, timestamps = extract_frame_features(rows, extractor)

    top_kp = np.stack([frame["top_kp"] for frame in frames], axis=0)
    bot_kp = np.stack([frame["bot_kp"] for frame in frames], axis=0)
    feat = np.stack([frame["feat"] for frame in frames], axis=0)
    mask = np.array([frame["mask"] for frame in frames], dtype=np.float32)

    predictions: list[dict] = []
    max_fall_prob = 0.0
    best_window: dict | None = None

    for window in slide_windows(top_kp, bot_kp, feat, mask, window_size, stride):
        batch = window_to_batch(window)
        ratio = valid_frame_ratio(window.mask)
        pred = predictor.predict(batch, valid_ratio=ratio)
        fall_prob = float(pred["fall_prob"])
        confidence = float(pred["confidence"])
        ts_start = timestamps[window.start]
        ts_end = timestamps[min(window.end - 1, len(timestamps) - 1)]

        entry = {
            "track_id": track_id,
            "frame_start": window.start,
            "frame_end": window.end,
            "timestamp_start": round(ts_start, 4),
            "timestamp_end": round(ts_end, 4),
            "label": pred["label"],
            "fall_prob": round(fall_prob, 4),
            "confidence": round(confidence, 4),
            "valid_ratio": round(ratio, 4),
            "ready": bool(pred["ready"]),
        }
        predictions.append(entry)

    if predictions:
        fall_probs = np.array([entry["fall_prob"] for entry in predictions], dtype=np.float64)
        confidences = np.array([entry["confidence"] for entry in predictions], dtype=np.float64)
        best_idx = int(np.lexsort((-confidences, -fall_probs))[0])
        best_window = predictions[best_idx]
        max_fall_prob = float(best_window["fall_prob"])

    clip_label = "not_fall"
    clip_confidence = 0.0
    if best_window is not None and max_fall_prob >= 0.5:
        clip_label = "fall"
        clip_confidence = best_window["confidence"]
    elif best_window is not None:
        clip_confidence = best_window["confidence"]

    summary = {
        "track_id": track_id,
        "num_frames": len(frames),
        "num_windows": len(predictions),
        "duration_s": round(timestamps[-1] - timestamps[0], 4) if timestamps else 0.0,
        "clip_label": clip_label,
        "max_fall_prob": round(max_fall_prob, 4),
        "clip_confidence": round(clip_confidence, 4),
    }
    return predictions, summary


def load_config_values(config_path: Path) -> tuple[int, int, float]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    window_size = int(data_cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    stride = int(data_cfg.get("stride", DEFAULT_WINDOW_STRIDE))
    min_valid = float(data_cfg.get("min_valid_frame_ratio", 0.5))
    return window_size, stride, min_valid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline fall inference on upsampled 30fps keypoint CSV or input video.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--source",
        type=Path,
        help="Input video file (runs YOLO ~7fps extract + Kalman upsample).",
    )
    input_group.add_argument(
        "--keypoints",
        type=Path,
        help="Input 30fps keypoint CSV (Kalman upsampled).",
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
        help="Sparse pose sampling rate for --source (default: 7).",
    )
    parser.add_argument(
        "--upsample-fps",
        type=float,
        default=30.0,
        help="Kalman upsample target rate for --source (default: 30).",
    )
    add_pose_backend_args(parser)
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
        "--process-noise",
        type=float,
        default=5.0,
        help="Kalman process noise for --source (default: 5.0).",
    )
    parser.add_argument(
        "--meas-noise-base",
        type=float,
        default=20.0,
        help="Kalman measurement noise base for --source (default: 20.0).",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=1.0,
        help="Bilateral absence window for --source Kalman upsample (default: 1.0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bigru_hierarchical.yaml",
        help="Training config YAML (window_size, stride).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/bigru_hierarchical/best.pt",
        help="Fall classifier checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write predictions JSON (default: stdout only).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1

    source_video: Path | None = None
    if args.source is not None:
        source_video = args.source.resolve()
        if not source_video.is_file():
            print(f"Error: source video not found: {source_video}", file=sys.stderr)
            return 1
        try:
            artifacts = prepare_keypoints_from_video(
                source_video,
                work_dir=args.work_dir,
                extract_fps=args.extract_fps,
                upsample_fps=args.upsample_fps,
                device=args.device,
                conf=args.conf,
                source_fps=args.source_fps,
                jitter_ms=args.jitter_ms,
                process_noise=args.process_noise,
                meas_noise_base=args.meas_noise_base,
                max_gap_sec=args.max_gap_sec,
                pose_backend_name=args.pose_backend,
                yolo_weights=args.yolo_weights,
                rtmo_onnx=args.rtmo_onnx,
            )
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        keypoints_path = artifacts.high_fps_csv
        print(f"[pipeline] Using upsampled keypoints: {keypoints_path}")
    else:
        keypoints_path = args.keypoints.resolve()
        if not keypoints_path.is_file():
            print(f"Error: keypoints CSV not found: {keypoints_path}", file=sys.stderr)
            return 1

    try:
        rows = read_keypoint_csv(keypoints_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"Error: keypoints CSV is empty: {keypoints_path}", file=sys.stderr)
        return 1

    window_size, stride, _ = load_config_values(args.config.resolve())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = FallPredictor(args.checkpoint, config_path=args.config, device=device)

    groups = group_rows(rows)
    all_predictions: list[dict] = []
    summaries: list[dict] = []

    for (video_name, person_id), track_rows in sorted(groups.items()):
        track_id = f"{video_name}#person{person_id}"
        preds, summary = infer_track(
            track_id,
            track_rows,
            predictor,
            window_size=window_size,
            stride=stride,
        )
        all_predictions.extend(preds)
        summaries.append(summary)

        for pred in preds:
            print(
                f"  [{track_id}] t={pred['timestamp_start']:.3f}-{pred['timestamp_end']:.3f}s "
                f"label={pred['label']} fall_prob={pred['fall_prob']:.3f} "
                f"conf={pred['confidence']:.3f}"
            )

    overall_fall_prob = max((s["max_fall_prob"] for s in summaries), default=0.0)
    overall_label = "fall" if overall_fall_prob >= 0.5 else "not_fall"
    result = {
        "source_video": str(source_video) if source_video is not None else None,
        "source_csv": str(keypoints_path),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "window_size": window_size,
        "stride": stride,
        "class_names": list(CLASS_NAMES),
        "tracks": summaries,
        "windows": all_predictions,
        "summary": {
            "num_tracks": len(summaries),
            "num_windows": len(all_predictions),
            "clip_label": overall_label,
            "max_fall_prob": round(overall_fall_prob, 4),
        },
    }

    print(
        f"\nSummary: {overall_label} (max_fall_prob={overall_fall_prob:.3f}) "
        f"| {len(all_predictions)} windows across {len(summaries)} track(s)"
    )

    if args.output_json is not None:
        out_path = args.output_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
