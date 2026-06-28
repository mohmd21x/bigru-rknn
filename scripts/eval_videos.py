#!/usr/bin/env python3
"""Event-level evaluation of a fall model on untrimmed videos.

Unlike clip-level offline evaluation (``scripts/evaluate.py``), this runs the
*full production pipeline* (YOLO pose -> online features -> windowed BiGRU)
over whole videos and scores **events**, which is what production cares about:

- Did each video raise a fall alarm? (event precision / recall)
- How many false alarms on not_fall videos (per video and per hour)?
- Detection latency from video start (or labeled fall onset) to first alarm.

A fall "event" is declared when the windowed fall probability stays at/above
``--fall-threshold`` for ``--consecutive`` model windows (debouncing), matching
the recommended production decision rule.

Labels:
    Provide ``--labels CSV`` with columns ``filename,label[,fall_start]`` or, by
    default, the label is inferred from the filename (``not_fall`` vs ``fall``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import DEFAULT_WINDOW_SIZE
from src.inference.fall_predictor import FallPredictor
from src.inference.frame_buffer import FallFrameBuffer
from src.inference.pose_backend import PoseBackend, add_pose_backend_args, create_pose_backend
from src.inference.pose_features import PoseFeatureExtractor

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}


def infer_label(filename: str) -> str:
    return "not_fall" if "not_fall" in filename.lower() else (
        "fall" if "fall" in filename.lower() else "not_fall"
    )


def load_labels(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("filename") or "").strip()
            if not name:
                continue
            entry: dict = {"label": (row.get("label") or "").strip() or infer_label(name)}
            fs = (row.get("fall_start") or "").strip()
            if fs:
                entry["fall_start"] = float(fs)
            out[name] = entry
    return out


def probe_fps(cap: cv2.VideoCapture, override: float | None) -> float:
    if override and override > 0:
        return float(override)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1.0 else 25.0


class FrameSampler:
    """Sample decoded frames down to ``process_fps`` model frames/sec."""

    def __init__(self, source_fps: float, process_fps: float) -> None:
        self.source_fps = max(source_fps, 1e-3)
        self.process_fps = max(process_fps, 1e-3)
        self._next = 0.0

    def should_process(self, idx: int) -> tuple[bool, float]:
        mid = idx / self.source_fps + 0.5 / self.source_fps
        if mid + 1e-9 >= self._next:
            ts = self._next
            self._next += 1.0 / self.process_fps
            return True, ts
        return False, idx / self.source_fps


def evaluate_video(
    video_path: Path,
    pose_backend: PoseBackend,
    predictor: FallPredictor,
    *,
    window_size: int,
    clip_frames: int | None,
    process_fps: float,
    fall_threshold: float,
    consecutive: int,
) -> dict:
    """Run the pipeline over one video and return its event-level result."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "cannot open"}

    source_fps = probe_fps(cap, None)
    sampler = FrameSampler(source_fps, process_fps)
    feature_extractor = PoseFeatureExtractor()
    buffer = FallFrameBuffer(window_size=window_size, clip_frames=clip_frames)

    consec = 0
    first_alarm_ts: float | None = None
    n_alarms = 0
    max_fall_prob = 0.0
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        should, ts = sampler.should_process(idx)
        if not should:
            continue

        kpts = pose_backend.predict(frame)
        if kpts is None:
            feature_extractor.reset()
            buffer.reset()
            consec = 0
            continue

        buffer.add(feature_extractor.update(kpts, ts))
        if not buffer.is_ready:
            continue

        pred = predictor.predict(buffer.as_batch(), valid_ratio=buffer.valid_ratio())
        fall_prob = float(pred["fall_prob"])
        max_fall_prob = max(max_fall_prob, fall_prob)

        if fall_prob >= fall_threshold:
            consec += 1
            if consec >= consecutive:
                n_alarms += 1
                if first_alarm_ts is None:
                    first_alarm_ts = ts
        else:
            consec = 0

    cap.release()
    duration = (idx + 1) / source_fps if idx >= 0 else 0.0
    return {
        "source_fps": round(source_fps, 2),
        "duration_s": round(duration, 2),
        "predicted_fall": first_alarm_ts is not None,
        "first_alarm_s": None if first_alarm_ts is None else round(first_alarm_ts, 2),
        "num_alarms": n_alarms,
        "max_fall_prob": round(max_fall_prob, 4),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-level fall evaluation on untrimmed videos.")
    p.add_argument("--videos-dir", type=Path, required=True, help="Directory of video files.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/bigru_hierarchical_7fps_v2.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    add_pose_backend_args(p)
    p.add_argument("--labels", type=Path, default=None, help="CSV: filename,label[,fall_start].")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--clip-frames", type=int, default=None)
    p.add_argument("--process-fps", type=float, default=7.5)
    p.add_argument("--fall-threshold", type=float, default=0.5)
    p.add_argument("--consecutive", type=int, default=2, help="Consecutive windows above threshold to alarm.")
    p.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports/video_eval")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.videos_dir.is_dir():
        print(f"Error: videos dir not found: {args.videos_dir}", file=sys.stderr)
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

    print(f"Device: {device} | Pose backend: {args.pose_backend} | checkpoint: {args.checkpoint}")

    predictor = FallPredictor(args.checkpoint, config_path=args.config, device=device)
    labels = load_labels(args.labels)

    videos = sorted(p for p in args.videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"Error: no videos in {args.videos_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    tp = fp = tn = fn = 0
    latencies: list[float] = []
    total_fp_alarms = 0
    total_not_fall_duration = 0.0

    for video in videos:
        meta = labels.get(video.name, {"label": infer_label(video.name)})
        true_label = meta.get("label", infer_label(video.name))
        t0 = time.perf_counter()
        result = evaluate_video(
            video, pose_backend, predictor,
            window_size=args.window_size, clip_frames=args.clip_frames,
            process_fps=args.process_fps,
            fall_threshold=args.fall_threshold, consecutive=args.consecutive,
        )
        result["elapsed_s"] = round(time.perf_counter() - t0, 1)
        result["filename"] = video.name
        result["true_label"] = true_label
        predicted_fall = bool(result.get("predicted_fall"))

        if true_label == "fall":
            if predicted_fall:
                tp += 1
                onset = meta.get("fall_start", 0.0)
                if result.get("first_alarm_s") is not None:
                    latencies.append(max(result["first_alarm_s"] - onset, 0.0))
            else:
                fn += 1
        else:
            total_not_fall_duration += result.get("duration_s", 0.0)
            total_fp_alarms += result.get("num_alarms", 0)
            if predicted_fall:
                fp += 1
            else:
                tn += 1

        rows.append(result)
        print(
            f"  {video.name:45s} true={true_label:8s} pred_fall={predicted_fall} "
            f"alarms={result.get('num_alarms')} max_p={result.get('max_fall_prob')}"
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_per_hour = (total_fp_alarms / total_not_fall_duration * 3600.0) if total_not_fall_duration else 0.0

    summary = {
        "videos": len(videos),
        "event_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "false_alarms_per_hour": round(fp_per_hour, 3),
        "mean_detection_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "settings": {
            "window_size": args.window_size,
            "clip_frames": args.clip_frames,
            "process_fps": args.process_fps,
            "fall_threshold": args.fall_threshold,
            "consecutive": args.consecutive,
            "pose_backend": args.pose_backend,
        },
    }

    out_dir = args.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "video_eval.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_video": rows}, handle, indent=2)
    with (out_dir / "video_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["filename", "true_label", "predicted_fall", "first_alarm_s",
                      "num_alarms", "max_fall_prob", "duration_s", "source_fps", "elapsed_s"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Event-level summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReports written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
