"""Video -> sparse keypoints -> Kalman upsample helpers for inference/visualization."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extract_7fps_from_video import extract_video_keypoints  # noqa: E402
from kalman_upsample import upsample_keypoint_csv  # noqa: E402

from src.inference.pose_backend import PoseBackend  # noqa: E402


@dataclass(frozen=True)
class VideoKeypointArtifacts:
    """Paths produced by the Kalman video preprocessing pipeline."""

    source_video: Path
    work_dir: Path
    low_fps_csv: Path
    high_fps_csv: Path


def default_work_dir(source: Path, work_dir: Path | None) -> Path:
    if work_dir is not None:
        path = work_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    base = REPO_ROOT / "reports" / "kalman_pipeline" / source.stem
    base.mkdir(parents=True, exist_ok=True)
    return base


def prepare_keypoints_from_video(
    source: Path | str,
    *,
    work_dir: Path | None = None,
    extract_fps: float = 7.0,
    upsample_fps: float = 30.0,
    pose_backend: PoseBackend | None = None,
    device: str | None = None,
    conf: float | None = None,
    source_fps: float | None = None,
    jitter_ms: float = 0.0,
    seed: int | None = None,
    process_noise: float | None = None,
    meas_noise_base: float | None = None,
    max_gap_sec: float | None = None,
    conf_decay_sec: float = 0.3,
    mode: str = "causal",
    # Legacy kwargs for callers that still pass yolo_weights / pose-backend args.
    yolo_weights: Path | None = None,
    pose_backend_name: str | None = None,
    rtmo_onnx: Path | None = None,
) -> VideoKeypointArtifacts:
    """Extract ~7fps pose keypoints from video and upsample to 30fps via Kalman."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source video not found: {source_path}")

    out_dir = default_work_dir(source_path, work_dir)
    low_csv = out_dir / f"{source_path.stem}_7fps.csv"
    high_csv = out_dir / f"{source_path.stem}_30fps.csv"

    print(f"[pipeline] Extracting ~{extract_fps:g} FPS keypoints from {source_path.name}")
    extract_video_keypoints(
        source_path,
        low_csv,
        target_fps=extract_fps,
        pose_backend=pose_backend,
        device=device,
        conf=conf,
        fps=source_fps,
        jitter_ms=jitter_ms,
        seed=seed,
        yolo_weights=yolo_weights,
        pose_backend_name=pose_backend_name,
        rtmo_onnx=rtmo_onnx,
    )

    print(f"[pipeline] Upsampling to {upsample_fps:g} FPS via Kalman filter")
    upsample_keypoint_csv(
        low_csv,
        high_csv,
        target_fps=upsample_fps,
        process_noise=process_noise,
        meas_noise_base=meas_noise_base,
        max_gap_sec=max_gap_sec,
        conf_decay_sec=conf_decay_sec,
        mode=mode,
    )

    return VideoKeypointArtifacts(
        source_video=source_path,
        work_dir=out_dir,
        low_fps_csv=low_csv,
        high_fps_csv=high_csv,
    )
