"""Drawing helpers for realtime pose + fall overlay."""

from __future__ import annotations

import cv2
import numpy as np

# COCO-17 skeleton pairs for YOLO pose.
SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

FALL_COLOR = (0, 0, 255)
SAFE_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
PANEL_BG = (20, 20, 20)


def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    conf_threshold: float = 0.3,
) -> None:
    """Draw COCO skeleton and joints on ``frame`` in place."""
    if keypoints is None or len(keypoints) == 0:
        return

    for i, j in SKELETON:
        if keypoints[i, 2] < conf_threshold or keypoints[j, 2] < conf_threshold:
            continue
        p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    for kp in keypoints:
        if kp[2] < conf_threshold:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    *,
    label: str,
    confidence: float,
    fall_prob: float,
    pose_ms: float,
    fall_ms: float,
    fps: float,
    source_fps: float,
    video_time: float,
    window_fill: int,
    window_size: int,
    source: str,
) -> None:
    """Draw status panel and timing metrics."""
    is_fall = label == "fall"
    accent = FALL_COLOR if is_fall else SAFE_COLOR
    h, w = frame.shape[:2]
    window_seconds = window_size / max(source_fps, 1e-3)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 460), 218), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    status = "FALL" if is_fall else "NOT FALL"
    cv2.putText(frame, status, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, accent, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"conf={confidence:.2f}  fall_prob={fall_prob:.2f}",
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"pose={pose_ms:.1f}ms  fall={fall_ms:.1f}ms  wall_fps={fps:.1f}",
        (24, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"video_fps={source_fps:.2f}  t={video_time:.2f}s  win={window_fill}/{window_size} ({window_seconds:.1f}s)",
        (24, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        source[:58],
        (24, 166),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    bar_x, bar_y, bar_w, bar_h = 24, 188, 400, 12
    fill_w = int(bar_w * (window_fill / max(window_size, 1)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent, -1)
