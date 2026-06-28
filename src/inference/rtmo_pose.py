"""RTMO-S ONNX pose inference using MMPOSE exports (NMS + decode in-graph)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]

SCORE_THR = 0.5
INPUT_SIZE = 640
NUM_KEYPOINTS = 17

DEFAULT_ONNX = REPO_ROOT / "rtmo/rtmo-s.onnx"

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def resolve_rtmo_onnx(path: Path) -> Path:
    """Return an existing RTMO ONNX path, with common fallbacks."""
    candidates = [
        path,
        REPO_ROOT / "rtmo/rtmo-s.onnx",
        REPO_ROOT / "rtmo/rtmo-t.onnx",
        REPO_ROOT / "rtmo/rtmo-m.onnx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "RTMO ONNX model not found. Place rtmo-s.onnx under rtmo/ "
        f"(tried {path} and rtmo/rtmo-s.onnx)."
    )


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def preprocess(frame_bgr: np.ndarray, input_size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Letterbox to square input and return padded image plus scale ratio."""
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    ratio = min(input_size / frame_bgr.shape[0], input_size / frame_bgr.shape[1])
    resized = cv2.resize(
        frame_bgr,
        (int(frame_bgr.shape[1] * ratio), int(frame_bgr.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, ratio


def infer_rtmo_s(
    image_bgr: np.ndarray,
    session: ort.InferenceSession,
    score_thr: float = SCORE_THR,
) -> list[dict]:
    """Run RTMO-S ONNX inference and return detections with COCO-17 keypoints."""
    padded, ratio = preprocess(image_bgr)
    inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
    dets, keypoints = session.run(None, {"input": inp})

    det_scores = dets[0, :, 4]
    keep = det_scores >= score_thr
    if not np.any(keep):
        return []

    boxes = dets[0, keep, :4] / ratio
    scores = det_scores[keep]
    kpts = keypoints[0, keep].copy()
    kpts[:, :, :2] /= ratio

    h, w = image_bgr.shape[:2]
    results: list[dict] = []
    for box, score, person_kpts in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        keypoint_rows = []
        for k in range(NUM_KEYPOINTS):
            kx, ky, kc = person_kpts[k]
            kx, ky = np.clip([kx, ky], 0, [w - 1, h - 1])
            keypoint_rows.append((float(kx), float(ky), float(kc)))
        results.append({
            "score": float(score),
            "box_xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "keypoints": keypoint_rows,
        })
    return results


def pick_rtmo_person_keypoints(poses: list[dict], conf_threshold: float) -> np.ndarray | None:
    """Select the highest-score person from RTMO detections."""
    if not poses:
        return None
    best = max(poses, key=lambda p: p["score"])
    if best["score"] < conf_threshold:
        return None
    return np.asarray(best["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)


def draw_poses(img: np.ndarray, poses: list[dict]) -> np.ndarray:
    out = img.copy()
    for p in poses:
        x1, y1, x2, y2 = map(int, p["box_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pts = p["keypoints"]
        for a, b in SKELETON:
            if pts[a][2] > 0.5 and pts[b][2] > 0.5:
                cv2.line(
                    out,
                    (int(pts[a][0]), int(pts[a][1])),
                    (int(pts[b][0]), int(pts[b][1])),
                    (0, 255, 255),
                    2,
                )
        for x, y, c in pts:
            if c > 0.5:
                cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return out
