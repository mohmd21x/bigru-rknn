"""Online pose feature extraction matching the C++ training pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.constants import (
    BOTTOM_JOINT_INDICES,
    ENGINEERED_FEATURE_COLUMNS,
    TOP_JOINT_INDICES,
)

MIN_CONF = 0.3
TORSO_CONF = 0.5
MIN_HEIGHT = 1.0
ROLLING_VY_WINDOW = 5
HIP_HEIGHT_WINDOW = 5
TORSO_ANGLE_WINDOW = 10


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = a - b
    v2 = c - b
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 < 1e-5 or len2 < 1e-5:
        return None
    cos_theta = float(np.dot(v1, v2) / (len1 * len2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def _torso_side_angle_deg(shoulder: np.ndarray, hip: np.ndarray) -> float:
    dx = abs(float(shoulder[0] - hip[0]))
    dy = abs(float(shoulder[1] - hip[1]))
    if dy == 0.0:
        return 90.0
    return float(np.degrees(np.arctan2(dx, dy)))


@dataclass
class _TrackState:
  valid_pose: bool = False
  keypoints: np.ndarray = field(default_factory=lambda: np.zeros((17, 3), dtype=np.float32))
  bbox_height: float = 0.0
  timestamp: float = 0.0
  hip_center: np.ndarray | None = None
  recent_vy: deque[float] = field(default_factory=deque)
  recent_hip_height: deque[float] = field(default_factory=deque)
  recent_torso_angle: deque[float] = field(default_factory=deque)
  last_rolling_mean_vy: float | None = None


class PoseFeatureExtractor:
    """Convert YOLO COCO-17 keypoints into model-ready per-frame tensors."""

    def __init__(self) -> None:
        self._prev = _TrackState()

    def reset(self) -> None:
        self._prev = _TrackState()

    @staticmethod
    def _estimate_bbox_height(keypoints: np.ndarray, conf_min: float = MIN_CONF) -> float:
        valid = keypoints[keypoints[:, 2] >= conf_min]
        if len(valid) == 0:
            return 0.0
        ys = valid[:, 1]
        span = float(ys.max() - ys.min())
        return max(span, MIN_HEIGHT)

    @staticmethod
    def _hip_center(keypoints: np.ndarray, conf_min: float = MIN_CONF) -> np.ndarray | None:
        left = keypoints[11]
        right = keypoints[12]
        left_ok = left[2] >= conf_min
        right_ok = right[2] >= conf_min
        if not left_ok and not right_ok:
            return None
        if left_ok and right_ok:
            return np.array([(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5], dtype=np.float32)
        if left_ok:
            return left[:2].copy()
        return right[:2].copy()

    @staticmethod
    def _point_if_conf(keypoints: np.ndarray, idx: int, conf_min: float = MIN_CONF) -> np.ndarray | None:
        kp = keypoints[idx]
        if kp[2] < conf_min:
            return None
        return kp[:2]

    def _valid_pose(self, keypoints: np.ndarray, bbox_height: float) -> bool:
        if keypoints.shape[0] != 17:
            return False
        hip = self._hip_center(keypoints, MIN_CONF)
        return hip is not None and bbox_height >= MIN_HEIGHT

    def _compute_torso_angle(self, keypoints: np.ndarray) -> float | None:
        left_ok = keypoints[5, 2] > TORSO_CONF and keypoints[11, 2] > TORSO_CONF
        right_ok = keypoints[6, 2] > TORSO_CONF and keypoints[12, 2] > TORSO_CONF
        if not left_ok and not right_ok:
            return None
        if left_ok and right_ok:
            return 0.5 * (
                _torso_side_angle_deg(keypoints[5, :2], keypoints[11, :2])
                + _torso_side_angle_deg(keypoints[6, :2], keypoints[12, :2])
            )
        if left_ok:
            return _torso_side_angle_deg(keypoints[5, :2], keypoints[11, :2])
        return _torso_side_angle_deg(keypoints[6, :2], keypoints[12, :2])

    def _normalized_pose(self, keypoints: np.ndarray, bbox_height: float) -> np.ndarray | None:
        hip = self._hip_center(keypoints, MIN_CONF)
        if hip is None or bbox_height <= 0:
            return None
        out = np.zeros(34, dtype=np.float32)
        for i in range(17):
            out[2 * i] = (keypoints[i, 0] - hip[0]) / bbox_height
            out[2 * i + 1] = (keypoints[i, 1] - hip[1]) / bbox_height
        return out

    def _pairwise_distances(self, keypoints: np.ndarray, bbox_height: float) -> dict[str, float]:
        h = max(bbox_height, MIN_HEIGHT)
        p = lambda idx: self._point_if_conf(keypoints, idx)
        hip_c = self._hip_center(keypoints)
        out: dict[str, float] = {}
        ls, rs = p(5), p(6)
        lh, rh = p(11), p(12)
        if ls is not None and rs is not None:
            out["dist_shoulder_width"] = _dist(ls, rs) / h
        if lh is not None and rh is not None:
            out["dist_hip_width"] = _dist(lh, rh) / h
        nose = p(0)
        if nose is not None and hip_c is not None:
            out["dist_nose_to_hip"] = _dist(nose, hip_c) / h
        lk, rk, la, ra = p(13), p(14), p(15), p(16)
        if lh is not None and lk is not None:
            out["dist_left_thigh"] = _dist(lh, lk) / h
        if rh is not None and rk is not None:
            out["dist_right_thigh"] = _dist(rh, rk) / h
        if lk is not None and la is not None:
            out["dist_left_shin"] = _dist(lk, la) / h
        if rk is not None and ra is not None:
            out["dist_right_shin"] = _dist(rk, ra) / h
        lw, rw = p(9), p(10)
        if lw is not None and rw is not None:
            out["dist_hand_to_hand"] = _dist(lw, rw) / h
        if lw is not None and hip_c is not None:
            out["dist_left_hand_to_hip"] = _dist(lw, hip_c) / h
        if rw is not None and hip_c is not None:
            out["dist_right_hand_to_hip"] = _dist(rw, hip_c) / h
        return out

    def _joint_angles(self, keypoints: np.ndarray) -> dict[str, float | None]:
        p = lambda idx: self._point_if_conf(keypoints, idx)
        lh, lk, ls = p(11), p(13), p(5)
        rh, rk, rs = p(12), p(14), p(6)
        la, ra = p(15), p(16)
        return {
            "angle_left_hip": _angle_at_joint(lk, lh, ls) if all(x is not None for x in (lk, lh, ls)) else None,
            "angle_right_hip": _angle_at_joint(rk, rh, rs) if all(x is not None for x in (rk, rh, rs)) else None,
            "angle_left_knee": _angle_at_joint(lh, lk, la) if all(x is not None for x in (lh, lk, la)) else None,
            "angle_right_knee": _angle_at_joint(rh, rk, ra) if all(x is not None for x in (rh, rk, ra)) else None,
        }

    def _hip_height(self, keypoints: np.ndarray, bbox_height: float) -> float | None:
        hip = self._hip_center(keypoints)
        if hip is None or bbox_height <= 0:
            return None
        return float(hip[1] / bbox_height)

    def _bbox_aspect_ratio(self, keypoints: np.ndarray, bbox_height: float) -> float | None:
        valid = keypoints[keypoints[:, 2] >= MIN_CONF]
        if len(valid) == 0 or bbox_height <= 0:
            return None
        width = max(float(valid[:, 0].max() - valid[:, 0].min()), 1.0)
        return width / bbox_height

    def _velocity_features(
        self,
        prev_kp: np.ndarray,
        curr_kp: np.ndarray,
        prev_hip: np.ndarray,
        curr_hip: np.ndarray,
        dt: float,
        bbox_height: float,
    ) -> dict[str, float]:
        h = max(bbox_height, MIN_HEIGHT)
        vx = ((curr_hip[0] - prev_hip[0]) / dt) / h
        vy = ((curr_hip[1] - prev_hip[1]) / dt) / h
        speed = float(np.hypot(vx, vy))

        def limb_speed(idx: int) -> float:
            if prev_kp[idx, 2] < MIN_CONF or curr_kp[idx, 2] < MIN_CONF:
                return 0.0
            dx = curr_kp[idx, 0] - prev_kp[idx, 0]
            dy = curr_kp[idx, 1] - prev_kp[idx, 1]
            return float(np.hypot(dx, dy) / (dt * h))

        return {
            "vel_hip_vx": vx,
            "vel_hip_vy": vy,
            "vel_hip_speed": speed,
            "vel_max_wrist_speed": max(limb_speed(9), limb_speed(10)),
            "vel_max_ankle_speed": max(limb_speed(15), limb_speed(16)),
        }

    def update(self, keypoints: np.ndarray, timestamp: float) -> dict[str, Any]:
        """Update track state and return per-frame tensors for the model."""
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        bbox_height = self._estimate_bbox_height(keypoints)
        valid_pose = self._valid_pose(keypoints, bbox_height)

        top_kp = np.zeros(22, dtype=np.float32)
        bot_kp = np.zeros(12, dtype=np.float32)
        feat = np.zeros(len(ENGINEERED_FEATURE_COLUMNS), dtype=np.float32)
        mask = 1.0 if valid_pose else 0.0

        engineered: dict[str, float | None] = {name: None for name in ENGINEERED_FEATURE_COLUMNS}

        if valid_pose:
            norm = self._normalized_pose(keypoints, bbox_height)
            if norm is not None:
                top_idx = [2 * i for i in TOP_JOINT_INDICES] + [2 * i + 1 for i in TOP_JOINT_INDICES]
                bot_idx = [2 * i for i in BOTTOM_JOINT_INDICES] + [2 * i + 1 for i in BOTTOM_JOINT_INDICES]
                # reorder to match column order: x0,y0,x1,y1 for top joints 0-10
                top_pairs = [(norm[2 * i], norm[2 * i + 1]) for i in TOP_JOINT_INDICES]
                bot_pairs = [(norm[2 * i], norm[2 * i + 1]) for i in BOTTOM_JOINT_INDICES]
                top_kp = np.array([v for pair in top_pairs for v in pair], dtype=np.float32)
                bot_kp = np.array([v for pair in bot_pairs for v in pair], dtype=np.float32)

            for name, value in self._pairwise_distances(keypoints, bbox_height).items():
                engineered[name] = value
            for name, value in self._joint_angles(keypoints).items():
                engineered[name] = value

            torso_angle = self._compute_torso_angle(keypoints)
            hip_height = self._hip_height(keypoints, bbox_height)
            engineered["torso_angle"] = torso_angle
            engineered["hip_height"] = hip_height
            engineered["bbox_aspect_ratio"] = self._bbox_aspect_ratio(keypoints, bbox_height)

            if torso_angle is not None:
                self._prev.recent_torso_angle.append(torso_angle)
                if len(self._prev.recent_torso_angle) > TORSO_ANGLE_WINDOW:
                    self._prev.recent_torso_angle.popleft()
                if len(self._prev.recent_torso_angle) >= 2:
                    arr = np.array(self._prev.recent_torso_angle, dtype=np.float32)
                    engineered["torso_angle_std"] = float(arr.std())

            if hip_height is not None:
                self._prev.recent_hip_height.append(hip_height)
                if len(self._prev.recent_hip_height) > HIP_HEIGHT_WINDOW:
                    self._prev.recent_hip_height.popleft()
                engineered["min_hip_height_over_window"] = float(min(self._prev.recent_hip_height))

            curr_hip = self._hip_center(keypoints)
            dt = timestamp - self._prev.timestamp
            velocity: dict[str, float] = {}
            rolling_mean_vy: float | None = None
            acceleration: float | None = None

            if (
                self._prev.valid_pose
                and valid_pose
                and self._prev.hip_center is not None
                and curr_hip is not None
                and dt > 0
            ):
                velocity = self._velocity_features(
                    self._prev.keypoints,
                    keypoints,
                    self._prev.hip_center,
                    curr_hip,
                    dt,
                    bbox_height,
                )
                prev_torso = self._compute_torso_angle(self._prev.keypoints)
                prev_hip_h = self._hip_height(self._prev.keypoints, self._prev.bbox_height)
                if torso_angle is not None and prev_torso is not None:
                    engineered["angle_change"] = torso_angle - prev_torso
                    engineered["torso_angular_velocity"] = engineered["angle_change"] / dt
                if hip_height is not None and prev_hip_h is not None:
                    engineered["hip_height_change"] = hip_height - prev_hip_h

                vy = velocity["vel_hip_vy"]
                self._prev.recent_vy.append(vy)
                if len(self._prev.recent_vy) > ROLLING_VY_WINDOW:
                    self._prev.recent_vy.popleft()
                rolling_mean_vy = float(np.mean(self._prev.recent_vy))
                velocity["rolling_mean_vertical_velocity"] = rolling_mean_vy
                if self._prev.last_rolling_mean_vy is not None:
                    acceleration = (rolling_mean_vy - self._prev.last_rolling_mean_vy) / dt
                velocity["acceleration"] = acceleration if acceleration is not None else 0.0

            for name, value in velocity.items():
                engineered[name] = value

        for i, name in enumerate(ENGINEERED_FEATURE_COLUMNS):
            value = engineered.get(name)
            if value is not None and np.isfinite(value):
                feat[i] = float(value)

        self._prev = _TrackState(
            valid_pose=valid_pose,
            keypoints=keypoints.copy(),
            bbox_height=bbox_height,
            timestamp=timestamp,
            hip_center=self._hip_center(keypoints),
            recent_vy=deque(self._prev.recent_vy, maxlen=ROLLING_VY_WINDOW),
            recent_hip_height=deque(self._prev.recent_hip_height, maxlen=HIP_HEIGHT_WINDOW),
            recent_torso_angle=deque(self._prev.recent_torso_angle, maxlen=TORSO_ANGLE_WINDOW),
            last_rolling_mean_vy=engineered.get("rolling_mean_vertical_velocity"),
        )

        return {
            "top_kp": top_kp,
            "bot_kp": bot_kp,
            "feat": feat,
            "mask": mask,
            "valid_pose": valid_pose,
        }
