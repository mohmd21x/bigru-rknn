"""Streaming causal Kalman upsampler for realtime pose feature extraction.

Design:
- Accepts sparse YOLO detections at ~7fps with irregular timestamps.
- Generates interpolated frames at ``target_fps`` (default 30) using a causal
  forward-only Kalman filter (no RTS, no future data).
- Person presence is controlled by a miss counter, not time-based decay:
    miss_count <= miss_tolerance  → TRACKING / OCCLUDED: keep predicting at
                                    held confidence (no fade)
    miss_count == miss_tolerance+1 → EXIT: immediately zero all output,
                                     reset all state, signal buffer reset
    miss_count >  miss_tolerance+1 → ABSENT: keep returning empty frames
- When the person reappears after a long absence (miss_count > miss_tolerance),
  each keypoint filter is reset before the first update so the new position
  is adopted immediately with zero velocity (no blending from old location).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from kalman_upsample import KalmanFilter4D, NUM_KEYPOINTS  # noqa: E402

from src.inference.pose_features import MIN_CONF as POSE_MIN_CONF
from src.inference.pose_features import PoseFeatureExtractor

DEFAULT_MIN_KPT_CONF = POSE_MIN_CONF
DEFAULT_MIN_VALID_KEYPOINTS = 3


def count_valid_keypoints(
    keypoints: np.ndarray,
    min_kpt_conf: float,
) -> int:
    """Return how many COCO-17 keypoints have confidence >= ``min_kpt_conf``."""
    kpts = np.asarray(keypoints, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
    return sum(1 for k in range(NUM_KEYPOINTS) if float(kpts[k, 2]) >= min_kpt_conf)


def is_valid_pose_detection(
    keypoints: np.ndarray | None,
    *,
    min_kpt_conf: float = DEFAULT_MIN_KPT_CONF,
    min_valid_keypoints: int = DEFAULT_MIN_VALID_KEYPOINTS,
) -> bool:
    """True when enough keypoints pass the confidence gate (rejects YOLO ghost boxes)."""
    if keypoints is None:
        return False
    if min_valid_keypoints <= 0:
        raise ValueError("min_valid_keypoints must be > 0")
    return count_valid_keypoints(keypoints, min_kpt_conf) >= min_valid_keypoints


@dataclass
class KalmanPushResult:
    """Result of a single :meth:`KalmanPoseInterpolator.push` call."""

    frames: list[dict[str, Any]]
    """Feature dicts compatible with :class:`~src.inference.frame_buffer.FallFrameBuffer`."""

    keypoint_frames: list[np.ndarray]
    """COCO-17 keypoints ``(17, 3)`` for each upsampled output timestep."""

    timestamps: list[float]
    """Wall-clock time (seconds) for each entry in ``keypoint_frames``."""

    reset_occurred: bool
    """True when the interpolator crossed the miss threshold and fully reset.

    The caller **must** call ``frame_buffer.reset()`` when this is True to
    prevent mixing pre-absence and post-absence feature sequences.
    """


class KalmanPoseInterpolator:
    """Stateful streaming upsampler: sparse YOLO keypoints → 30fps feature dicts.

    Parameters
    ----------
    target_fps:
        Output frame rate.  Should match the model's training FPS (default 30).
    process_noise:
        Kalman process noise sigma.  Higher = more responsive to measurements,
        less smooth between detections.  Default 20.0.
    meas_noise_base:
        Base measurement noise sigma.  Lower = more trust in YOLO detections.
        Default 10.0.
    miss_tolerance:
        Number of consecutive missed YOLO frames tolerated as occlusion before
        treating the person as having left the scene.  During this window the KF
        keeps predicting at the last real confidence (no fade).
        At 7fps: 2 misses ≈ 0.28 s.  Default 2.
    min_kpt_conf:
        Per-keypoint confidence floor for KF updates and pose validity (default 0.3).
    min_valid_keypoints:
        Minimum keypoints at or above ``min_kpt_conf`` to treat a frame as a real
        person (rejects ghost boxes with box conf but no keypoints).  Default 3.
    max_keypoint_gap_sec:
        Per-keypoint absence threshold (seconds).  After this gap since the last
        valid measurement, the keypoint is zeroed in output.  Default 0.5.
    conf_decay_sec:
        Exponential confidence decay time constant during keypoint gaps.  Default 0.3.
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        process_noise: float = 20.0,
        meas_noise_base: float = 10.0,
        miss_tolerance: int = 2,
        min_kpt_conf: float = DEFAULT_MIN_KPT_CONF,
        min_valid_keypoints: int = DEFAULT_MIN_VALID_KEYPOINTS,
        max_keypoint_gap_sec: float = 0.5,
        conf_decay_sec: float = 0.3,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        if miss_tolerance < 0:
            raise ValueError("miss_tolerance must be >= 0")
        if min_valid_keypoints <= 0:
            raise ValueError("min_valid_keypoints must be > 0")
        if max_keypoint_gap_sec <= 0:
            raise ValueError("max_keypoint_gap_sec must be > 0")
        if conf_decay_sec <= 0:
            raise ValueError("conf_decay_sec must be > 0")

        self.target_fps = float(target_fps)
        self.process_noise = float(process_noise)
        self.meas_noise_base = float(meas_noise_base)
        self.miss_tolerance = int(miss_tolerance)
        self.min_kpt_conf = float(min_kpt_conf)
        self.min_valid_keypoints = int(min_valid_keypoints)
        self.max_keypoint_gap_sec = float(max_keypoint_gap_sec)
        self.conf_decay_sec = float(conf_decay_sec)

        self._filters: list[KalmanFilter4D] = [
            KalmanFilter4D(self.process_noise, self.meas_noise_base)
            for _ in range(NUM_KEYPOINTS)
        ]
        self._feature_extractor = PoseFeatureExtractor()

        # Temporal state
        self._last_push_time: float | None = None
        self._filter_time: float | None = None

        # Per-keypoint presence state
        self._last_valid_conf: list[float] = [0.0] * NUM_KEYPOINTS
        self._last_valid_time: list[float | None] = [None] * NUM_KEYPOINTS

        # Person-level miss counter
        self._consecutive_miss_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Force a full state reset (source change, explicit track loss, etc.)."""
        self._hard_reset()

    def push(
        self,
        keypoints: np.ndarray | None,
        timestamp: float,
    ) -> KalmanPushResult:
        """Advance to ``timestamp`` and return interpolated 30fps feature frames.

        Parameters
        ----------
        keypoints:
            COCO-17 keypoints array of shape ``(17, 3)`` from YOLO, or ``None``
            when no person was detected in this frame.
        timestamp:
            Wall-clock time of this YOLO sample (seconds from video/stream start).

        Returns
        -------
        KalmanPushResult
            ``frames``: list of feature dicts for ``FallFrameBuffer.add()``.
            ``reset_occurred``: caller must call ``frame_buffer.reset()`` if True.
        """
        timestamp = float(timestamp)

        if keypoints is not None:
            keypoints = np.asarray(keypoints, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)

        has_valid = self._has_valid_keypoints(keypoints)

        # --- update miss counter BEFORE any logic ---
        prev_miss_count = self._consecutive_miss_count
        if has_valid:
            self._consecutive_miss_count = 0
        else:
            self._consecutive_miss_count += 1

        # --- handle very first call ---
        if self._last_push_time is None:
            self._last_push_time = timestamp
            self._filter_time = timestamp
            if has_valid:
                assert keypoints is not None
                self._apply_measurement(keypoints, timestamp)
                output_kp = self._build_output_keypoints()
                frames = [self._feature_extractor.update(output_kp, timestamp)]
                return KalmanPushResult(
                    frames=frames,
                    keypoint_frames=[output_kp.copy()],
                    timestamps=[timestamp],
                    reset_occurred=False,
                )
            return KalmanPushResult(
                frames=[],
                keypoint_frames=[],
                timestamps=[],
                reset_occurred=False,
            )

        # --- person just crossed the exit threshold: reset and return empty ---
        if self._consecutive_miss_count == self.miss_tolerance + 1:
            self._hard_reset()
            self._last_push_time = timestamp
            self._filter_time = timestamp
            return KalmanPushResult(
                frames=[],
                keypoint_frames=[],
                timestamps=[],
                reset_occurred=True,
            )

        # --- person already confirmed absent: keep returning empty frames ---
        if self._consecutive_miss_count > self.miss_tolerance + 1:
            self._last_push_time = timestamp
            return KalmanPushResult(
                frames=[],
                keypoint_frames=[],
                timestamps=[],
                reset_occurred=False,
            )

        # --- normal path: person present or within occlusion tolerance ---

        # If the person is returning after a long gap, reset KF per keypoint
        # so we initialize fresh at the new position (no blending from old).
        kf_reset_occurred = False
        if has_valid and prev_miss_count > self.miss_tolerance:
            assert keypoints is not None
            for k in range(NUM_KEYPOINTS):
                if self._kp_valid(keypoints, k):
                    self._filters[k].reset()
            self._feature_extractor.reset()
            kf_reset_occurred = True

        # Build output_times: 30fps grid from just after last_push to timestamp
        output_times = self._compute_output_times(timestamp)

        frames: list[dict[str, Any]] = []
        keypoint_frames: list[np.ndarray] = []
        timestamps: list[float] = []
        for idx, t_out in enumerate(output_times):
            self._predict_to(t_out)

            # Apply the YOLO measurement only at the last (newest) output frame
            if idx == len(output_times) - 1 and has_valid:
                assert keypoints is not None
                self._apply_measurement(keypoints, timestamp)

            output_kp = self._build_output_keypoints()
            keypoint_frames.append(output_kp.copy())
            timestamps.append(float(t_out))
            frames.append(self._feature_extractor.update(output_kp, t_out))

        self._last_push_time = timestamp
        return KalmanPushResult(
            frames=frames,
            keypoint_frames=keypoint_frames,
            timestamps=timestamps,
            reset_occurred=kf_reset_occurred,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hard_reset(self) -> None:
        """Reset all KF state, feature extractor, and tracking counters."""
        for filt in self._filters:
            filt.reset()
        self._feature_extractor.reset()
        self._last_push_time = None
        self._filter_time = None
        self._last_valid_conf = [0.0] * NUM_KEYPOINTS
        self._last_valid_time = [None] * NUM_KEYPOINTS
        self._consecutive_miss_count = 0

    def _kp_valid(self, keypoints: np.ndarray, k: int) -> bool:
        return float(keypoints[k, 2]) >= self.min_kpt_conf

    def _has_valid_keypoints(self, keypoints: np.ndarray | None) -> bool:
        return is_valid_pose_detection(
            keypoints,
            min_kpt_conf=self.min_kpt_conf,
            min_valid_keypoints=self.min_valid_keypoints,
        )

    def _apply_measurement(self, keypoints: np.ndarray, timestamp: float) -> None:
        """Update filters with a valid YOLO measurement."""
        for k in range(NUM_KEYPOINTS):
            if not self._kp_valid(keypoints, k):
                continue
            xy = np.array([keypoints[k, 0], keypoints[k, 1]], dtype=np.float64)
            conf = float(keypoints[k, 2])

            gap_since_valid = (
                timestamp - self._last_valid_time[k]
                if self._last_valid_time[k] is not None
                else float("inf")
            )
            if (
                not self._filters[k].initialized
                or gap_since_valid > self.max_keypoint_gap_sec
            ):
                self._filters[k].reset()
                self._filters[k].initialize(float(xy[0]), float(xy[1]))
            else:
                self._filters[k].update(xy, conf)

            self._last_valid_time[k] = timestamp
            self._last_valid_conf[k] = conf

    def _build_output_keypoints(self) -> np.ndarray:
        """Build a (17, 3) keypoint array from the current KF state.

        Keypoints absent longer than ``max_keypoint_gap_sec`` are zeroed.
        Within the gap window, confidence decays exponentially from the last
        valid measurement so downstream feature gates suppress phantom motion.
        """
        output_kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        current_t = self._filter_time
        for k in range(NUM_KEYPOINTS):
            if self._last_valid_time[k] is None or not self._filters[k].initialized:
                continue
            gap = current_t - self._last_valid_time[k]
            if gap > self.max_keypoint_gap_sec:
                continue
            x_pos, y_pos = self._filters[k].position()
            output_kp[k, 0] = float(x_pos)
            output_kp[k, 1] = float(y_pos)
            output_kp[k, 2] = float(
                self._last_valid_conf[k] * np.exp(-gap / self.conf_decay_sec)
            )
        return output_kp

    def _predict_to(self, t_out: float) -> None:
        """Advance all KF states to ``t_out`` via the predict step."""
        if self._filter_time is None:
            self._filter_time = t_out
            return
        dt = t_out - self._filter_time
        if dt > 1e-12:
            for filt in self._filters:
                filt.predict(dt)
            self._filter_time = t_out

    def _compute_output_times(self, timestamp: float) -> list[float]:
        """Return 30fps output timestamps from last_push_time+dt to timestamp."""
        assert self._last_push_time is not None
        dt_elapsed = timestamp - self._last_push_time
        dt_out = 1.0 / self.target_fps
        n_out = max(1, round(dt_elapsed * self.target_fps))
        if n_out == 1:
            return [timestamp]
        times = [self._last_push_time + dt_out * i for i in range(1, n_out)]
        times.append(timestamp)
        return times
