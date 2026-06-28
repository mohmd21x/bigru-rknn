"""Train-time pose augmentation for fall-detection windows.

Augmentations operate on a single pre-windowed sample (numpy arrays):

- ``top_kp``: ``(T, TOP_KPT_DIM)``   normalized (x, y) for joints 0..10
- ``bot_kp``: ``(T, BOTTOM_KPT_DIM)`` normalized (x, y) for joints 11..16
- ``feat``:   ``(T, FEAT_DIM)``       engineered features
- ``mask``:   ``(T,)``                per-frame validity

All keypoints are hip-centered and bbox-height-normalized, so a horizontal
flip is a sign flip of x plus a left/right joint swap. Augmentations are only
applied on the train split and are individually gated by probabilities from
``data.augment`` in the config.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constants import (
    BOTTOM_JOINT_INDICES,
    ENGINEERED_FEATURE_COLUMNS,
    TOP_JOINT_INDICES,
)

# COCO-17 left/right joint index pairs (mirror symmetry).
_LR_JOINT_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

# Engineered feature columns whose left/right counterparts swap on h-flip.
_LR_FEATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("dist_left_thigh", "dist_right_thigh"),
    ("dist_left_shin", "dist_right_shin"),
    ("dist_left_hand_to_hip", "dist_right_hand_to_hip"),
    ("angle_left_hip", "angle_right_hip"),
    ("angle_left_knee", "angle_right_knee"),
)

# Engineered features that flip sign on horizontal mirror (signed horizontal vel).
_SIGN_FLIP_FEATURES: tuple[str, ...] = ("vel_hip_vx",)

# Distance / velocity features that scale linearly with a spatial scale factor.
_SCALE_FEATURES: tuple[str, ...] = (
    "dist_shoulder_width",
    "dist_hip_width",
    "dist_nose_to_hip",
    "dist_left_thigh",
    "dist_right_thigh",
    "dist_left_shin",
    "dist_right_shin",
    "dist_hand_to_hand",
    "dist_left_hand_to_hip",
    "dist_right_hand_to_hip",
    "vel_hip_vx",
    "vel_hip_vy",
    "rolling_mean_vertical_velocity",
    "acceleration",
    "vel_hip_speed",
    "vel_max_wrist_speed",
    "vel_max_ankle_speed",
)


def _joint_to_local_cols(joint_indices: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Map a joint index to its (x_col, y_col) within a flattened kp stream."""
    return {j: (2 * pos, 2 * pos + 1) for pos, j in enumerate(joint_indices)}


_TOP_COLS = _joint_to_local_cols(TOP_JOINT_INDICES)
_BOT_COLS = _joint_to_local_cols(BOTTOM_JOINT_INDICES)
_FEAT_INDEX = {name: i for i, name in enumerate(ENGINEERED_FEATURE_COLUMNS)}


class PoseAugmentor:
    """Apply randomized, label-preserving augmentations to a single window."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.hflip_prob = float(config.get("hflip_prob", 0.5))
        self.scale_prob = float(config.get("scale_prob", 0.5))
        self.scale_range = tuple(config.get("scale_range", (0.9, 1.1)))
        self.coord_noise_std = float(config.get("coord_noise_std", 0.02))
        self.joint_dropout_prob = float(config.get("joint_dropout_prob", 0.0))
        self.max_dropped_joints = int(config.get("max_dropped_joints", 2))
        self.frame_dropout_prob = float(config.get("frame_dropout_prob", 0.0))
        self.max_dropped_frames = int(config.get("max_dropped_frames", 2))

    # -- individual ops ---------------------------------------------------
    def _hflip(self, top: np.ndarray, bot: np.ndarray, feat: np.ndarray) -> None:
        # Negate x for every joint in both streams.
        for cols in (_TOP_COLS, _BOT_COLS):
            stream = top if cols is _TOP_COLS else bot
            for _, (xc, _yc) in cols.items():
                stream[:, xc] *= -1.0

        # Swap left/right joint (x, y) pairs within each stream.
        for a, b in _LR_JOINT_PAIRS:
            for cols, stream in ((_TOP_COLS, top), (_BOT_COLS, bot)):
                if a in cols and b in cols:
                    axc, ayc = cols[a]
                    bxc, byc = cols[b]
                    stream[:, [axc, ayc, bxc, byc]] = stream[:, [bxc, byc, axc, ayc]]

        # Swap left/right engineered feature pairs.
        for left, right in _LR_FEATURE_PAIRS:
            li, ri = _FEAT_INDEX.get(left), _FEAT_INDEX.get(right)
            if li is not None and ri is not None:
                feat[:, [li, ri]] = feat[:, [ri, li]]

        # Negate signed horizontal-velocity features.
        for name in _SIGN_FLIP_FEATURES:
            idx = _FEAT_INDEX.get(name)
            if idx is not None:
                feat[:, idx] *= -1.0

    def _scale(self, top: np.ndarray, bot: np.ndarray, feat: np.ndarray, rng: np.random.Generator) -> None:
        s = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        top *= s
        bot *= s
        for name in _SCALE_FEATURES:
            idx = _FEAT_INDEX.get(name)
            if idx is not None:
                feat[:, idx] *= s

    def _coord_noise(self, top: np.ndarray, bot: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
        valid = mask > 0
        if not valid.any():
            return
        for stream in (top, bot):
            noise = rng.normal(0.0, self.coord_noise_std, size=stream.shape).astype(stream.dtype)
            noise[~valid] = 0.0
            stream += noise

    def _joint_dropout(self, top: np.ndarray, bot: np.ndarray, rng: np.random.Generator) -> None:
        n_drop = int(rng.integers(1, self.max_dropped_joints + 1))
        joints = rng.choice(17, size=min(n_drop, 17), replace=False)
        for j in joints:
            if j in _TOP_COLS:
                xc, yc = _TOP_COLS[j]
                top[:, [xc, yc]] = 0.0
            elif j in _BOT_COLS:
                xc, yc = _BOT_COLS[j]
                bot[:, [xc, yc]] = 0.0

    def _frame_dropout(
        self,
        top: np.ndarray,
        bot: np.ndarray,
        feat: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        valid_idx = np.flatnonzero(mask > 0)
        # Keep at least one valid frame so the window stays usable.
        if valid_idx.size <= 1:
            return
        n_drop = int(rng.integers(1, self.max_dropped_frames + 1))
        n_drop = min(n_drop, valid_idx.size - 1)
        drop = rng.choice(valid_idx, size=n_drop, replace=False)
        top[drop] = 0.0
        bot[drop] = 0.0
        feat[drop] = 0.0
        mask[drop] = 0.0

    # -- entry point ------------------------------------------------------
    def __call__(self, sample: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        if not self.enabled:
            return sample

        top = np.array(sample["top_kp"], dtype=np.float32, copy=True)
        bot = np.array(sample["bot_kp"], dtype=np.float32, copy=True)
        feat = np.array(sample["feat"], dtype=np.float32, copy=True)
        mask = np.array(sample["mask"], dtype=np.float32, copy=True)

        if self.hflip_prob > 0 and rng.random() < self.hflip_prob:
            self._hflip(top, bot, feat)
        if self.scale_prob > 0 and rng.random() < self.scale_prob:
            self._scale(top, bot, feat, rng)
        if self.coord_noise_std > 0:
            self._coord_noise(top, bot, mask, rng)
        if self.joint_dropout_prob > 0 and rng.random() < self.joint_dropout_prob:
            self._joint_dropout(top, bot, rng)
        if self.frame_dropout_prob > 0 and rng.random() < self.frame_dropout_prob:
            self._frame_dropout(top, bot, feat, mask, rng)

        return {**sample, "top_kp": top, "bot_kp": bot, "feat": feat, "mask": mask}
