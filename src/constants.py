"""Shared constants for pose feature columns, COCO-17 splits, and tensor dimensions."""

from __future__ import annotations

# COCO-17 keypoint names (index order used by the feature extractor).
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_KEYPOINTS = len(COCO_KEYPOINT_NAMES)

# Hierarchical body split: head/arms (0–10) vs hips/legs (11–16).
TOP_JOINT_INDICES: tuple[int, ...] = tuple(range(0, 11))
BOTTOM_JOINT_INDICES: tuple[int, ...] = tuple(range(11, 17))

# Manifest rows are joined on these columns; the `path` column in split CSVs is stale.
ALIGN_COLUMNS: tuple[str, ...] = ("video_name", "frame_index", "person_id")

METADATA_COLUMNS: tuple[str, ...] = (
    "video_name",
    "frame_index",
    "timestamp",
    "person_id",
    "valid_prev_pose",
    "estimated_bbox_height",
    "valid_pose",
)

DISTANCE_COLUMNS: tuple[str, ...] = (
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
)

ANGLE_COLUMNS: tuple[str, ...] = (
    "angle_left_hip",
    "angle_right_hip",
    "angle_left_knee",
    "angle_right_knee",
)

TORSO_HIP_COLUMNS: tuple[str, ...] = (
    "torso_angle",
    "torso_angle_std",
    "angle_change",
    "torso_angular_velocity",
    "hip_height",
    "min_hip_height_over_window",
    "hip_height_change",
    "bbox_aspect_ratio",
)

VELOCITY_COLUMNS: tuple[str, ...] = (
    "vel_hip_vx",
    "vel_hip_vy",
    "rolling_mean_vertical_velocity",
    "acceleration",
    "vel_hip_speed",
    "vel_max_wrist_speed",
    "vel_max_ankle_speed",
)

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    DISTANCE_COLUMNS
    + ANGLE_COLUMNS
    + TORSO_HIP_COLUMNS
    + VELOCITY_COLUMNS
)


def norm_kpt_column(joint_index: int, axis: str) -> str:
    """Return a normalized keypoint column name, e.g. ``norm_kpt5_x``."""
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return f"norm_kpt{joint_index}_{axis}"


def kpt_columns_for_joints(joint_indices: tuple[int, ...]) -> tuple[str, ...]:
    """Flatten (x, y) normalized keypoint columns for the given joint indices."""
    cols: list[str] = []
    for idx in joint_indices:
        cols.append(norm_kpt_column(idx, "x"))
        cols.append(norm_kpt_column(idx, "y"))
    return tuple(cols)


NORM_KPT_COLUMNS: tuple[str, ...] = kpt_columns_for_joints(tuple(range(NUM_KEYPOINTS)))
TOP_KPT_COLUMNS: tuple[str, ...] = kpt_columns_for_joints(TOP_JOINT_INDICES)
BOTTOM_KPT_COLUMNS: tuple[str, ...] = kpt_columns_for_joints(BOTTOM_JOINT_INDICES)

# Feature CSV column counts (used for model input sizing).
NORM_KPT_DIM = len(NORM_KPT_COLUMNS)
TOP_KPT_DIM = len(TOP_KPT_COLUMNS)
BOTTOM_KPT_DIM = len(BOTTOM_KPT_COLUMNS)
FEAT_DIM = len(ENGINEERED_FEATURE_COLUMNS)

# Binary classification: index 0 = not_fall, index 1 = fall (matches CrossEntropy class weights order).
CLASS_NAMES: tuple[str, str] = ("not_fall", "fall")
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL: dict[int, str] = {idx: name for name, idx in LABEL_TO_ID.items()}

# Default sliding-window settings (overridden by config YAML).
DEFAULT_WINDOW_SIZE = 64
DEFAULT_WINDOW_STRIDE = 32
