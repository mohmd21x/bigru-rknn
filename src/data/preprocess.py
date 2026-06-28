"""CSV loading, manifest resolution, and clip-level preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    ALIGN_COLUMNS,
    BOTTOM_KPT_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    LABEL_TO_ID,
    TOP_KPT_COLUMNS,
)


class EmptyClipError(ValueError):
    """Raised when a clip CSV exists but contains no usable frames."""


@dataclass(frozen=True)
class ManifestRow:
    """One clip entry resolved from a split manifest."""

    filename: str
    label: str
    label_id: int
    keypoints_path: Path
    features_path: Path
    split: str


@dataclass
class ClipTensors:
    """Preprocessed per-frame tensors for a single clip."""

    clip_id: str
    label_id: int
    top_kp: np.ndarray
    bot_kp: np.ndarray
    feat: np.ndarray
    valid_mask: np.ndarray


def feature_path_for_keypoints(keypoints_path: Path, features_dir: Path) -> Path:
    """Map ``foo_keypoints.csv`` to ``features_dir/foo_keypoints_features.csv``."""
    return features_dir / f"{keypoints_path.stem}_features.csv"


def load_manifest(
    manifest_path: Path | str,
    outputs_dir: Path | str,
    features_dir: Path | str | None = None,
) -> list[ManifestRow]:
    """Read a split CSV and resolve keypoint/feature paths from ``filename``.

    The ``path`` column in split manifests is stale; paths are always built from
    ``filename`` under ``outputs_dir`` and ``features_dir``.
    """
    manifest_path = Path(manifest_path)
    outputs_dir = Path(outputs_dir)
    features_dir = Path(features_dir) if features_dir is not None else outputs_dir.parent / "features"

    df = pd.read_csv(manifest_path)
    required = {"filename", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} missing columns: {sorted(missing)}")

    split_name = df["split"].iloc[0] if "split" in df.columns and len(df) else manifest_path.stem

    rows: list[ManifestRow] = []
    for _, record in df.iterrows():
        filename = str(record["filename"]).strip()
        if not filename:
            continue
        label = str(record["label"]).strip()
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unknown label {label!r} in {manifest_path}")

        keypoints_path = outputs_dir / filename
        features_path = feature_path_for_keypoints(keypoints_path, features_dir)
        row_split = str(record["split"]).strip() if "split" in df.columns else split_name

        rows.append(
            ManifestRow(
                filename=filename,
                label=label,
                label_id=LABEL_TO_ID[label],
                keypoints_path=keypoints_path,
                features_path=features_path,
                split=row_split,
            )
        )
    return rows


def make_train_val_split(
    train_manifest: list[ManifestRow],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Stratified hold-out of ``val_ratio`` clips from the train manifest."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if len(train_manifest) < 2:
        raise ValueError("Need at least 2 train clips for a validation split")

    labels = [row.label_id for row in train_manifest]
    indices = np.arange(len(train_manifest))

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        # Too few samples per class for stratification; fall back to random split.
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )

    train_rows = [train_manifest[i] for i in train_idx]
    val_rows = [train_manifest[i] for i in val_idx]
    return train_rows, val_rows


def _to_bool_mask(series: pd.Series) -> np.ndarray:
    """Convert ``valid_pose`` column values to a float mask in {0.0, 1.0}."""
    if series.dtype == bool:
        return series.to_numpy(dtype=np.float32)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric > 0).to_numpy(dtype=np.float32)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    """Raise a clear error when expected columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}: {path}")


def _align_frames(keypoints_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join keypoints and features on ``ALIGN_COLUMNS``."""
    keypoints_df = keypoints_df.copy()
    features_df = features_df.copy()
    for col in ALIGN_COLUMNS:
        if col in ("frame_index", "person_id"):
            keypoints_df[col] = pd.to_numeric(keypoints_df[col], errors="coerce").astype("Int64")
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype("Int64")

    merged = keypoints_df.merge(
        features_df,
        on=list(ALIGN_COLUMNS),
        how="inner",
        suffixes=("_kp", "_feat"),
    )
    for col in list(merged.columns):
        if col.endswith("_feat"):
            base = col[: -len("_feat")]
            kp_col = f"{base}_kp"
            if kp_col in merged.columns:
                merged[base] = merged[col]
                merged = merged.drop(columns=[kp_col, col])
    return merged


def load_clip(
    keypoints_path: Path | str,
    features_path: Path | str,
    clip_id: str | None = None,
) -> ClipTensors:
    """Load and align keypoint + feature CSVs into per-frame model inputs.

    Rows are aligned on ``(video_name, frame_index, person_id)``. NaN feature
    values are replaced with zero. ``valid_mask`` comes from ``valid_pose``.
    """
    keypoints_path = Path(keypoints_path)
    features_path = Path(features_path)

    if not features_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {features_path}")

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        raise EmptyClipError(f"Feature CSV has no rows: {features_path}")
    _require_columns(features_df, ALIGN_COLUMNS, features_path)
    _require_columns(
        features_df,
        TOP_KPT_COLUMNS + BOTTOM_KPT_COLUMNS + ENGINEERED_FEATURE_COLUMNS + ("valid_pose",),
        features_path,
    )

    if keypoints_path.is_file():
        keypoints_df = pd.read_csv(keypoints_path)
        _require_columns(keypoints_df, ALIGN_COLUMNS, keypoints_path)
        merged = _align_frames(keypoints_df, features_df)
    else:
        merged = features_df

    if merged.empty:
        raise EmptyClipError(f"No aligned frames for clip {features_path}")

    merged = merged.sort_values(list(ALIGN_COLUMNS)).reset_index(drop=True)

    top_kp = (
        merged[list(TOP_KPT_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
    )
    bot_kp = (
        merged[list(BOTTOM_KPT_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    feat = (
        merged[list(ENGINEERED_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    if "valid_pose" not in merged.columns:
        raise ValueError(f"Feature CSV missing 'valid_pose' column: {features_path}")
    valid_mask = _to_bool_mask(merged["valid_pose"])

    resolved_clip_id = clip_id or keypoints_path.stem
    return ClipTensors(
        clip_id=resolved_clip_id,
        label_id=-1,
        top_kp=top_kp,
        bot_kp=bot_kp,
        feat=feat,
        valid_mask=valid_mask,
    )


def load_clip_from_manifest(row: ManifestRow) -> ClipTensors:
    """Load a clip and attach manifest label metadata."""
    clip = load_clip(row.keypoints_path, row.features_path, clip_id=row.filename)
    clip.label_id = row.label_id
    return clip


def manifest_summary(manifest: list[ManifestRow]) -> dict[str, Any]:
    """Return clip counts per label for logging."""
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row.label] = counts.get(row.label, 0) + 1
    return {"total": len(manifest), "per_label": counts}
