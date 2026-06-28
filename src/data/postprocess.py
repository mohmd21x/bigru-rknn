"""Clip-level aggregation of window-level model predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import NUM_CLASSES


@dataclass(frozen=True)
class ClipPrediction:
    """Aggregated prediction for one source clip."""

    clip_id: str
    label: int
    pred: int
    fall_prob: float
    confidence: float


def aggregate_clip_predictions(
    clip_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    aggregation: str = "clip_max",
    fall_class_id: int = 1,
) -> list[ClipPrediction]:
    """Aggregate window predictions to one prediction per clip.

    For ``clip_max``, the clip fall probability is the maximum fall probability
    across its windows. The predicted class is fall when that probability is at
    least 0.5. Ties on fall probability are broken by the window with the
    highest overall confidence (max class probability).
    """
    if aggregation != "clip_max":
        raise ValueError(f"Unsupported aggregation {aggregation!r}; expected 'clip_max'")

    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"probs must have shape (N, {NUM_CLASSES}), got {probs.shape}")
    if len(clip_ids) != len(labels) or len(clip_ids) != len(probs):
        raise ValueError("clip_ids, labels, and probs must have the same length")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, clip_id in enumerate(clip_ids):
        grouped[clip_id].append(index)

    results: list[ClipPrediction] = []
    for clip_id in sorted(grouped):
        indices = grouped[clip_id]
        clip_labels = labels[indices]
        if not np.all(clip_labels == clip_labels[0]):
            raise ValueError(f"Inconsistent labels for clip {clip_id!r}")
        clip_label = int(clip_labels[0])

        window_probs = probs[indices]
        fall_probs = window_probs[:, fall_class_id]
        confidences = window_probs.max(axis=1)

        best_index = int(np.lexsort((-confidences, -fall_probs))[0])
        fall_prob = float(fall_probs[best_index])
        confidence = float(confidences[best_index])
        pred = fall_class_id if fall_prob >= 0.5 else 1 - fall_class_id

        results.append(
            ClipPrediction(
                clip_id=clip_id,
                label=clip_label,
                pred=pred,
                fall_prob=fall_prob,
                confidence=confidence,
            )
        )

    return results


def clip_predictions_to_arrays(
    predictions: list[ClipPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert clip predictions to parallel label, pred, and fall-prob arrays."""
    if not predictions:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    labels = np.array([item.label for item in predictions], dtype=np.int64)
    preds = np.array([item.pred for item in predictions], dtype=np.int64)
    fall_probs = np.array([item.fall_prob for item in predictions], dtype=np.float64)
    return labels, preds, fall_probs


def summarize_clip_predictions(predictions: list[ClipPrediction]) -> dict[str, Any]:
    """Return simple clip-level counts for logging."""
    labels = [item.label for item in predictions]
    preds = [item.pred for item in predictions]
    return {
        "clips": len(predictions),
        "true_fall": sum(label == 1 for label in labels),
        "true_not_fall": sum(label == 0 for label in labels),
        "pred_fall": sum(pred == 1 for pred in preds),
        "pred_not_fall": sum(pred == 0 for pred in preds),
    }
