"""Evaluation metrics and clip-level aggregation."""

from src.data.postprocess import (
    ClipPrediction,
    aggregate_clip_predictions,
    clip_predictions_to_arrays,
    summarize_clip_predictions,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix_array,
    format_classification_report,
    save_confusion_matrix_plot,
    save_evaluation_artifacts,
)

__all__ = [
    "ClipPrediction",
    "aggregate_clip_predictions",
    "clip_predictions_to_arrays",
    "compute_classification_metrics",
    "compute_confusion_matrix_array",
    "format_classification_report",
    "save_confusion_matrix_plot",
    "save_evaluation_artifacts",
    "summarize_clip_predictions",
]
