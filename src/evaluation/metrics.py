"""Classification metrics and evaluation artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import CLASS_NAMES, ID_TO_LABEL, NUM_CLASSES


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute per-class and macro precision, recall, F1, plus accuracy."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    labels = list(range(NUM_CLASSES))
    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class: dict[str, dict[str, float]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        per_class[class_name] = {
            "precision": float(per_class_precision[class_id]),
            "recall": float(per_class_recall[class_id]),
            "f1": float(per_class_f1[class_id]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def compute_confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a ``(num_classes, num_classes)`` confusion matrix."""
    return confusion_matrix(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        labels=list(range(NUM_CLASSES)),
    )


def format_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a sklearn classification report string."""
    target_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    return classification_report(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        zero_division=0,
    )


def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: Path | str,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> None:
    """Save a confusion matrix heatmap to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write metrics JSON, confusion matrix PNG, and classification report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_classification_metrics(y_true, y_pred)
    if extra:
        metrics = {**metrics, **extra}

    cm = compute_confusion_matrix_array(y_true, y_pred)
    report_text = format_classification_report(y_true, y_pred)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    save_confusion_matrix_plot(cm, output_dir / "confusion_matrix.png")

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    return metrics
