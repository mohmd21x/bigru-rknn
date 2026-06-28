"""Data loading, preprocessing, and windowing."""

from src.data.dataset import FallWindowDataset, build_dataloader, build_dataset
from src.data.preprocess import (
    ClipTensors,
    ManifestRow,
    load_clip,
    load_clip_from_manifest,
    load_manifest,
    make_train_val_split,
)
from src.data.postprocess import (
    ClipPrediction,
    aggregate_clip_predictions,
    clip_predictions_to_arrays,
    summarize_clip_predictions,
)
from src.data.windowing import WindowSlice, slide_windows

__all__ = [
    "ClipPrediction",
    "ClipTensors",
    "FallWindowDataset",
    "ManifestRow",
    "WindowSlice",
    "aggregate_clip_predictions",
    "build_dataloader",
    "build_dataset",
    "clip_predictions_to_arrays",
    "load_clip",
    "load_clip_from_manifest",
    "load_manifest",
    "make_train_val_split",
    "slide_windows",
    "summarize_clip_predictions",
]
