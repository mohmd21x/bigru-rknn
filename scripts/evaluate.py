#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split at clip level."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_dataloader, describe_split
from src.data.postprocess import (
    aggregate_clip_predictions,
    clip_predictions_to_arrays,
    summarize_clip_predictions,
)
from src.evaluation.metrics import save_evaluation_artifacts
from src.models.base import build_model
from src.training.trainer import move_batch_to_device, set_seed


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fall-detection checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bigru_hierarchical.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a model checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
        help="Skip clips whose feature CSV is missing.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Override evaluation.reports_dir from config.",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def collect_window_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Run inference and return clip ids, labels, and class probabilities."""
    model.eval()
    clip_ids: list[str] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="eval", unit="batch"):
        batch_clip_ids = batch["clip_id"]
        batch_labels = batch["label"]
        if torch.is_tensor(batch_labels):
            batch_labels = batch_labels.detach().cpu().tolist()

        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        batch_probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

        clip_ids.extend(batch_clip_ids)
        labels.extend(int(label) for label in batch_labels)
        probs.append(batch_probs)

    if not probs:
        return [], np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64)

    return clip_ids, np.asarray(labels, dtype=np.int64), np.concatenate(probs, axis=0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()

    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1
    if not checkpoint_path.is_file():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint.get("config"), dict):
        config = checkpoint["config"]

    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    print(f"Using device: {device}")

    test_summary = describe_split(config, "test", REPO_ROOT)
    print(f"test clips: {test_summary['clips']}")

    test_loader = build_dataloader(
        config,
        split="test",
        repo_root=REPO_ROOT,
        shuffle=False,
        skip_missing_features=args.skip_missing_features,
    )

    model_name = checkpoint.get("model_name") or config.get("model", {}).get(
        "name", "bigru_hierarchical"
    )
    model = build_model(model_name, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    clip_ids, window_labels, window_probs = collect_window_predictions(
        model, test_loader, device
    )
    if len(clip_ids) == 0:
        print("Error: no windows available for evaluation.", file=sys.stderr)
        return 1

    aggregation = config.get("evaluation", {}).get("aggregation", "clip_max")
    clip_preds = aggregate_clip_predictions(
        clip_ids,
        window_labels,
        window_probs,
        aggregation=aggregation,
    )
    y_true, y_pred, _fall_probs = clip_predictions_to_arrays(clip_preds)
    summary = summarize_clip_predictions(clip_preds)

    run_name = str(config.get("run_name", model_name))
    reports_root = args.reports_dir or config.get("evaluation", {}).get("reports_dir", "reports")
    output_dir = Path(reports_root)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / run_name

    metrics = save_evaluation_artifacts(
        y_true,
        y_pred,
        output_dir,
        extra={
            "aggregation": aggregation,
            "checkpoint": str(checkpoint_path),
            "clip_summary": summary,
            "num_windows": len(clip_ids),
            "num_clips": len(clip_preds),
        },
    )

    print(f"Clip-level evaluation ({aggregation}):")
    print(f"  clips={summary['clips']} windows={len(clip_ids)}")
    print(f"  accuracy={metrics['accuracy']:.4f}")
    print(f"  macro_f1={metrics['macro_f1']:.4f}")
    print(f"  fall_f1={metrics['per_class']['fall']['f1']:.4f}")
    print(f"Reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
