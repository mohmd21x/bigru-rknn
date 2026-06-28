#!/usr/bin/env python3
"""Train a fall-detection model from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (
    build_dataloader,
    compute_feature_stats,
    describe_split,
    resolve_manifest_for_split,
)
from src.models.base import build_model
from src.training.trainer import Trainer, set_seed


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def count_missing_features(config: dict, repo_root: Path) -> int:
    missing = 0
    for split in ("train", "val"):
        try:
            manifest = resolve_manifest_for_split(config, split, repo_root)
        except ValueError:
            continue
        for row in manifest:
            if not row.features_path.is_file():
                missing += 1
    return missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fall-detection model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bigru_hierarchical.yaml",
        help="Path to training config YAML.",
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
        help="Skip clips whose feature CSV is missing (for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    missing_features = count_missing_features(config, REPO_ROOT)
    if missing_features and not args.skip_missing_features:
        print(
            f"Error: {missing_features} train/val clips are missing feature CSVs in "
            f"{config.get('data', {}).get('features_dir', 'dataset/features')}.",
            file=sys.stderr,
        )
        print(
            "Run feature extraction first: python scripts/extract_features.py --from-manifests",
            file=sys.stderr,
        )
        print("Or pass --skip-missing-features for a partial smoke run.", file=sys.stderr)
        return 1

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")

    for split in ("train", "val"):
        summary = describe_split(config, split, REPO_ROOT)
        print(f"{split} clips: {summary['clips']}")

    try:
        train_loader = build_dataloader(
            config,
            split="train",
            repo_root=REPO_ROOT,
            skip_missing_features=args.skip_missing_features,
        )
        val_loader = build_dataloader(
            config,
            split="val",
            repo_root=REPO_ROOT,
            skip_missing_features=args.skip_missing_features,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.skip_missing_features:
            print(
                "No windows were built. Extract more feature CSVs or disable "
                "--skip-missing-features once features are available.",
                file=sys.stderr,
            )
        return 1

    model_name = config.get("model", {}).get("name", "bigru_hierarchical")
    model = build_model(model_name, config).to(device)

    # Populate feature-standardization buffers from train statistics (models
    # that support it, e.g. bigru_hierarchical_v2). Saved with the checkpoint.
    if hasattr(model, "set_feature_stats") and not getattr(model, "feature_stats_ready", False):
        mean, std = compute_feature_stats(train_loader.dataset)
        model.set_feature_stats(torch.from_numpy(mean), torch.from_numpy(std))
        print("Initialized feature standardization stats from train split.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        repo_root=REPO_ROOT,
    )
    best_path = trainer.train()
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
