"""PyTorch dataset and DataLoader factory for fall-detection windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.constants import (
    BOTTOM_KPT_DIM,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    FEAT_DIM,
    TOP_KPT_DIM,
)
from src.data.augment import PoseAugmentor
from src.data.preprocess import (
    EmptyClipError,
    ManifestRow,
    load_clip_from_manifest,
    load_manifest,
    make_train_val_split,
    manifest_summary,
)
from src.data.windowing import slide_windows, valid_frame_ratio


class FallWindowDataset(Dataset):
    f"""Dataset of fixed-length windows for hierarchical BiGRU training.

    Each item contains:

    - ``top_kp``: ``(T, {TOP_KPT_DIM})``
    - ``bot_kp``: ``(T, {BOTTOM_KPT_DIM})``
    - ``feat``: ``(T, {FEAT_DIM})``
    - ``mask``: ``(T,)``
    - ``label``: int class id
    - ``clip_id``: source clip filename
    """

    def __init__(
        self,
        manifest: list[ManifestRow],
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_WINDOW_STRIDE,
        min_valid_frame_ratio: float = 0.5,
        skip_missing_features: bool = False,
        augmentor: PoseAugmentor | None = None,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.min_valid_frame_ratio = min_valid_frame_ratio
        self.augmentor = augmentor

        self.samples: list[dict[str, Any]] = []
        self._skipped: list[str] = []
        self._skipped_empty: list[str] = []

        for row in manifest:
            if skip_missing_features and not row.features_path.is_file():
                self._skipped.append(row.filename)
                continue
            try:
                clip = load_clip_from_manifest(row)
            except EmptyClipError:
                self._skipped_empty.append(row.filename)
                continue
            except FileNotFoundError as exc:
                if skip_missing_features:
                    self._skipped.append(row.filename)
                    continue
                raise RuntimeError(f"Failed to load clip {row.filename}") from exc
            except ValueError as exc:
                raise RuntimeError(f"Failed to load clip {row.filename}") from exc

            for window in slide_windows(
                clip.top_kp,
                clip.bot_kp,
                clip.feat,
                clip.valid_mask,
                window_size=window_size,
                stride=stride,
            ):
                if valid_frame_ratio(window.mask) < min_valid_frame_ratio:
                    continue
                self.samples.append(
                    {
                        "top_kp": window.top_kp,
                        "bot_kp": window.bot_kp,
                        "feat": window.feat,
                        "mask": window.mask,
                        "label": clip.label_id,
                        "clip_id": clip.clip_id,
                    }
                )

        if not self.samples:
            detail_parts = []
            if self._skipped:
                detail_parts.append(f"missing features: {len(self._skipped)}")
            if self._skipped_empty:
                detail_parts.append(f"empty clips: {len(self._skipped_empty)}")
            detail = f" ({'; '.join(detail_parts)})" if detail_parts else ""
            raise RuntimeError(f"No valid windows built from manifest{detail}")

    @property
    def skipped_clips(self) -> tuple[str, ...]:
        """Filenames skipped when ``skip_missing_features`` is enabled."""
        return tuple(self._skipped)

    @property
    def skipped_empty_clips(self) -> tuple[str, ...]:
        """Filenames skipped because the keypoint/feature CSVs contain no frames."""
        return tuple(self._skipped_empty)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if self.augmentor is not None and self.augmentor.enabled:
            rng = np.random.default_rng()
            sample = self.augmentor(sample, rng)
        return {
            "top_kp": torch.from_numpy(np.ascontiguousarray(sample["top_kp"])),
            "bot_kp": torch.from_numpy(np.ascontiguousarray(sample["bot_kp"])),
            "feat": torch.from_numpy(np.ascontiguousarray(sample["feat"])),
            "mask": torch.from_numpy(np.ascontiguousarray(sample["mask"])),
            "label": sample["label"],
            "clip_id": sample["clip_id"],
        }

    def label_counts(self) -> dict[int, int]:
        """Count windows per class id."""
        counts: dict[int, int] = {}
        for sample in self.samples:
            label = sample["label"]
            counts[label] = counts.get(label, 0) + 1
        return counts


def compute_feature_stats(dataset: "FallWindowDataset") -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std of the engineered ``feat`` stream over valid frames.

    Padded / invalid frames (``mask == 0``) are excluded so the statistics
    reflect real poses only. Returns ``(mean, std)`` arrays of shape
    ``(FEAT_DIM,)``.
    """
    total = np.zeros(FEAT_DIM, dtype=np.float64)
    total_sq = np.zeros(FEAT_DIM, dtype=np.float64)
    count = 0
    for sample in dataset.samples:
        feat = np.asarray(sample["feat"], dtype=np.float64)
        mask = np.asarray(sample["mask"], dtype=np.float64) > 0
        if not mask.any():
            continue
        valid = feat[mask]
        total += valid.sum(axis=0)
        total_sq += (valid ** 2).sum(axis=0)
        count += int(mask.sum())

    if count == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32), np.ones(FEAT_DIM, dtype=np.float32)

    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _resolve_repo_path(path: Path | str, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _load_config(config: Path | str | dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    config_path = Path(config)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def resolve_manifest_for_split(
    config: dict[str, Any],
    split: str,
    repo_root: Path,
    seed: int | None = None,
) -> list[ManifestRow]:
    """Resolve the manifest rows for ``train``, ``val``, or ``test``."""
    data_cfg = config.get("data", {})
    splits_dir = _resolve_repo_path(data_cfg.get("splits_dir", "dataset/splits"), repo_root)
    outputs_dir = _resolve_repo_path(data_cfg.get("outputs_dir", "dataset/outputs"), repo_root)
    features_dir = _resolve_repo_path(data_cfg.get("features_dir", "dataset/features"), repo_root)

    if split == "test":
        manifest_name = data_cfg.get("test_manifest", "test.csv")
        return load_manifest(splits_dir / manifest_name, outputs_dir, features_dir)

    train_manifest_name = data_cfg.get("train_manifest", "train.csv")
    train_rows = load_manifest(splits_dir / train_manifest_name, outputs_dir, features_dir)

    # Prefer an explicit, leak-free val manifest (group-aware split). When
    # ``val_manifest`` is configured and present, train uses all of train.csv
    # and val reads val.csv directly -- no random carve-out from train.
    val_manifest_name = data_cfg.get("val_manifest")
    val_manifest_path = splits_dir / val_manifest_name if val_manifest_name else None
    has_explicit_val = val_manifest_path is not None and val_manifest_path.is_file()

    if split == "train":
        if has_explicit_val:
            return train_rows
        val_ratio = float(data_cfg.get("val_ratio", 0.1))
        if val_ratio <= 0.0:
            return train_rows
        train_part, _ = make_train_val_split(
            train_rows,
            val_ratio=val_ratio,
            seed=seed if seed is not None else int(config.get("training", {}).get("seed", 42)),
        )
        return train_part

    if split == "val":
        if has_explicit_val:
            return load_manifest(val_manifest_path, outputs_dir, features_dir)
        val_ratio = float(data_cfg.get("val_ratio", 0.1))
        if val_ratio <= 0.0:
            raise ValueError("val split requested but data.val_ratio is 0 and no val_manifest is set")
        _, val_part = make_train_val_split(
            train_rows,
            val_ratio=val_ratio,
            seed=seed if seed is not None else int(config.get("training", {}).get("seed", 42)),
        )
        return val_part

    raise ValueError(f"Unknown split {split!r}; expected 'train', 'val', or 'test'")


def build_dataset(
    config: Path | str | dict[str, Any],
    split: str,
    repo_root: Path | None = None,
    skip_missing_features: bool = False,
) -> FallWindowDataset:
    """Construct a :class:`FallWindowDataset` for the given config split."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    cfg = _load_config(config, repo_root)
    data_cfg = cfg.get("data", {})

    manifest = resolve_manifest_for_split(cfg, split, repo_root)

    # Augmentation applies to the train split only; val/test stay deterministic.
    augmentor = None
    augment_cfg = data_cfg.get("augment")
    if split == "train" and isinstance(augment_cfg, dict) and augment_cfg.get("enabled", False):
        augmentor = PoseAugmentor(augment_cfg)

    return FallWindowDataset(
        manifest=manifest,
        window_size=int(data_cfg.get("window_size", DEFAULT_WINDOW_SIZE)),
        stride=int(data_cfg.get("stride", DEFAULT_WINDOW_STRIDE)),
        min_valid_frame_ratio=float(data_cfg.get("min_valid_frame_ratio", 0.5)),
        skip_missing_features=skip_missing_features,
        augmentor=augmentor,
    )


def build_dataloader(
    config: Path | str | dict[str, Any],
    split: str,
    repo_root: Path | None = None,
    shuffle: bool | None = None,
    skip_missing_features: bool = False,
) -> DataLoader:
    """Build a :class:`DataLoader` for ``train``, ``val``, or ``test``."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    cfg = _load_config(config, repo_root)
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    dataset = build_dataset(
        cfg,
        split=split,
        repo_root=repo_root,
        skip_missing_features=skip_missing_features,
    )

    if dataset.skipped_empty_clips:
        print(
            f"{split}: skipped {len(dataset.skipped_empty_clips)} "
            "empty clips (header-only CSVs with no pose frames)"
        )
    if dataset.skipped_clips:
        print(
            f"{split}: skipped {len(dataset.skipped_clips)} clips "
            "with missing feature CSVs"
        )

    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))

    if shuffle is None:
        shuffle = split == "train"

    sampler = None
    if split == "train" and shuffle and data_cfg.get("use_weighted_sampler", False):
        counts = dataset.label_counts()
        weights = [1.0 / counts[sample["label"]] for sample in dataset.samples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def describe_split(
    config: Path | str | dict[str, Any],
    split: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Summarize clip and window counts for a split (for logging)."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    cfg = _load_config(config, repo_root)
    manifest = resolve_manifest_for_split(cfg, split, repo_root)
    return {
        "split": split,
        "clips": manifest_summary(manifest),
    }
