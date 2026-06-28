# BiGRU Fall Detection Training Pipeline

End-to-end training stack for clip-level fall detection from COCO-17 pose keypoints. The pipeline extracts engineered pose features with a C++ tool, builds 64-frame sliding windows, trains a hierarchical dual-stream BiGRU model, and evaluates at **clip level** (max fall probability across windows).

## Prerequisites

- **Python 3.10+**
- **CUDA** (optional; training and evaluation fall back to CPU)
- **C++ feature extractor** at `pose_features/build/extract_pose_features_from_csv` (see [Building the feature extractor](#building-the-feature-extractor))
- **Dataset layout** (expected paths relative to the repo root):
  - Raw keypoints: `dataset/outputs/*.csv` (~12,862 clips, variable length)
  - Split manifests: `dataset/splits/train.csv`, `dataset/splits/test.csv`
  - Feature output: `dataset/features/{stem}_features.csv`

> **Note:** Split manifests include a `path` column from an old machine path. The training code resolves clips via `filename` joined to `dataset/outputs/` and `dataset/features/` — ignore the stale `path` value.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, pandas, numpy, scikit-learn, PyYAML, tqdm, matplotlib.

## Quick Start

```bash
# Phase 1 — extract pose features (smoke test on 5 files)
python scripts/extract_features.py --from-manifests --skip-existing --limit 5

# Phase 1 — full extraction for train + test manifests
python scripts/extract_features.py --from-manifests --skip-existing

# Phase 2–3 — train (requires feature CSVs for train/val clips)
python scripts/train.py --config configs/bigru_hierarchical.yaml

# Phase 4 — evaluate best checkpoint on the test split (clip-level)
python scripts/evaluate.py \
  --config configs/bigru_hierarchical.yaml \
  --checkpoint checkpoints/bigru_hierarchical/best.pt
```

For development before all features are extracted, training and evaluation accept `--skip-missing-features` to run on available clips only.

## Pipeline Overview

```text
dataset/outputs/*.csv
        │
        ▼  scripts/extract_features.py  (C++ binary)
dataset/features/*_features.csv
        │
        ▼  preprocess + windowing (64 frames, stride 32)
FallWindowDataset  ──►  HierarchicalDualStreamBiGRU  ──►  Trainer
        │
        ▼  clip-level max fall probability
reports/{run_name}/metrics.json, confusion_matrix.png, ...
```

### Data splits

| Split | Source | Clips (approx.) |
|-------|--------|-----------------|
| Train | `dataset/splits/train.csv` | ~10,290 |
| Val | 10% stratified hold-out from train | ~1,029 |
| Test | `dataset/splits/test.csv` | ~2,572 |

Validation is created automatically at training time (`data.val_ratio` in config). There is no separate val manifest file.

## Phase 1 — Feature Extraction

`scripts/extract_features.py` wraps the pre-built C++ binary and writes one feature CSV per keypoint file:

```text
dataset/outputs/Dataset_1-fall-0001_keypoints.csv
    → dataset/features/Dataset_1-fall-0001_keypoints_features.csv
```

### Common options

| Flag | Description |
|------|-------------|
| `--from-manifests` | Process only filenames listed in `train.csv` and `test.csv` |
| `--skip-existing` | Skip inputs whose output CSV already exists and is non-empty |
| `--limit N` | Process at most N files (smoke tests) |
| `--outputs-dir`, `--features-dir` | Override input/output directories |
| `--error-log` | Log failures (default: `dataset/features/extract_errors.log`) |

Example:

```bash
python scripts/extract_features.py \
  --from-manifests \
  --skip-existing \
  --limit 10
```

### Feature CSV contents

Each feature file has ~70 columns:

- **Metadata:** `video_name`, `frame_index`, `timestamp`, `person_id`, `valid_pose`, etc.
- **Normalized keypoints (34):** `norm_kpt0_x` … `norm_kpt16_y` (hip-centered, height-normalized)
- **Engineered features (30):** distances (10), joint angles (4), torso/hip (9), velocities (7)

Rows are aligned across raw keypoints and features on `(video_name, frame_index, person_id)`.

### Building the feature extractor

If the binary is missing or fails with `libonnxruntime.so.1` / `GLIBC_2.38` errors, rebuild from source (no ONNX Runtime required for CSV extraction):

```bash
cd pose_features
rm -rf build && mkdir build && cd build
cmake ..
cmake --build .
```

Requirements:

- CMake 3.10+, C++17 compiler (g++), OpenCV

The CSV extractor uses a local `PoseEstimator.h` stub and only needs OpenCV. A pre-built binary copied from another machine may require ONNX Runtime 1.19.2 and a newer glibc — rebuilding locally avoids both.

If you still use an older ONNX-linked binary, install ONNX Runtime 1.19.2 to `~/.local/onnxruntime-linux-x64-gpu-1.19.2` and pass `--extractor-lib-dir ~/.local/onnxruntime-linux-x64-gpu-1.19.2/lib`.

## Phase 2–3 — Training

```bash
python scripts/train.py --config configs/bigru_hierarchical.yaml
```

Optional flags:

- `--device cuda:0` — force a specific device
- `--skip-missing-features` — skip clips without feature CSVs (smoke runs)

### What happens during training

1. **Load manifests** — resolve `filename` → `dataset/outputs/` and `dataset/features/`
2. **Train/val split** — stratified 10% hold-out from the train manifest
3. **Windowing** — 64-frame windows, stride 32; short clips are zero-padded at the end
4. **Filtering** — windows with fewer than 50% valid frames (`valid_pose`) are dropped
5. **Model** — hierarchical dual-stream BiGRU (see [Architecture](#architecture))
6. **Optimization** — AdamW, weighted cross-entropy, ReduceLROnPlateau on val loss, gradient clip 1.0
7. **Checkpointing** — best model by validation **fall F1**; early stopping patience 10

### Training outputs

Written to `checkpoints/{run_name}/` (default `checkpoints/bigru_hierarchical/`):

| File | Description |
|------|-------------|
| `best.pt` | Best checkpoint (highest val fall F1) |
| `last.pt` | Most recent epoch |
| `config.yaml` | Copy of the training config |
| `history.csv` | Per-epoch train/val loss, accuracy, F1, learning rate |

## Phase 4 — Evaluation

```bash
python scripts/evaluate.py \
  --config configs/bigru_hierarchical.yaml \
  --checkpoint checkpoints/bigru_hierarchical/best.pt
```

Evaluation runs on the **test split** at **clip level**:

1. Run window-level inference on all test windows
2. Group predictions by `clip_id`
3. **Clip prediction** = class with max fall probability across windows (`clip_max` aggregation)
4. Tie-break: window with highest overall confidence wins

### Evaluation outputs

Written to `reports/{run_name}/` (default `reports/bigru_hierarchical/`):

| File | Description |
|------|-------------|
| `metrics.json` | Accuracy, macro/per-class precision, recall, F1 |
| `confusion_matrix.png` | 2×2 fall vs not_fall plot |
| `classification_report.txt` | sklearn-style text report |

## Export BiGRU for Rockchip (ONNX → RKNN)

Export the trained classifier to ONNX, then convert to RKNN for Rockchip NPU deployment.
Conversion requires **RKNN-Toolkit2** (`from rknn.api import RKNN`) on your host or board.
See `INTEGRATION.md` for full C++ integration and input tensor layout.

### Step 1 — Checkpoint → ONNX

```bash
python scripts/export_bigru_onnx.py \
  --checkpoint checkpoints/bigru_hierarchical/best.pt \
  --onnx checkpoints/bigru_hierarchical/bigru.onnx
```

For the 7.5-fps model (`window_size: 16`), point `--checkpoint` at
`checkpoints/bigru_hierarchical_7fps/best.pt` and choose a matching `--onnx` path.

### Step 2 — ONNX → RKNN

```bash
python scripts/convert_bigru_onnx_to_rknn.py \
  --onnx checkpoints/bigru_hierarchical/bigru.onnx \
  --rknn checkpoints/bigru_hierarchical/bigru.rknn \
  --platform rk3588
```

Common options:

| Flag | Description |
|------|-------------|
| `--onnx` | Input ONNX model path (required) |
| `--rknn` | Output RKNN model path (required) |
| `--platform` | Rockchip SoC, e.g. `rk3588`, `rk3568`, `rv1106` (default: `rk3588`) |
| `--window-size` | Sliding-window length `T` (64 or 16); inferred from ONNX or `--checkpoint` if omitted |
| `--checkpoint` | Optional `.pt` file to read `window_size` when ONNX has dynamic `T` |
| `--quantize` | Enable INT8 quantization (requires `--dataset`) |
| `--dataset` | Calibration dataset file for INT8 quantization |
| `--verbose` | Verbose RKNN build logs |

Example with explicit window size and 7.5-fps checkpoint:

```bash
python scripts/convert_bigru_onnx_to_rknn.py \
  --onnx checkpoints/bigru_hierarchical_7fps/bigru.onnx \
  --rknn checkpoints/bigru_hierarchical_7fps/bigru.rknn \
  --checkpoint checkpoints/bigru_hierarchical_7fps/best.pt \
  --platform rk3588
```

Start with FP32 (`do_quantization=False`, the default). If GRU conversion fails on the
NPU, keep BiGRU on CPU and run only pose on the NPU — see `INTEGRATION.md` §8.3.

## Configuration

Default config: `configs/bigru_hierarchical.yaml`

```yaml
run_name: bigru_hierarchical

model:
  name: bigru_hierarchical
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3

data:
  window_size: 64
  stride: 32
  val_ratio: 0.1
  min_valid_frame_ratio: 0.5
  use_weighted_sampler: true   # address train class imbalance

training:
  batch_size: 64
  epochs: 50
  lr: 1.0e-4
  class_weights: auto          # inverse-frequency from train manifest
  early_stopping_patience: 10

evaluation:
  aggregation: clip_max
```

Key paths (`outputs_dir`, `features_dir`, `splits_dir`, `checkpoint_dir`, `reports_dir`) are relative to the repo root unless absolute.

## Architecture

`HierarchicalDualStreamBiGRU` (`src/models/bigru_hierarchical.py`) uses three parallel bidirectional GRU branches:

| Branch | Input | Dims |
|--------|-------|------|
| Top keypoints | COCO joints 0–10 (head, arms) | 22 |
| Bottom keypoints | COCO joints 11–16 (hips, legs) | 12 |
| Engineered features | distances, angles, torso/hip, velocity | 30 |

Each branch: 2-layer BiGRU (hidden 128, dropout 0.3) → masked temporal attention pooling → 256-d vector. The three vectors are concatenated (768-d) and passed through a linear classifier (2 classes: `not_fall`, `fall`).

## Tensor Shapes Reference

| Tensor / field | Shape | Dtype | Description |
|----------------|-------|-------|-------------|
| `top_kp` | `(B, 64, 22)` | float32 | Top-body normalized keypoints |
| `bot_kp` | `(B, 64, 12)` | float32 | Bottom-body normalized keypoints |
| `feat` | `(B, 64, 30)` | float32 | Engineered pose features |
| `mask` | `(B, 64)` | float32 | 1 = valid frame, 0 = padded/invalid |
| `label` | `(B,)` | int64 | 0 = not_fall, 1 = fall |
| `clip_id` | list of str | — | Source clip filename |
| Model output (logits) | `(B, 2)` | float32 | Unnormalized class scores |

`B` = batch size (default 64). Window length `T` = 64 is configurable via `data.window_size`.

### Per-stream column mapping

Defined in `src/constants.py`:

- **Top keypoints:** `norm_kpt0_x/y` … `norm_kpt10_x/y`
- **Bottom keypoints:** `norm_kpt11_x/y` … `norm_kpt16_x/y`
- **Engineered:** `dist_*`, `angle_*`, `torso_*`, `hip_*`, `bbox_aspect_ratio`, `vel_*`

NaN values in feature CSVs are replaced with 0. The validity mask comes from the `valid_pose` column.

## Project Layout

```text
LSTM-GRU/
├── README.md
├── requirements.txt
├── configs/
│   └── bigru_hierarchical.yaml
├── scripts/
│   ├── extract_features.py           # Phase 1: C++ feature export
│   ├── train.py                      # Phase 2–3: training CLI
│   ├── evaluate.py                   # Phase 4: test evaluation CLI
│   ├── export_bigru_onnx.py          # Checkpoint → ONNX (Rockchip deploy)
│   └── convert_bigru_onnx_to_rknn.py # ONNX → RKNN (Rockchip deploy)
├── src/
│   ├── constants.py               # COCO splits, column lists, dims
│   ├── data/
│   │   ├── preprocess.py          # CSV loading, NaN/mask handling
│   │   ├── windowing.py           # 64-frame sliding windows
│   │   ├── dataset.py             # FallWindowDataset + DataLoader
│   │   └── postprocess.py         # Clip-level aggregation
│   ├── models/
│   │   ├── base.py                # BaseFallModel ABC + registry
│   │   └── bigru_hierarchical.py  # v1 model
│   ├── training/
│   │   └── trainer.py             # Generic Trainer
│   └── evaluation/
│       └── metrics.py             # Metrics + report artifacts
├── dataset/
│   ├── outputs/                   # Raw keypoint CSVs
│   ├── features/                  # Extracted feature CSVs
│   └── splits/                    # train.csv, test.csv
└── pose_features/                 # C++ feature extractor
```

## Adding a New Model

The training and evaluation code is model-agnostic. To plug in a new architecture:

### 1. Subclass `BaseFallModel`

Create `src/models/your_model.py`:

```python
from typing import Any
from torch import Tensor
from src.models.base import BaseFallModel, register_model

@register_model("your_model")
class YourFallModel(BaseFallModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        # build layers from model_cfg

    @property
    def model_name(self) -> str:
        return getattr(self, "_registry_name", "your_model")

    def forward(self, batch: dict[str, Any]) -> Tensor:
        # consume batch["top_kp"], batch["bot_kp"], batch["feat"], batch["mask"]
        # return logits of shape (batch_size, num_classes)
        ...
```

The batch dict matches `FallWindowDataset` output. Use `batch["mask"]` when pooling over time so padded frames are ignored.

### 2. Register the model

The `@register_model("your_model")` decorator adds the class to `MODEL_REGISTRY`. `build_model(name, config)` in `src/models/base.py` instantiates it by name.

If your module is not imported elsewhere, add an import in `_ensure_builtin_models_loaded()` in `base.py` (same pattern as `bigru_hierarchical`).

### 3. Add a config

Copy `configs/bigru_hierarchical.yaml` and set:

```yaml
run_name: your_model_run
model:
  name: your_model
  # your hyperparameters
```

### 4. Train and evaluate

```bash
python scripts/train.py --config configs/your_model.yaml
python scripts/evaluate.py --config configs/your_model.yaml --checkpoint checkpoints/your_model_run/best.pt
```

No changes to the data pipeline, trainer, or evaluation scripts are required as long as the model accepts the standard batch dict and returns `(B, num_classes)` logits.

### Extension hooks (future)

- **Data augmentation** — add transforms in `FallWindowDataset.__getitem__` in `src/data/dataset.py`
- **New aggregation strategies** — implement in `src/data/postprocess.py` and set `evaluation.aggregation` in config
- **Additional input streams** — extend `preprocess.py` and `constants.py`; update or subclass the dataset if tensor keys change

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Extractor binary not found / `libonnxruntime.so.1` / `GLIBC_2.38` | Rebuild: `cd pose_features && rm -rf build && mkdir build && cd build && cmake .. && cmake --build .` |
| `libonnxruntime.so` (old pre-built binary only) | Install ORT 1.19.2 to `~/.local/onnxruntime-linux-x64-gpu-1.19.2` or rebuild locally (preferred) |
| Train fails: missing feature CSVs | Run `extract_features.py --from-manifests` or use `--skip-missing-features` |
| No valid windows built | Check feature files exist; lower `min_valid_frame_ratio` in config |
| Stale manifest paths | Use `filename` column only; paths are resolved under `dataset/outputs/` |

Failed extractions are logged to `dataset/features/extract_errors.log`.
