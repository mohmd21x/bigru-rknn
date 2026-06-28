# Fall Detection Tests

This directory contains the realtime demo and ONNX vs RKNN benchmark for the BiGRU fall classifier.

## Step 0: ONNX → RKNN conversion

Export a trained checkpoint to ONNX (if you do not already have one):

```bash
python scripts/export_bigru_onnx.py \
  --checkpoint checkpoints/bigru_hierarchical/best.pt \
  --onnx  weights/bigru_hierarchical.onnx
```

Convert ONNX to RKNN for Rockchip NPU deployment (requires `rknn-toolkit2` on the host):

```bash
python scripts/convert_bigru_onnx_to_rknn.py \
  --onnx  weights/bigru_hierarchical.onnx \
  --rknn  weights/bigru_hierarchical.rknn \
  --platform rk3588 \
  --window-size 64
```

| Flag | Description |
|------|-------------|
| `--platform` | Rockchip SoC target (default: `rk3588`) |
| `--window-size` | Temporal window length `T` (default: inferred from ONNX) |
| `--quantize` | Enable INT8 quantization (requires `--dataset`) |

A pre-converted model is expected at `weights/bigru_hierarchical.rknn` for the default 64-frame checkpoint.

## Deploy via Git (PC → Rockchip)

These files are tracked in Git so you can `git push` on your PC and `git pull` on the board:

| File | Size | Purpose |
|------|------|---------|
| `weights/bigru_hierarchical.onnx` | ~5 MB | ONNX reference + benchmark baseline |
| `weights/bigru_hierarchical.rknn` | ~23 MB | NPU inference on Rockchip |
| `weights/yolov8n-pose.pt` | ~7 MB | YOLO pose for the benchmark pipeline |

**On your PC (first time):**

```bash
git init
git add .
git commit -m "Fall detection: source + deployment weights"
git remote add origin <your-repo-url>
git push -u origin main
```

**On the Rockchip board:**

```bash
git clone <your-repo-url> LSTM-GRU
cd LSTM-GRU
pip install -r requirements.txt
# rknnlite2 is usually pre-installed with the Rockchip SDK / system image
```

Copy test videos separately (too large for Git):

```bash
# From PC
scp Videos/*.mp4 user@board:/path/to/LSTM-GRU/Videos/
```

**Run benchmark on board:**

```bash
python test/benchmark_onnx_vs_rknn.py \
  --onnx  weights/bigru_hierarchical.onnx \
  --rknn  weights/bigru_hierarchical.rknn \
  --video Videos/adl1-cam-2-1-1.mp4 \
  --yolo-weights weights/yolov8n-pose.pt \
  --device cpu
```

Use `--device cpu` for YOLO on the board unless you have a GPU/NPU pose backend configured. RKNN fall inference uses `rknnlite2` automatically.

To re-convert ONNX → RKNN on the board (optional, usually done on PC with `rknn-toolkit2`):

```bash
python scripts/convert_bigru_onnx_to_rknn.py \
  --onnx  weights/bigru_hierarchical.onnx \
  --rknn  weights/bigru_hierarchical.rknn \
  --platform rk3588 \
  --window-size 64
```

## Step 1: Run ONNX vs RKNN benchmark

The benchmark script runs a headless pipeline on one or more videos:

```
Video → YOLO pose → PoseFeatureExtractor (python) → FallFrameBuffer (64 frames)
  → OnnxFallPredictor (CPU)  ─┐
  → RknnFallPredictor (NPU)  ─┴─ compare per window
```

Both predictors receive **identical** numpy tensors from the shared buffer. RKNN runtime selection:

1. `rknnlite2` (`RKNNLite`) on the Rockchip board
2. `rknn-toolkit2` (`RKNN.init_runtime(target=None)`) for PC simulation

```bash
python test/benchmark_onnx_vs_rknn.py \
  --onnx  weights/bigru_hierarchical.onnx \
  --rknn  weights/bigru_hierarchical.rknn \
  --video Videos/adl1-cam-2-1-1.mp4 Videos/shakiba-fall-fast-phone-5-1.mp4 \
  --yolo-weights weights/yolov8n-pose.pt
```

### Useful benchmark flags

| Flag | Default | Description |
|------|---------|-------------|
| `--onnx` | `weights/bigru_hierarchical.onnx` | ONNX fall classifier |
| `--rknn` | `weights/bigru_hierarchical.rknn` | RKNN fall classifier |
| `--video` | *(required)* | One or more input video paths |
| `--yolo-weights` | `weights/yolo8n.pt` | YOLOv8 pose model |
| `--window-size` | `64` | Model temporal window |
| `--clip-frames` | `window-size` | Real frames before predicting |
| `--fall-threshold` | `0.5` | Label threshold on `fall_prob` |
| `--min-valid-frame-ratio` | `0.5` | Skip inference below this valid-frame ratio |
| `--process-fps` | source FPS | Model sampling rate |
| `--infer-every` | `1` | Run models every N processed frames once window is full |
| `--output-dir` | `test/output` | CSV export directory |

## Interpreting benchmark output

For each video the script prints an aligned table and writes a CSV to `test/output/bench_<video>_<timestamp>.csv`.

| Column | Meaning |
|--------|---------|
| `win` | Window index (1-based, after buffer fills) |
| `video_t` | Model timestamp (seconds) at window end |
| `onnx_fall_p` | ONNX fall probability |
| `rknn_fall_p` | RKNN fall probability |
| `delta` | `rknn_fall_p − onnx_fall_p` |
| `onnx_label` / `rknn_label` | `fall` or `not_fall` at `--fall-threshold` |

Summary row:

| Metric | Meaning |
|--------|---------|
| **MAE** | Mean absolute delta between ONNX and RKNN fall probabilities |
| **max\|delta\|** | Worst per-window probability disagreement |
| **pearson** | Linear correlation of ONNX vs RKNN probabilities across windows |
| **label_match** | Fraction of windows where both backends agree on the label |

**What to expect:** RKNN uses FP16 internally by default, so small probability deltas (MAE ≪ 0.05) are normal. Larger gaps or low label-match rates may indicate conversion issues, wrong `--window-size`, or quantization artifacts (if `--quantize` was used). Use the per-window table to inspect where disagreements occur (e.g. near the 0.5 decision boundary).

---

## Realtime fall detection demo

Test **video files**, **webcam**, or **RTSP** streams with:

1. **Pose estimation** — YOLOv8 pose (default) or RTMO-S (`--pose-backend`) for COCO-17 keypoints
2. **Online pose features** (same layout as training)
3. **BiGRU fall classifier** (`checkpoints/bigru_hierarchical/best.pt`)

The OpenCV window shows:

- Pose skeleton (green = not fall, red = fall)
- Label: `FALL` / `NOT FALL`
- Confidence and fall probability
- Pose inference time, fall inference time, FPS
- Sliding window fill bar (64 frames by default)

### Setup

```bash
pip install -r requirements.txt

# Pose weights -> see weights/README.md (YOLO) or rtmo/ (RTMO)
#   YOLO (default): weights/yolo8n.pt
#   RTMO: rtmo/rtmo-s.onnx
```

### Run on a video file

```bash
# YOLO pose (default)
python test/run_realtime.py \
  --source /path/to/video.mp4 \
  --checkpoint checkpoints/bigru_hierarchical/best.pt

# RTMO-S pose
python test/run_realtime.py \
  --source /path/to/video.mp4 \
  --pose-backend rtmo \
  --conf 0.5 \
  --checkpoint checkpoints/bigru_hierarchical/best.pt
```

### Run on RTSP

```bash
python test/run_realtime.py \
  --source "rtsp://user:pass@192.168.1.100:554/stream1" \
  --checkpoint checkpoints/bigru_hierarchical/best.pt
```

### Webcam

```bash
python test/run_realtime.py --source 0
```

### Export labeled output video

```bash
# Default path: test/output/<video_stem>_labeled.mp4
python test/run_realtime.py --source /path/to/video.mp4 --export

# Explicit file
python test/run_realtime.py \
  --source /path/to/video.mp4 \
  --export test/output/demo_labeled.mp4

# Directory (auto-named inside)
python test/run_realtime.py \
  --source /path/to/video.mp4 \
  --export test/output/
```

`--export` runs headless and writes pose skeleton + `FALL` / `NOT FALL` overlay with timing stats.

### Useful demo flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pose-backend` | `yolo` | `yolo` or `rtmo` |
| `--yolo-weights` | `weights/yolo8n.pt` | YOLOv8 pose model (YOLO backend) |
| `--rtmo-onnx` | `rtmo/rtmo-s.onnx` | RTMO ONNX model (RTMO backend) |
| `--checkpoint` | `checkpoints/bigru_hierarchical/best.pt` | Fall BiGRU checkpoint |
| `--config` | `configs/bigru_hierarchical.yaml` | Model config |
| `--window-size` | `64` | Temporal window |
| `--process-fps` | source FPS | Model frames per second of video |
| `--fps` | auto | Override source FPS if metadata is wrong |
| `--infer-every` | `1` | Run fall model every N **processed** frames |
| `--fall-threshold` | `0.5` | BiGRU fall decision threshold on `fall_prob` |
| `--conf` | `0.35` (yolo) / `0.5` (rtmo) | Person detection confidence |
| `--device` | `cuda` if available | Torch device |
| `--export` | off | Save labeled MP4 (optional path/dir) |
| `--no-show` | off | Disable preview window |

Press **q** or **Esc** to quit the display window.

### Demo notes

- The fall model needs **64 processed frames** before the first prediction.
- At **10 fps**, the 64-frame window covers **6.4 seconds** of video; at **30 fps** it covers **~2.1 seconds**.
- FPS is read from the video file; use `--fps 10` if RTSP metadata is wrong.
- Use a **pose** checkpoint (`yolov8n-pose.pt`), not plain `yolov8n.pt` detection-only weights.
- RTMO uses `onnxruntime` with `rtmo/rtmo-s.onnx` (NMS + decode in-graph). Install `onnxruntime-gpu` for CUDA.
- BiGRU was trained on YOLO keypoints; expect small accuracy shifts when switching to RTMO until re-evaluated.
- RTSP latency depends on the camera stream; lower `--infer-every` for smoother labels at higher GPU load.
