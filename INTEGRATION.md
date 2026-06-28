# Fall Detection — C++ / Rockchip NPU Integration Guide

> **Model family:** Hierarchical Dual-Stream BiGRU (`bigru_hierarchical`)  
> **Reference Python entry-point:** `test/run_realtime.py`  
> **Target platform:** Rockchip SoC with NPU (RK3566 / RK3568 / RK3588 / RV1106 …)  
> **Toolchain:** RKNN-Toolkit2 (host) + RKNN Runtime C API (device)

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Complete Pipeline — Stage by Stage](#2-complete-pipeline--stage-by-stage)
3. [Frame Feed Mechanism](#3-frame-feed-mechanism)
4. [Keypoint & Feature Extraction (C++ Logic)](#4-keypoint--feature-extraction-c-logic)
5. [Kalman Filter — Design, State, and Flow](#5-kalman-filter--design-state-and-flow)
6. [Frame Buffer — Rolling Window Management](#6-frame-buffer--rolling-window-management)
7. [BiGRU Model — Architecture, Inputs, and Outputs](#7-bigru-model--architecture-inputs-and-outputs)
8. [Exporting Models for Rockchip NPU](#8-exporting-models-for-rockchip-npu)
9. [C++ Integration Skeleton](#9-c-integration-skeleton)
10. [Tensor Layout Reference](#10-tensor-layout-reference)
11. [Configuration Reference](#11-configuration-reference)
12. [Accuracy-Preserving Checklist](#12-accuracy-preserving-checklist)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VIDEO / RTSP SOURCE                         │
│                      (OpenCV VideoCapture)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │  raw BGR frames @ source_fps
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  VideoFrameSampler (frame gate)                     │
│  Drops frames so the model only sees process_fps frames/second.     │
│  Default: process_fps = source_fps.                                 │
│  7.5-fps model: process_fps = 7.5 (1 of every 4 frames at 30fps)   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  selected BGR frame + timestamp (s)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   YOLOv8n-Pose (NPU / GPU / CPU)                    │
│  Input : BGR frame  (any resolution — model letterboxes internally) │
│  Output: person detections → keypoints (17, 3) [x, y, conf]        │
│  Pick  : highest-confidence person above --conf threshold (0.35)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │  keypoints (17, 3) or NULL (no person)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Feature Backend (choose one)                       │
│                                                                     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  cpp backend  │  │python backend│  │    kalman backend        │ │
│  │ (subprocess)  │  │ (PoseFeature │  │  KalmanPoseInterpolator  │ │
│  │ exact training│  │  Extractor)  │  │  7fps YOLO → 30fps KF    │ │
│  │  match ✓      │  │ Python port  │  │  → PoseFeatureExtractor  │ │
│  └───────┬───────┘  └──────┬───────┘  └───────────┬──────────────┘ │
│          └─────────────────┴──────────────────────┘                │
└────────────────────────────┬────────────────────────────────────────┘
                             │  per-frame dict:
                             │  { top_kp[22], bot_kp[12], feat[29], mask }
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FallFrameBuffer                                 │
│  Circular deque of clip_frames (default: window_size) dicts.       │
│  When full → pad to window_size (if clip_frames < window_size)     │
│  → assemble batch tensors (1, T, D).                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │  batch: top_kp(1,T,22) bot_kp(1,T,12)
                             │         feat(1,T,29)   mask(1,T)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Hierarchical Dual-Stream BiGRU (FallPredictor)             │
│                                                                     │
│  gru_top  (22 → BiGRU 128×2) → TemporalAttentionPooling → 256-d    │
│  gru_bot  (12 → BiGRU 128×2) → TemporalAttentionPooling → 256-d    │
│  gru_feat (29 → BiGRU 128×2) → TemporalAttentionPooling → 256-d    │
│                                                                     │
│  concat(256, 256, 256) = 768-d → Dropout → Linear(768, 2)          │
│  → softmax → [P(not_fall), P(fall)]                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │  label, fall_prob, confidence
                             ▼
                      ALARM / HUD / MQTT
```

---

## 2. Complete Pipeline — Stage by Stage

### Stage 0 — Capture

```
source_fps = cap.get(CAP_PROP_FPS)   // e.g. 30.0 for RTSP
```

For RTSP use `--fps 25` (or whatever the camera advertises) to override
unreliable metadata.

### Stage 1 — Frame Gate (`VideoFrameSampler`)

```
process_fps = source_fps   // or 7.5 for the 7-fps model

sample_interval = 1.0 / process_fps
next_sample_time = 0.0

for each decoded frame at index i:
    video_time = i / source_fps
    frame_mid  = video_time + 0.5 / source_fps
    if frame_mid >= next_sample_time:
        timestamp = next_sample_time
        next_sample_time += sample_interval
        → feed frame to YOLO
```

This produces exactly `process_fps` model frames per second of video, even
if YOLO is slower and frames arrive in bursts.

### Stage 2 — YOLOv8 Pose Inference

Run YOLOv8n-pose on the selected BGR frame.

Output per frame: `N_persons × (5_box + 17×3_keypoints)` but we take only
the single highest-confidence person above `conf_threshold = 0.35`.

Result keypoint array shape: `(17, 3)` — `[x_pixel, y_pixel, confidence]`
in COCO-17 order (see §10).

If no person detected → `keypoints = NULL` → feature extractor is **reset**
and a zero-padded frame with `mask=0` is added (or buffer is flushed,
depending on backend).

### Stage 3 — Feature Extraction

Three equivalent backends; **cpp** is the reference and matches training exactly.

| Backend | How | When to use |
|---------|-----|-------------|
| `cpp` | Subprocess call to `pose_features/build/extract_pose_features_from_csv` | Training-matched, highest accuracy |
| `python` | `PoseFeatureExtractor` in `src/inference/pose_features.py` | When C++ binary not available |
| `kalman` | `KalmanPoseInterpolator` → internally calls python extractor | 7-fps camera with interpolation |

**Output of all backends** (per frame dict):

```
{
  "top_kp" : float32[22]   // hip-centered + height-normalized, joints 0–10
  "bot_kp" : float32[12]   // same, joints 11–16
  "feat"   : float32[29]   // engineered features (distances, angles, velocities)
  "mask"   : float          // 1.0 = valid pose, 0.0 = no/bad detection
}
```

See §4 for exact feature definitions.

### Stage 4 — Frame Buffer

A deque of `clip_frames` dicts (default: same as `window_size`).

`is_ready` is `True` when `len(buffer) == clip_frames`.

`as_batch()` stacks all dicts and zero-pads the tail to `window_size`:

```
top_kp  shape: (1, window_size, 22)
bot_kp  shape: (1, window_size, 12)
feat    shape: (1, window_size, 29)
mask    shape: (1, window_size)
```

### Stage 5 — BiGRU Inference

```python
if buffer.is_ready and frame_index % infer_every == 0:
    batch = buffer.as_batch()
    valid_ratio = sum(mask) / window_size
    if valid_ratio >= min_valid_frame_ratio:  // default 0.5
        logits = model(batch)                  // (1, 2)
        probs  = softmax(logits)               // [P_not_fall, P_fall]
        label  = argmax(probs)                 // 0 or 1
```

Returns:
```
{
  "label"      : "fall" | "not_fall"
  "label_id"   : 1      | 0
  "fall_prob"  : float  // P(fall)
  "confidence" : float  // P(predicted_class)
  "ready"      : bool
}
```

---

## 3. Frame Feed Mechanism

### 30-fps Model (`bigru_hierarchical`)

```
window_size  = 64 frames  ≈ 2.13 s at 30 fps
window_stride = 32 frames (training; inference: infer_every)
```

At 30 fps, the buffer is primed after 64 frames (≈ 2.1 s). After that, every
new frame shifts the window by 1 and a new inference fires every `infer_every`
frames (default 1 = every frame).

### 7.5-fps Model (`bigru_hierarchical_7fps`)

```
window_size   = 16 frames  ≈ 2.13 s at 7.5 fps
clip_frames   = 14 (pass --clip-frames 14)
process_fps   = 7.5 (pass --process-fps 7.5)
```

Use `--clip-frames 14` to match the training distribution (original training
clips were ~56 frames at 30 fps → ~14 frames at 7.5 fps, zero-padded to 16).

### Reset Conditions

The buffer must be **fully reset** whenever track continuity breaks:

| Event | Action |
|-------|--------|
| YOLO detects no person (non-kalman backends) | `feature_extractor.reset()` + `frame_buffer.reset()` |
| Kalman: consecutive misses > `miss_tolerance` | `kalman_interpolator.push()` returns `reset_occurred=True` → `frame_buffer.reset()` |
| Source change (new video / RTSP reconnect) | `feature_extractor.reset()` + `frame_buffer.reset()` |

---

## 4. Keypoint & Feature Extraction (C++ Logic)

### 4.1 Normalized Pose Coordinates

For each of the 17 COCO keypoints, compute hip-centered, height-normalized
coordinates:

```
hip_center     = (kpt[11] + kpt[12]) / 2    // average left+right hip (if conf ≥ 0.3)
bbox_height    = max(y_i) - min(y_i)        // over keypoints with conf ≥ 0.3
norm_kpt[i].x  = (kpt[i].x - hip_center.x) / bbox_height
norm_kpt[i].y  = (kpt[i].y - hip_center.y) / bbox_height
```

These 34 values (17 × x/y) are split into:

```
top_kp[22] = norm_kpt[0..10], interleaved x/y   // joints 0–10: head + arms
bot_kp[12] = norm_kpt[11..16], interleaved x/y  // joints 11–16: hips + legs
```

### 4.2 Engineered Features (feat[29])

The 29-element `feat` vector is assembled in this exact order:

#### Distances × 10 (all normalized by `bbox_height`)
| Index | Name | Formula |
|-------|------|---------|
| 0 | `dist_shoulder_width` | `‖kpt[5] − kpt[6]‖ / h` |
| 1 | `dist_hip_width` | `‖kpt[11] − kpt[12]‖ / h` |
| 2 | `dist_nose_to_hip` | `‖kpt[0] − hip_center‖ / h` |
| 3 | `dist_left_thigh` | `‖kpt[11] − kpt[13]‖ / h` |
| 4 | `dist_right_thigh` | `‖kpt[12] − kpt[14]‖ / h` |
| 5 | `dist_left_shin` | `‖kpt[13] − kpt[15]‖ / h` |
| 6 | `dist_right_shin` | `‖kpt[14] − kpt[16]‖ / h` |
| 7 | `dist_hand_to_hand` | `‖kpt[9] − kpt[10]‖ / h` |
| 8 | `dist_left_hand_to_hip` | `‖kpt[9] − hip_center‖ / h` |
| 9 | `dist_right_hand_to_hip` | `‖kpt[10] − hip_center‖ / h` |

#### Joint Angles × 4 (degrees, vectors meeting at joint)
| Index | Name | Joints |
|-------|------|--------|
| 10 | `angle_left_hip` | knee(13) → hip(11) → shoulder(5) |
| 11 | `angle_right_hip` | knee(14) → hip(12) → shoulder(6) |
| 12 | `angle_left_knee` | hip(11) → knee(13) → ankle(15) |
| 13 | `angle_right_knee` | hip(12) → knee(14) → ankle(16) |

`angle_at_joint(A, B, C) = arccos(dot(A−B, C−B) / (‖A−B‖ · ‖C−B‖))`

#### Torso / Hip × 8
| Index | Name | Definition |
|-------|------|-----------|
| 14 | `torso_angle` | `arctan2(|shoulder.x − hip.x|, |shoulder.y − hip.y|)` in degrees; averaged over left and right side. 0° = upright, ~90° = horizontal |
| 15 | `torso_angle_std` | std-dev of `torso_angle` over rolling window of 10 frames |
| 16 | `angle_change` | `torso_angle[t] − torso_angle[t-1]` |
| 17 | `torso_angular_velocity` | `angle_change / dt` (deg/s) |
| 18 | `hip_height` | `hip_center.y / bbox_height` |
| 19 | `min_hip_height_over_window` | min of `hip_height` over rolling 5-frame window |
| 20 | `hip_height_change` | `hip_height[t] − hip_height[t-1]` |
| 21 | `bbox_aspect_ratio` | `keypoint_span_x / bbox_height` |

#### Velocity × 7 (all normalized by `bbox_height`)
| Index | Name | Definition |
|-------|------|-----------|
| 22 | `vel_hip_vx` | `(hip_center.x[t] − hip_center.x[t-1]) / (dt · h)` |
| 23 | `vel_hip_vy` | `(hip_center.y[t] − hip_center.y[t-1]) / (dt · h)` |
| 24 | `rolling_mean_vertical_velocity` | running mean of `vel_hip_vy` over 5 frames |
| 25 | `acceleration` | `(rolling_mean_vy[t] − rolling_mean_vy[t-1]) / dt` |
| 26 | `vel_hip_speed` | `hypot(vel_hip_vx, vel_hip_vy)` |
| 27 | `vel_max_wrist_speed` | `max(‖Δkpt[9]‖, ‖Δkpt[10]‖) / (dt · h)` |
| 28 | `vel_max_ankle_speed` | `max(‖Δkpt[15]‖, ‖Δkpt[16]‖) / (dt · h)` |

> **Important:** All velocity / temporal features require the **previous valid
> frame**. When the track is reset (person reappears or buffer cleared), the
> first new frame will have all velocity features = 0 and `mask = 1`. This is
> exactly what the training data contained for clip boundaries — do not
> attempt to interpolate across a reset boundary.

### 4.3 Validity (`mask`)

```
valid_pose = (hip_center != NULL) AND (bbox_height >= 1.0 px)
mask       = 1.0 if valid_pose else 0.0
```

Only frames where `mask = 1.0` contribute to the attention weighting inside
the BiGRU. Frames where `mask = 0.0` are zero-padded pass-throughs.

---

## 5. Kalman Filter — Design, State, and Flow

The Kalman backend is used when YOLO runs at a lower rate (~7 fps) and the
model expects 30 fps input. It upsamples sparse detections without future-frame
look-ahead (causal / online-safe).

### 5.1 KalmanFilter4D

One independent filter per COCO keypoint (17 total).

**State vector (4D):** `[x, y, vx, vy]`

**Constant-velocity motion model:**
```
x(t+dt) = x(t) + vx(t) · dt
y(t+dt) = y(t) + vy(t) · dt
vx(t+dt) = vx(t)             (velocity assumed constant)
vy(t+dt) = vy(t)
```

**Matrices:**
```
F (transition):
  [1, 0, dt, 0 ]
  [0, 1, 0,  dt]
  [0, 0, 1,  0 ]
  [0, 0, 0,  1 ]

H (observation, x/y only):
  [1, 0, 0, 0]
  [0, 1, 0, 0]

Q (process noise) ∝ process_noise² · I₄   (σ_Q = 20.0 default)
R (measurement noise) = (meas_noise_base / conf)² · I₂  (σ_R_base = 10.0)
```

**Measurement noise scales inversely with keypoint confidence** — a high-conf
detection is trusted more than a low-conf one.

**API:**
```cpp
kf.initialize(x0, y0)              // first measurement
kf.predict(dt)                     // advance state
kf.update(xy, conf)                // fuse YOLO measurement
[x, y] = kf.position()            // read estimated position
kf.reset()                         // clear state (track lost)
```

### 5.2 Streaming Logic (`KalmanPoseInterpolator.push`)

Called once per YOLO frame (at ~7 fps). Returns **a list of 30-fps output
feature dicts** covering the elapsed time since the last call.

```
Input:  keypoints (17,3) or NULL, timestamp (s)
Output: KalmanPushResult { frames[], reset_occurred }
```

**Step-by-step inside `push()`:**

```
1. MISSING-PERSON GATE
   if keypoints != NULL AND valid_keypoints ≥ min_valid_keypoints (3):
       has_valid = True
       miss_count = 0
   else:
       has_valid = False
       miss_count += 1

2. EXIT THRESHOLD CHECK
   if miss_count == miss_tolerance + 1 (default: 3rd consecutive miss):
       _hard_reset()                // wipe all 17 KF states
       feature_extractor.reset()
       return KalmanPushResult(frames=[], reset_occurred=True)
       // caller must call frame_buffer.reset()

3. STILL ABSENT
   if miss_count > miss_tolerance + 1:
       return KalmanPushResult(frames=[], reset_occurred=False)

4. COMPUTE OUTPUT TIMESTAMPS
   n_out = round((timestamp - last_push_time) * target_fps)   // e.g. 4
   output_times = [last_t + dt_out, last_t + 2*dt_out, ..., timestamp]

5. FOR EACH OUTPUT TIMESTAMP t_out:
   a) kf.predict(dt) for all 17 keypoints
   b) If this is the LAST output time AND has_valid:
         kf.update(xy, conf) for valid keypoints (conf ≥ min_kpt_conf=0.3)
         // only the most recent output gets the real YOLO measurement
   c) Build output_kp (17,3):
         output_kp[k].xy  = kf.position(k)
         output_kp[k].conf = last_valid_conf[k] * exp(-gap / conf_decay_sec)
         // zero out if gap > max_keypoint_gap_sec (0.5 s)
   d) frame_dict = pose_feature_extractor.update(output_kp, t_out)
   e) append frame_dict to frames list

6. return KalmanPushResult(frames=frames, reset_occurred=kf_reset_occurred)
```

**Timeline example (7 fps YOLO → 30 fps output):**

```
YOLO detections:     t=0    t=0.143        t=0.286
KF output frames:    t=0  t=0.033  t=0.067  t=0.100  t=0.133  t=0.143 ...
                          ◄──── 4-5 interpolated frames ────►
```

### 5.3 Person Presence State Machine

```
TRACKING  (miss=0)  ──miss──►  OCCLUDED_1 (miss=1)
                                    │
                               ──miss──►  OCCLUDED_2 (miss=2, still within tolerance)
                                               │
                                          ──miss──►  EXIT (miss=3)
                                                         │
                                                   _hard_reset()
                                                   reset_occurred=True
                                                         │
                                                    ABSENT (keep returning [])
                                                         │
                                                   ──detect──►  re-initialize KF
                                                               kf_reset_occurred=True
```

At `miss_tolerance=2` (default): 2 consecutive missed YOLO frames are treated
as temporary occlusion (the KF keeps predicting). The 3rd consecutive miss
triggers full reset. At 7 fps this corresponds to ~0.29 s of occlusion tolerance.

---

## 6. Frame Buffer — Rolling Window Management

```cpp
// C++ equivalent of FallFrameBuffer
struct FrameFeatures {
    float top_kp[22];
    float bot_kp[12];
    float feat[29];
    float mask;        // 1.0 or 0.0
};

class FallFrameBuffer {
    std::deque<FrameFeatures> _frames;
    int window_size;   // model sequence length T
    int clip_frames;   // real frames to accumulate (≤ window_size)
public:
    void reset() { _frames.clear(); }
    void add(const FrameFeatures& f) {
        _frames.push_back(f);
        if (_frames.size() > clip_frames) _frames.pop_front();
    }
    bool is_ready() { return _frames.size() == clip_frames; }
    float valid_ratio() {
        float sum = 0;
        for (auto& f : _frames) sum += f.mask;
        return sum / window_size;   // denominator is window_size, not clip_frames
    }
    // assemble_batch() — stack + zero-pad — described in §7
};
```

**Zero-padding rule:** If `clip_frames < window_size`, the batch tensor is
zero-padded at the **end** (tail) to `window_size`. The mask for padded frames
is 0. This exactly mirrors the training-time windowing behavior where short
clips are end-padded.

```
Real frames:  [f0, f1, ..., f13]          (clip_frames=14)
Model input:  [f0, f1, ..., f13, 0, 0]    (window_size=16, 2 zero frames appended)
```

---

## 7. BiGRU Model — Architecture, Inputs, and Outputs

### 7.1 Model Architecture

```
top_kp  (B, T, 22) ──► BiGRU(22→128, 2 layers, bidirectional) ──► (B, T, 256)
                                                                         │
                                                           TemporalAttentionPool
                                                                         │
                                                                   top_repr (B, 256)
                                                                         │
bot_kp  (B, T, 12) ──► BiGRU(12→128, 2 layers, bidirectional) ──► AttPool ──► bot_repr (B, 256)

feat    (B, T, 29) ──► BiGRU(29→128, 2 layers, bidirectional) ──► AttPool ──► feat_repr (B, 256)

concat([top_repr, bot_repr, feat_repr]) ──► (B, 768)
                                                │
                                           Dropout(0.3)
                                                │
                                        Linear(768 → 2)
                                                │
                                           logits (B, 2)
                                                │
                                          softmax(dim=-1)
                                                │
                             [P(not_fall), P(fall)]  in [0, 1]
```

### 7.2 TemporalAttentionPooling

The attention pooling layer computes a **weighted sum** over time, using the
`mask` to suppress invalid frames:

```
scores   = Linear(input_dim=256, out=1)(sequence)  // (B, T, 1)
scores   = scores.masked_fill(mask==0, -∞)          // mute padded/invalid frames
weights  = softmax(scores, dim=T)                   // (B, T)
weights  = weights * mask                           // re-zero padded
weights  = weights / sum(weights)                   // re-normalize
output   = sum(sequence * weights, dim=T)           // (B, 256)
```

This means: **frames where the person is absent or occluded (mask=0) contribute
zero** to the final representation. Only valid-pose frames vote.

### 7.3 Input Tensor Specification

| Tensor | Shape | dtype | Range / Notes |
|--------|-------|-------|---------------|
| `top_kp` | `(1, T, 22)` | float32 | Hip-centered, height-normalized. Joints 0–10. |
| `bot_kp` | `(1, T, 12)` | float32 | Same normalization. Joints 11–16. |
| `feat` | `(1, T, 29)` | float32 | Mixed units — see §4.2. NaN → 0. |
| `mask` | `(1, T)` | float32 | Exactly 0.0 or 1.0 per frame. |

`T = window_size`: **64** for 30-fps model, **16** for 7.5-fps model.

### 7.4 Output Specification

```
logits  : float32 (1, 2)   — raw pre-softmax scores
probs   : float32 (1, 2)   — after softmax
label   : int               — argmax(probs): 0=not_fall, 1=fall
fall_prob: float            — probs[1]
```

**Threshold logic:**
```
if valid_ratio < min_valid_frame_ratio (0.5):
    → skip inference, return not_fall with confidence=0.0
else:
    → run model, label = argmax(softmax(logits))
```

### 7.5 v2 Model Differences (`bigru_hierarchical_v2`)

- `hidden_dim = 96` (smaller, faster)
- `bidirectional = True`, `use_layernorm = True`
- **z-score normalization** on `feat` stream: `feat = (feat - feat_mean) / feat_std`
  where `feat_mean` and `feat_std` are saved as model buffers in the checkpoint.
  Apply **before** feeding to the model; these stats are loaded from the `.pt` file.

---

## 8. Exporting Models for Rockchip NPU

### 8.1 Export YOLOv8n-Pose to ONNX → RKNN

```bash
# 1. Export from ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('weights/yolov8n-pose.pt')
model.export(format='onnx', imgsz=640, opset=12)
"

# 2. Convert to RKNN (on host with rknn-toolkit2 installed)
python - <<'EOF'
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform='rk3588')  # change as needed
rknn.load_onnx(model='weights/yolov8n-pose.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('weights/yolov8n-pose.rknn')
rknn.release()
EOF
```

### 8.2 Export BiGRU Classifier to ONNX

The BiGRU takes **three separate inputs** plus the mask. Export with fixed
`batch=1`, dynamic `T`:

```python
# export_bigru.py
import torch
from pathlib import Path
from src.models.base import build_model
import yaml

checkpoint_path = "checkpoints/bigru_hierarchical/best.pt"
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = ckpt["config"]
model_name = ckpt.get("model_name", config["model"]["name"])

model = build_model(model_name, config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

T = config["data"]["window_size"]  # 64 or 16

# Dummy inputs
top_kp = torch.zeros(1, T, 22)
bot_kp = torch.zeros(1, T, 12)
feat   = torch.zeros(1, T, 29)
mask   = torch.ones(1, T)

class BiGRUWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, top_kp, bot_kp, feat, mask):
        return self.m({"top_kp": top_kp, "bot_kp": bot_kp,
                       "feat": feat, "mask": mask})

wrapper = BiGRUWrapper(model)

torch.onnx.export(
    wrapper,
    (top_kp, bot_kp, feat, mask),
    "checkpoints/bigru_hierarchical/bigru.onnx",
    input_names=["top_kp", "bot_kp", "feat", "mask"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={
        "top_kp": {0: "batch"},
        "bot_kp": {0: "batch"},
        "feat":   {0: "batch"},
        "mask":   {0: "batch"},
    },
)
print("Exported bigru.onnx")
```

### 8.3 Convert BiGRU ONNX → RKNN

```python
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588', quantized_dtype='asymmetric_quantized-8')
rknn.load_onnx(
    model='checkpoints/bigru_hierarchical/bigru.onnx',
    inputs=['top_kp', 'bot_kp', 'feat', 'mask'],
    input_size_list=[[1,64,22], [1,64,12], [1,64,29], [1,64]],
)
# For INT8 quantization, provide representative data:
# rknn.build(do_quantization=True, dataset='quant_dataset.txt')
rknn.build(do_quantization=False)  # FP32 baseline
rknn.export_rknn('checkpoints/bigru_hierarchical/bigru.rknn')
rknn.release()
```

> **Note on GRU support:** Rockchip NPU has limited support for dynamic
> unrolled GRUs. If conversion fails, use **RKNN's CPU fallback** for the
> BiGRU and only run YOLO on the NPU. The BiGRU is lightweight (< 2 ms on
> an ARM Cortex-A55) and does not need NPU acceleration.

---

## 9. C++ Integration Skeleton

The skeleton below maps precisely to `test/run_realtime.py` but in idiomatic C++
for an embedded RK3588 / RV1106 target.

```cpp
// fall_detector_main.cpp
// Compile with: g++ -O2 -std=c++17 fall_detector_main.cpp \
//   -I/usr/include/rknn -lrknn_api `pkg-config --cflags --libs opencv4` -o fall_detector

#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <array>
#include <cmath>
#include <cstring>

// ── Constants ──────────────────────────────────────────────────────────────
static constexpr int  NUM_KPT        = 17;
static constexpr int  TOP_KPT_DIM    = 22;   // joints 0-10, x/y
static constexpr int  BOT_KPT_DIM    = 12;   // joints 11-16, x/y
static constexpr int  FEAT_DIM       = 29;
static constexpr int  WINDOW_SIZE    = 64;   // T
static constexpr int  CLIP_FRAMES    = 64;   // same for 30fps model
static constexpr float CONF_THRESH   = 0.35f;
static constexpr float MIN_VALID_RATIO = 0.5f;

// ── FrameFeatures ──────────────────────────────────────────────────────────
struct FrameFeatures {
    float top_kp[TOP_KPT_DIM] = {};
    float bot_kp[BOT_KPT_DIM] = {};
    float feat[FEAT_DIM]      = {};
    float mask                = 0.0f;
};

// ── FallFrameBuffer ────────────────────────────────────────────────────────
class FallFrameBuffer {
    std::deque<FrameFeatures> _q;
public:
    void reset()              { _q.clear(); }
    void add(FrameFeatures f) { _q.push_back(f); if (_q.size() > CLIP_FRAMES) _q.pop_front(); }
    bool is_ready() const     { return (int)_q.size() == CLIP_FRAMES; }

    float valid_ratio() const {
        float s = 0;
        for (auto& f : _q) s += f.mask;
        return s / WINDOW_SIZE;
    }

    // Fill flat arrays (row-major) for RKNN input tensors
    // Each array must be pre-allocated to WINDOW_SIZE * DIM floats
    void assemble_batch(float* top_kp_out,   // [WINDOW_SIZE * TOP_KPT_DIM]
                        float* bot_kp_out,   // [WINDOW_SIZE * BOT_KPT_DIM]
                        float* feat_out,     // [WINDOW_SIZE * FEAT_DIM]
                        float* mask_out)     // [WINDOW_SIZE]
    const {
        memset(top_kp_out, 0, WINDOW_SIZE * TOP_KPT_DIM * sizeof(float));
        memset(bot_kp_out, 0, WINDOW_SIZE * BOT_KPT_DIM * sizeof(float));
        memset(feat_out,   0, WINDOW_SIZE * FEAT_DIM     * sizeof(float));
        memset(mask_out,   0, WINDOW_SIZE                * sizeof(float));

        int t = 0;
        for (auto& f : _q) {
            memcpy(top_kp_out + t * TOP_KPT_DIM, f.top_kp, TOP_KPT_DIM * sizeof(float));
            memcpy(bot_kp_out + t * BOT_KPT_DIM, f.bot_kp, BOT_KPT_DIM * sizeof(float));
            memcpy(feat_out   + t * FEAT_DIM,     f.feat,   FEAT_DIM     * sizeof(float));
            mask_out[t] = f.mask;
            ++t;
        }
        // Frames t..WINDOW_SIZE-1 remain zero (padding already applied by memset)
    }
};

// ── KalmanFilter4D (per keypoint) ─────────────────────────────────────────
struct KF4D {
    // state: [x, y, vx, vy]
    float state[4]    = {};
    float P[4][4]     = {};          // covariance
    float sigma_Q     = 20.0f;       // process noise
    float sigma_R_base = 10.0f;      // measurement noise base
    bool  initialized  = false;

    void reset() { initialized = false; memset(P, 0, sizeof(P)); }

    void initialize(float x, float y) {
        state[0] = x; state[1] = y; state[2] = 0; state[3] = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                P[i][j] = (i == j) ? 1.0f : 0.0f;
        initialized = true;
    }

    void predict(float dt) {
        if (!initialized) return;
        // state = F * state
        state[0] += state[2] * dt;
        state[1] += state[3] * dt;
        // P = F*P*Ft + Q  (simplified scalar Q)
        float Q = sigma_Q * sigma_Q;
        P[0][0] += dt*dt*P[2][2] + dt*(P[0][2]+P[2][0]) + Q;
        P[1][1] += dt*dt*P[3][3] + dt*(P[1][3]+P[3][1]) + Q;
        P[0][2] += dt*P[2][2]; P[2][0] = P[0][2];
        P[1][3] += dt*P[3][3]; P[3][1] = P[1][3];
        P[2][2] += Q; P[3][3] += Q;
    }

    void update(float meas_x, float meas_y, float conf) {
        if (!initialized) { initialize(meas_x, meas_y); return; }
        float R = (sigma_R_base / std::max(conf, 0.01f));
        R = R * R;
        // Kalman gain K = P*Ht / (H*P*Ht + R), H = [[1,0,0,0],[0,1,0,0]]
        float S0 = P[0][0] + R, S1 = P[1][1] + R;
        float K[4][2] = {
            {P[0][0]/S0, P[0][1]/S1},
            {P[1][0]/S0, P[1][1]/S1},
            {P[2][0]/S0, P[2][1]/S1},
            {P[3][0]/S0, P[3][1]/S1},
        };
        float inov0 = meas_x - state[0], inov1 = meas_y - state[1];
        for (int i = 0; i < 4; ++i)
            state[i] += K[i][0]*inov0 + K[i][1]*inov1;
        // simplified P update (Joseph form omitted for brevity)
        for (int i = 0; i < 4; ++i) {
            P[i][0] -= K[i][0]*P[0][0]; P[i][1] -= K[i][1]*P[1][1];
        }
    }

    void position(float& x, float& y) const { x = state[0]; y = state[1]; }
};

// ── PoseFeatureExtractor (C++ port of pose_features.py) ───────────────────
// NOTE: For production use the pre-compiled
//   pose_features/build/extract_pose_features_from_csv binary
//   or port the full PoseFeatures.cpp logic here.
// This is a minimal reference implementation.
struct PoseTrackState {
    bool  valid     = false;
    float kpts[17][3] = {};   // x, y, conf
    float bbox_h    = 0;
    float timestamp = 0;
    float hip[2]    = {};
    float vy_ring[5] = {}; int vy_n = 0;
    float hip_h_ring[5] = {}; int hip_h_n = 0;
    float ta_ring[10] = {}; int ta_n = 0;
    float last_roll_vy = 0; bool has_roll_vy = false;
};

// Full implementation follows the same logic as src/inference/pose_features.py.
// Key functions to implement:
//   float estimateBboxHeight(kpts, conf_min=0.3)
//   float* hipCenter(kpts, conf_min=0.3)           // returns [x,y] or null
//   bool   validPose(kpts, bbox_h)
//   void   normalizePose(kpts, bbox_h, hip, out_34f)
//   float  pairwiseDist(kpts, i, j, bbox_h)
//   float  angleAtJoint(A, B, C)
//   float  torsoAngle(kpts)
//   void   computeVelocity(prev_kpts, curr_kpts, prev_hip, curr_hip, dt, bbox_h, out_vel)

// ── RKNN wrapper for BiGRU ─────────────────────────────────────────────────
class BiGRUInference {
    rknn_context ctx = 0;
    // Input tensors: top_kp, bot_kp, feat, mask
    // Output tensor: logits (1, 2)
public:
    bool load(const char* model_path) {
        // Read .rknn file
        FILE* fp = fopen(model_path, "rb");
        if (!fp) return false;
        fseek(fp, 0, SEEK_END); size_t sz = ftell(fp); rewind(fp);
        std::vector<uint8_t> buf(sz);
        fread(buf.data(), 1, sz, fp); fclose(fp);

        int ret = rknn_init(&ctx, buf.data(), buf.size(), 0, nullptr);
        return ret == RKNN_SUCC;
    }

    // Returns [P_not_fall, P_fall]
    std::pair<float,float> infer(const float* top_kp,
                                 const float* bot_kp,
                                 const float* feat,
                                 const float* mask)
    {
        rknn_input inputs[4] = {};
        inputs[0] = {0, (void*)top_kp, WINDOW_SIZE*TOP_KPT_DIM*sizeof(float), 1, RKNN_TENSOR_FLOAT32};
        inputs[1] = {1, (void*)bot_kp, WINDOW_SIZE*BOT_KPT_DIM*sizeof(float), 1, RKNN_TENSOR_FLOAT32};
        inputs[2] = {2, (void*)feat,   WINDOW_SIZE*FEAT_DIM    *sizeof(float), 1, RKNN_TENSOR_FLOAT32};
        inputs[3] = {3, (void*)mask,   WINDOW_SIZE             *sizeof(float), 1, RKNN_TENSOR_FLOAT32};
        rknn_inputs_set(ctx, 4, inputs);

        rknn_run(ctx, nullptr);

        float logits[2] = {};
        rknn_output outputs[1] = {};
        outputs[0] = {1, logits, 2*sizeof(float)};
        rknn_outputs_get(ctx, 1, outputs, nullptr);
        rknn_outputs_release(ctx, 1, outputs);

        // softmax
        float e0 = expf(logits[0]), e1 = expf(logits[1]);
        float sum = e0 + e1;
        return {e0/sum, e1/sum};
    }

    ~BiGRUInference() { if (ctx) rknn_destroy(ctx); }
};

// ── Main loop ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // 1. Load models
    // BiGRUInference bigru; bigru.load("bigru.rknn");
    // Load YOLOv8 pose via RKNN or ultralytics C++ binding

    // 2. Open source
    cv::VideoCapture cap(argc > 1 ? argv[1] : "0");
    double source_fps = cap.get(cv::CAP_PROP_FPS);
    if (source_fps < 1) source_fps = 25;

    FallFrameBuffer buffer;
    PoseTrackState  track;

    // Flat input arrays
    float top_kp_buf[WINDOW_SIZE * TOP_KPT_DIM];
    float bot_kp_buf[WINDOW_SIZE * BOT_KPT_DIM];
    float feat_buf  [WINDOW_SIZE * FEAT_DIM];
    float mask_buf  [WINDOW_SIZE];

    int    frame_idx = 0;
    double next_sample = 0;
    double sample_interval = 1.0 / source_fps;

    cv::Mat frame;
    std::string label = "not_fall";
    float fall_prob = 0.0f;

    while (cap.read(frame)) {
        double frame_time = frame_idx / source_fps;
        double frame_mid  = frame_time + 0.5 / source_fps;

        if (frame_mid + 1e-9 >= next_sample) {
            double timestamp = next_sample;
            next_sample += sample_interval;

            // ── YOLO pose inference (replace with actual RKNN call) ──
            // float kpts[17][3];   // fill from YOLO output
            // bool person_found = run_yolo(frame, kpts, CONF_THRESH);

            // ── Feature extraction ──
            // if (person_found) {
            //     FrameFeatures ff = extract_features(kpts, timestamp, track);
            //     buffer.add(ff);
            // } else {
            //     reset_track(track);
            //     buffer.reset();
            // }

            // ── BiGRU inference ──
            if (buffer.is_ready() && buffer.valid_ratio() >= MIN_VALID_RATIO) {
                buffer.assemble_batch(top_kp_buf, bot_kp_buf, feat_buf, mask_buf);
                // auto [p_nf, p_f] = bigru.infer(top_kp_buf, bot_kp_buf, feat_buf, mask_buf);
                // fall_prob = p_f;
                // label = (p_f >= 0.5f) ? "fall" : "not_fall";
            }
        }
        ++frame_idx;

        // ── Draw HUD ──
        cv::putText(frame, label + " " + std::to_string(fall_prob),
                    {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    label == "fall" ? cv::Scalar(0,0,255) : cv::Scalar(0,220,0), 2);
        cv::imshow("Fall Detection", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
```

---

## 10. Tensor Layout Reference

### COCO-17 Keypoint Index Map

```
Index  Name              Body Region
  0    nose              head
  1    left_eye          head
  2    right_eye         head
  3    left_ear          head
  4    right_ear         head
  5    left_shoulder     top  (in top_kp)
  6    right_shoulder    top
  7    left_elbow        top
  8    right_elbow       top
  9    left_wrist        top
 10    right_wrist       top   ← last top joint
 11    left_hip          bot  (in bot_kp)
 12    right_hip         bot
 13    left_knee         bot
 14    right_knee        bot
 15    left_ankle        bot
 16    right_ankle       bot   ← last bot joint
```

`top_kp[22]` = `[x0, y0, x1, y1, ..., x10, y10]` (22 values, joints 0–10)
`bot_kp[12]` = `[x11, y11, x12, y12, ..., x16, y16]` (12 values, joints 11–16)

Both normalized: `x' = (x_pixel - hip_x) / bbox_height`

### Model Input Tensor Summary

| Tensor | Shape | Size (floats) | Notes |
|--------|-------|---------------|-------|
| `top_kp` | (1, 64, 22) | 1408 | Joints 0–10, hip-normalized |
| `bot_kp` | (1, 64, 12) | 768 | Joints 11–16, hip-normalized |
| `feat` | (1, 64, 29) | 1856 | Engineered features |
| `mask` | (1, 64) | 64 | 0.0 or 1.0 per frame |
| **Total input** | | **4096** | |
| `logits` (output) | (1, 2) | 2 | [not_fall_logit, fall_logit] |

For the 7.5-fps model replace 64 → 16 everywhere.

---

## 11. Configuration Reference

| Parameter | 30-fps model | 7.5-fps model |
|-----------|-------------|--------------|
| Config file | `configs/bigru_hierarchical.yaml` | `configs/bigru_hierarchical_7fps.yaml` |
| Checkpoint | `checkpoints/bigru_hierarchical/best.pt` | `checkpoints/bigru_hierarchical_7fps/best.pt` |
| `window_size` (T) | **64** | **16** |
| `clip_frames` | 64 (default) | **14** (pass `--clip-frames 14`) |
| `process_fps` | source_fps | **7.5** |
| `hidden_dim` | 128 | 128 |
| `num_layers` | 2 | 2 |
| `dropout` | 0.3 | 0.3 |
| `min_valid_frame_ratio` | 0.5 | 0.5 |
| YOLO conf threshold | 0.35 | 0.35 |
| Kalman miss_tolerance | 2 | 2 |
| Kalman target_fps | 30 | 7.5 (no Kalman needed) |

---

## 12. Accuracy-Preserving Checklist

Follow this exactly when porting to C++ / NPU to avoid any accuracy regression.

- [ ] **YOLO keypoints in pixel space** — do NOT normalize before passing to
      the feature extractor. The feature extractor normalizes internally using
      `bbox_height`.

- [ ] **Hip-center origin** — `norm_kpt.x = (pixel_x - hip_center_x) / bbox_height`.
      If both hips are invisible, use whichever single hip is visible.

- [ ] **bbox_height from keypoints**, not from the YOLO bounding box.
      `bbox_height = max_y(kpts with conf≥0.3) − min_y(kpts with conf≥0.3)`.
      Minimum clamp: `1.0 px`.

- [ ] **Confidence thresholds:**
      - Keypoint conf ≥ 0.3 for hip/position/distance features
      - Torso-shoulder/hip conf ≥ 0.5 for `torso_angle`
      - `valid_pose = True` only when hip_center is non-null AND bbox_height ≥ 1.0

- [ ] **NaN / missing features → 0.0**, not any other sentinel.

- [ ] **Rolling windows per-track**, not per-frame globally:
      - `vel_hip_vy` history: last 5 frames → `rolling_mean_vertical_velocity`
      - `hip_height` history: last 5 frames → `min_hip_height_over_window`
      - `torso_angle` history: last 10 frames → `torso_angle_std`

- [ ] **Velocity features are delta / dt**, not just pixel differences.
      `vel_hip_vx = Δhip_x / (dt × bbox_height)` — both dt and height
      normalization are required.

- [ ] **Reset track state on person absence** — do not let velocity /
      rolling-window state persist across track gaps; this would feed the model
      nonsensical temporal features.

- [ ] **Zero-padding at the END of the tensor**, not the beginning.
      `batch[clip_frames..window_size-1]` must be all zeros with `mask=0`.

- [ ] **Mask denominator = window_size** when computing `valid_ratio`,
      even if `clip_frames < window_size`. Gate inference only if
      `valid_ratio ≥ 0.5`.

- [ ] **v2 checkpoint**: apply `feat = (feat - feat_mean) / feat_std`
      using the `feat_mean` and `feat_std` buffers stored inside the `.pt`
      checkpoint before forwarding to the model.

- [ ] **Softmax on logits** — the model outputs raw logits, not probabilities.
      Apply `softmax(logits, dim=-1)` before reading `probs[1]` as `fall_prob`.

- [ ] **Single person only** — pick the highest-confidence detection above
      threshold, ignore all others. The feature extractor assumes track
      continuity for a single person.

---

*Generated from source analysis of `test/run_realtime.py`,
`src/inference/`, `src/models/bigru_hierarchical.py`, and
`src/constants.py`.*
