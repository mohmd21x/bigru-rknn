# Pose Keypoint Techniques:

**Purpose:** Map every pose-based feature technique to project usage, code location, improvement notes, and relevant papers.

---

## 1. Raw Geometric Features

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Joint coordinates** | Raw (x, y) or (x, y, z) per keypoint from the pose model | **Yes** | `include/PoseEstimator.h`: `struct Keypoint { x, y, conf }`, `PoseResult::keypoints` (17 keypoints). `src/PoseEstimator.cpp`: postprocess fills keypoints from model output. All callers use `detections[j].keypoints`, e.g. `FallDetector.cpp`, `visualizer.cpp`, `rtsp_monitor.cpp`, `batch_analyzer.cpp`. | Keep as base; consider storing optional z if moving to 3D. | — |
| **Pairwise distances** | Euclidean distance between selected joint pairs (e.g. shoulder–elbow, head–hip, hand–torso) | **No** | Not computed. Only implicit: hip center (midpoint of 11–12), box width/height. | Add distances (e.g. head–hip, hand–head, knee–hip) to better distinguish sitting/bending/kneeling from lying; use for rule thresholds or ML input. | MDPI “Synergistic Integration of Skeletal Kinematic Features”; Springer “Angle-Based Feature Extraction” |
| **Joint angles** | Angle at each joint (elbow, knee, shoulder, etc.) from adjacent segments | **Partial** | **Only torso angle.** `src/FallDetector.cpp`: `calculateTorsoAngle()` (L86–114): left shoulder–hip (5–11) and right (6–12), `atan2(dx,dy)` → degrees; averaged if both visible. Used in `update()` for rules and state logic. | Add knee/hip/ankle angles to discriminate sitting/kneeling from lying; consider elbow for “reaching” vs fall. | Springer “Fall Detection Using Angle-Based Feature Extraction” (70-angle features); IEEE “Advances in Skeleton-Based Fall Detection” |
| **Relative positions** | Joint positions in a body-centric frame (e.g. subtract pelvis/hip center from all keypoints) | **Partial** | Hip center used as **reference point** for tracking and matching: `FallDetector.cpp` `getCenter()` (L116–123), hip center in `update()` for Kalman and matching; `rtsp_monitor.cpp` `poseHipCenterOrBoxCenter()`. No full “all joints relative to hip” vector. | Compute full relative pose (all keypoints − hip_center) for scale/position-invariant features; optional rotation alignment. | — |

---

## 2. Normalization and Invariances

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Scale normalization** | Normalize distances/coordinates by body size (e.g. shoulder width, hip width, or bbox height) | **Yes** | `FallDetector.cpp` in `update()`: `height = det.box.height`, `normVy = vy / height` (L215–216), `hipHeight = currentCenter.y / height` (L233). Velocity and hip height are scale-normalized. | Consider normalizing more features (e.g. horizontal displacement, pairwise distances) by height or shoulder width for cross-person/camera robustness. | MDPI “Synergistic Integration…”; ElderFallGuard (arXiv) |
| **Translation normalization** | Re-center so a reference joint (e.g. pelvis) is at origin | **Partial** | Hip center is **used** as tracking anchor and for matching; coordinates are not explicitly re-centered for feature computation (raw image-space coords used for angle/velocity). | Re-center all keypoints by hip for feature vectors; makes position-invariant inputs for any future classifier. | — |
| **Rotation normalization** | Align coordinate frame to body orientation (e.g. shoulder line horizontal, forward direction fixed) | **No** | Torso angle is used as a **scalar feature** only; no rotation of joint coordinates into body frame. | Rotate skeleton into torso-aligned frame to get rotation-invariant pose representation; useful if adding learned models. | SkeleTR (ICCV 2023); skeleton action recognition literature |

---

## 3. Temporal / Motion Features (for Sequences)

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Velocities** | First derivative of position over time (e.g. joint or center velocity) | **Yes** | `FallDetector.cpp` in `update()`: `vy = (currentCenter.y - person.lastHipCenter.y) / dt` (L215), `normVy = vy / height`; stored in `PersonState::verticalVelocities` (deque, last 5), averaged as `avgVy`. Used in rules 1–4. | Add horizontal velocity or full 2D speed; optionally use Kalman state velocity (vx, vy) for smoother signal. | Nature TCN paper; MDPI “Synergistic Integration…” |
| **Accelerations / jerks** | Second (and optionally third) derivative of position | **Yes** | `FallDetector.cpp` in `update()`: `acceleration = (avgVy - person.lastVelocity) / dt` (L329); `maxAccelerationInFall` updated (L330–331). Used for logging/display; not in trigger rules. | Use acceleration explicitly in rules (e.g. impact spike threshold) to separate hard falls from slow lies. | ScienceDirect “Kalman filter-enhanced… accelerometer”; PMC “Enhancing elderly care” |
| **Temporal joint angle changes** | How joint angles change over time (e.g. knee angle over gait cycle) | **Yes** | `FallDetector.cpp`: `angleChange = angle - person.lastAngle` (L241), `angVel = (angle - person.lastAngle) / dt` (L219); `person.angles` deque (last 10). Used in rules (e.g. rule 1, 3). | Smooth angle (e.g. short EMA) before differencing to reduce pose jitter; consider leg angles over time for sit vs fall. | Springer “Angle-Based Feature Extraction”; LSTM+MediaPipe (IJACSA) |
| **Trajectories** | Path of keypoint or body center over time (path length, curvature, direction changes) | **Partial** | Kalman **predict** gives next position; no explicit trajectory features (path length, curvature). Position history exists implicitly in `lastHipCenter` and Kalman state. | Add trajectory summary features (e.g. path length over window, curvature) for complex activity recognition. | — |
| **Frequency-domain features** | FFT (or similar) on joint/angle time series (dominant frequency, harmonics) | **No** | Not implemented. | Optional for periodic motion (e.g. gait, repetitive actions); lower priority for binary fall detection. | — |

---

## 4. Symmetry and Coordination

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Left–right symmetry** | Compare left vs right limb (angles, positions, or distances) | **Partial** | Only in **torso angle**: `calculateTorsoAngle()` averages left (5–11) and right (6–12) segments for robustness; no explicit “left vs right” feature or asymmetry score. | Add left vs right leg/arm angle or position difference to detect asymmetric postures (e.g. leaning, one-sided collapse). | — |
| **Inter-joint coordination** | Phase/timing relationship between joints (e.g. hip vs knee angle phase) | **No** | Not computed. | Could add cross-correlation or relative phase between key angles for fine-grained activity recognition. | — |

---

## 5. Posture and Shape Descriptors (Single-Frame)

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Center of mass (approximate)** | Weighted average of joint positions (segment masses) or simple centroid | **Partial** | **Hip center** used as proxy: `getCenter()` (L116–123), and everywhere hip center is used (tracking, matching, velocity). No segment-weighted CoM. | Optional: approximate CoM from segment midpoints and use for balance/support analysis; hip center is often sufficient for fall. | — |
| **Body orientation** | Global orientation of torso/head (tilt, facing direction) | **Yes** | **Torso angle** from vertical: `calculateTorsoAngle()` (L86–114); used as main posture cue (e.g. horizontal ≈ lying). | Consider head or neck tilt for “looking down” vs “lying”; keep torso as primary. | Poseaware Fall Net (IEEE); ElderFallGuard |
| **Compactness / spread** | Single-frame spread of pose (e.g. mean distance from center, bbox aspect ratio) | **Partial** | **Aspect ratio only:** `aspectRatio = det.box.width / det.box.height` (L234), stored in `lastAspectRatio`; not used in current rules. | Use aspect ratio in rules (e.g. wide + horizontal → stronger fall evidence); optional: mean joint distance from hip as “spread”. | — |
| **Pose templates / embeddings** | Discrete pose clusters or learned embeddings (e.g. k-means on angles, neural embedding) | **No** | No clustering or embedding; decision is rule-based on scalars. | Optional: pose clusters or small embedding model for richer state representation or ML classifier. | Expressive Keypoints (arXiv); SkeletonAgent |

---

## 6. Task-Specific Feature Examples

| Task / Application | Relevant Techniques | Used in Project? | Where in Code | Improvement Notes | References |
|--------------------|---------------------|------------------|----------------|--------------------|------------|
| **Action recognition** (walking, sitting, waving) | Joint angles, velocities, temporal sequences, pose templates | **Partial** | Only **fall vs non-fall**; no multi-class actions. Uses torso angle, velocity, angular velocity, state machine. | To support multiple actions: add more angles, pairwise distances, and temporal features; consider LSTM/TCN or GCN. | SkeleTR; UniSTFormer; Nature GCN fall recognition |
| **Gait analysis** | Hip/knee/ankle angles over time, step frequency, symmetry, phase | **No** | No gait-specific features (step detection, cadence, symmetry). | Add if needed for “walking vs falling” or mobility assessment. | — |
| **Ergonomics / workplace safety** | Trunk flexion, shoulder elevation, time in risky posture | **No** | Torso angle is generic; no ergonomic risk angles or exposure time. | Add trunk flexion/rotation and shoulder angle with thresholds for ergonomics use case. | — |
| **Affect / emotion estimation** | Posture openness, head tilt, gesture speed/amplitude | **No** | Not in scope. | Would need openness (e.g. hand–hand distance), head pose, motion intensity. | — |
| **Sports performance** | Joint angles at key events, angular velocities, kinematic chain timing | **No** | Not in scope. | Would need event detection and segment angular velocities. | — |

---

## 7. Practical Considerations

| Topic | Description | Used? | Where in Code | Improvement Notes | References |
|-------|-------------|-------|----------------|--------------------|------------|
| **Missing / noisy keypoints** | Handling low confidence, occlusion, dropout (interpolation, smoothing, visibility flags) | **Partial** | **Confidence checks:** `calculateTorsoAngle()` uses `conf > 0.5` for shoulder/hip (L91–92); `getCenter()` uses `conf > 0.3` for hips (L118); matching skips detections if both hips &lt; 0.3 (L170). **No** keypoint interpolation or smoothing of raw coordinates. | Add temporal smoothing (e.g. EMA) or interpolation for missing keypoints; optional low-pass filter on angles/positions before derivatives. | IEEE “Advances in Skeleton-Based Fall Detection” (survey) |
| **2D vs 3D** | Using 2D keypoints vs 3D (depth, reconstruction) | **2D only** | All keypoints are (x, y, conf) from YOLOv8 pose; no depth or 3D. | 3D would improve angle/distance accuracy and viewpoint invariance; consider 2D→3D lifting if needed. | Nature “Multistage fall detection via 3D pose sequences and TCN” |
| **Feature dimensionality** | Number of features; selection or reduction (PCA, feature selection) | **Handcrafted scalars** | Small set: angle, angleChange, angVel, avgVy, hipHeight, aspectRatio, angleVariance, minHipHeight, maxVelocityInFall, maxAccelerationInFall. No PCA or formal selection. | If adding many features: add feature selection (e.g. Boruta) or PCA; keep thresholds interpretable for debugging. | Springer “Angle-Based Feature Extraction” (Boruta + SVM) |

---

## 8. Additional Techniques (Tracking, State, Logging)

| Technique | Description | Used? | Where in Code | Improvement Notes | References |
|-----------|-------------|-------|----------------|--------------------|------------|
| **Kalman filtering** | Smoothing and prediction of hip center position (and implicit velocity) | **Yes** | `FallDetector.h`: `PersonState::kf`, `stateMat`, `measMat`. `FallDetector.cpp`: `initKalman()` (L8–42), `predict()` (L44–47), `correct()` (L49–53); used in `update()` for prediction and correction. | Expose Q/R as parameters; optionally use KF velocity (state index 2,3) in rules. | ScienceDirect “Kalman filter-enhanced… accelerometer” |
| **Multi-person tracking** | Associate detections across frames (e.g. Hungarian, greedy by distance) | **Yes** | `FallDetector.cpp` in `update()`: greedy matching by Euclidean distance of hip center (L148–191), gating threshold 150 px; `cleanOldTracks()` drops tracks after 2 s loss. | Consider IoU of bboxes as cost or gate; tune distance threshold per resolution. | — |
| **State machine (temporal logic)** | Multi-state decision over time (e.g. NORMAL → FALLING → LANDED → CONFIRMED) | **Yes** | `FallDetector.h`: `FallState` enum. `FallDetector.cpp`: full state machine in `update()` (L271–499) with rules 1–5, velocity/angle thresholds, lay-down time, extended observation, angle variance, min hip height. | Central to design; tune thresholds (e.g. per camera/FPS); consider learning thresholds from data. | ElderFallGuard; Poseaware Fall Net; PMC “Enhancing elderly care” |
| **Logging / metrics export** | Persist features and state for analysis | **Yes** | `FallDetector.cpp`: `logMetrics()` (L75–84), CSV header (L59); `batch_analyzer.cpp`: metrics log with timestamp, frame, person_id, state, angle, velocity, hip_height, fall_reason. | Ensure all new features added to logs if used for tuning or ML. | — |

---

## 9. File Reference Summary

| File | Contents |
|------|----------|
| **`include/PoseEstimator.h`** | `Keypoint`, `PoseResult`, 17 keypoints, box. |
| **`src/PoseEstimator.cpp`** | Preprocess, inference, postprocess → boxes + keypoints (scale applied). |
| **`include/FallDetector.h`** | `FallState`, `PersonState` (Kalman, deques, thresholds), `FallDetector` API. |
| **`src/FallDetector.cpp`** | Torso angle, hip center, velocity/acceleration, rules, state machine, Kalman, matching, logging. |
| **`src/visualizer.cpp`** | Draw skeleton, overlay state/angle/velocity/trigger reason. |
| **`src/rtsp_monitor.cpp`** | RTSP loop, pose→person matching via hip center, drawing, fall output. |
| **`src/batch_analyzer.cpp`** | Video batch run, CSV metrics (angle, velocity, hip_height, fall_reason). |
| **`src/main.cpp`** | Benchmark harness (videos, fall tag, detector stats). |

---

## 10. Reference Papers (Short List)

| Topic | Paper / Source |
|-------|-----------------|
| Skeletal kinematic features (angles, distances, velocity) | MDPI Sensors: “Synergistic Integration of Skeletal Kinematic Features for Vision-Based Fall Detection” |
| Angle-based features + ML | Springer: “Fall Detection Using Angle-Based Feature Extraction from Human Skeleton and Machine Learning” |
| Survey (handcrafted → deep) | IEEE: “Advances in Skeleton-Based Fall Detection in RGB Videos: from Handcrafted to Deep Learning Approaches” |
| Pose-based fall detection | IEEE: “Poseaware Fall Net: Fall Detection Using Human Pose Estimation” |
| 3D pose + TCN | Nature Scientific Reports: “Multistage fall detection framework via 3D pose sequences and TCN integration” |
| GCN / spatio-temporal | Nature Scientific Reports: “Fall recognition using three stream spatio temporal GCN with adaptive feature aggregation” |
| MediaPipe + LSTM | SAI IJACSA: “A New Method for Real-Time Fall Detection Based on MediaPipe Pose Estimation and LSTM” |
| MediaPipe + RF, real-time | arXiv: “ElderFallGuard: Real-Time IoT and Computer Vision-Based Fall Detection System” |
| Kalman + accelerometer + ML | ScienceDirect: “Efficient fall detection using Kalman filter-enhanced triaxial accelerometer signals and machine learning” |
| Skeleton action recognition | arXiv: “Expressive Keypoints for Skeleton-based Action Recognition”; ICCV: “SkeleTR”; arXiv: “UniSTFormer” |

---

*Document generated for team use. Update “Used?” and “Where in Code” as the codebase changes.*
