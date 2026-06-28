Pose Features Extractor (standalone_utils/pose_features)
=======================================================

Project Overview
----------------
This module is a pose-feature extraction utility for downstream fall-detection
and general ML workflows. It converts pose keypoint CSV files (COCO 17 keypoints,
per-frame) into feature-rich CSV outputs with engineered geometric and temporal
signals.

- What it is: a feature extractor (offline / batch-friendly) that prepares
  ML-ready numeric features.
- What it is not: a fall decision system. It does not implement a fall state
  machine, tracking logic, or final fall event confirmation rules.


Input / Output
--------------
Input: CSV containing pose keypoints in COCO-17 format, per frame and per person:
  - 17 keypoints x (x, y, confidence) per frame
  - plus basic identifiers (e.g., video name / frame index / timestamp / person id),
    depending on your export source

Output: CSV containing:
  - original identifying metadata
  - normalized pose values
  - engineered features (distances, angles, temporal features, window-based aggregates)

Processing granularity:
  - Per-frame: features are computed for each frame.
  - Per-person: each person stream is processed independently (rows are associated
    with person_id when present).


Features Description (high-level, grouped)
------------------------------------------
This extractor outputs a wide feature set. The exact column list may evolve, but it
is broadly grouped as:

- Metadata
  - video/frame identifiers (e.g., video_name, frame_index, timestamp, person_id)
- Normalized pose
  - bbox-relative / scale-normalized keypoints (flattened 34 values for x/y across COCO-17)
- Distances
  - key body distances (e.g., shoulder width, hip width, limb segment lengths)
- Joint angles
  - major joint angles (e.g., hips, knees)
- Torso features
  - torso inclination / stability (including window-based variability)
- Hip features
  - hip height and its change over time (including window minima)
- Velocity features
  - approximate motion signals from keypoints (e.g., hip velocity, accelerations, peak limb speeds)
- Window-based features
  - small sliding-window aggregates (e.g., rolling means / minima / standard deviation over recent frames)


How to Build
------------
Run these exact commands from the repository root:

cd standalone_utils/pose_features
mkdir -p build
cd build
cmake ..
cmake --build .

This produces:
  ./build/extract_pose_features_from_csv


How to Run (Single File)
------------------------
Use this exact command:

./build/extract_pose_features_from_csv input.csv output.csv


How to Run (Batch Mode)
-----------------------
Create an output directory:

mkdir -p output_csv

Then run the batch loop:

for f in input_csv/*.csv; do
    filename=$(basename "$f" .csv)
    ./build/extract_pose_features_from_csv "$f" "output_csv/${filename}_features.csv"
done


Notes / Behavior
----------------
- Missing values: missing/invalid keypoints and feature values are written as NaN where appropriate.
- Temporal features: velocity/acceleration-style features require a previous valid frame; the first
  valid frame (or gaps) may not have meaningful temporal values.
- Validity flags: some rows may be marked invalid via helper columns such as valid_pose (and related
  flags) to support filtering in analytics/ML.
- Window-based features: small sliding windows are used for certain stability/aggregate features;
  at the start of a sequence, window-based values may be unavailable or less stable until enough
  history is present.


Limitations
-----------
- No fall decision logic (no final "fall/not-fall" output).
- No state machine (no confirmation timers or event-level gating).
- No tracking logic (no ID association across frames; assumes person_id is already provided if
  multi-person is present).
- Pure feature extraction only from pose keypoints and derived geometry/temporal signals.


Suggested Next Steps
--------------------
- Use the exported feature CSVs as inputs to ML models (classical or deep learning).
- Build temporal sequences/windows (e.g., 16-frame or 30-frame windows) per person for sequence models.
- If needed for debugging, add optional "rule-style" diagnostic columns in a separate layer (without
  merging a full fall detector/state machine into this extractor).

