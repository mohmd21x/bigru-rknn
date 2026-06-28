#pragma once

#include "PoseEstimator.h"
#include <array>
#include <optional>
#include <vector>

// Utility functions and feature extractors built on top of PoseEstimator keypoints.
// This header is intentionally self-contained and free of state so it can be
// reused from FallDetector, visualizer, batch_analyzer, etc.

namespace pose_features {

// COCO keypoint indices for reference.
// 0: Nose          5: L Shoulder    11: L Hip
// 1: L Eye         6: R Shoulder    12: R Hip
// 2: R Eye         7: L Elbow       13: L Knee
// 3: L Ear         8: R Elbow       14: R Knee
// 4: R Ear         9: L Wrist       15: L Ankle
//                 10: R Wrist       16: R Ankle

struct NormalizedPose {
    // Hip-centered, height-normalized 2D coordinates for all 17 keypoints.
    // If a keypoint has very low confidence it is still stored, but the caller
    // can check the original confidences from PoseResult::keypoints if needed.
    std::array<float, 34> values{}; // [x0', y0', x1', y1', ..., x16', y16']
};

struct PairwiseDistances {
    // All distances are normalized by person height (bbox height).
    float shoulderWidthH = 0.0f;
    float hipWidthH      = 0.0f;
    float headHipH       = 0.0f;

    float leftThighH     = 0.0f;
    float rightThighH    = 0.0f;
    float leftShinH      = 0.0f;
    float rightShinH     = 0.0f;

    float handHandH      = 0.0f;
    float leftHandHipH   = 0.0f;
    float rightHandHipH  = 0.0f;
};

struct JointAngles {
    // Angles in degrees. std::nullopt means angle could not be computed
    // reliably (missing/low-confidence keypoints).
    std::optional<float> leftHip;
    std::optional<float> rightHip;
    std::optional<float> leftKnee;
    std::optional<float> rightKnee;
};

struct VelocityFeatures {
    // Normalized by person height.
    float vxH   = 0.0f; // horizontal hip velocity (heights / second)
    float vyH   = 0.0f; // vertical hip velocity (heights / second)
    float speed = 0.0f; // total hip speed

    float handMaxSpeedH  = 0.0f; // max of both wrists
    float footMaxSpeedH  = 0.0f; // max of both ankles
};

// Compute hip center from keypoints (average of left and right hip).
// Returns std::nullopt if both hips have very low confidence.
std::optional<cv::Point2f> hipCenter(const std::vector<Keypoint>& kpts,
                                     float minConf = 0.3f);

// Torso angle (degrees): angle of each shoulder–hip segment vs vertical (legacy FallDetector).
// 0° ≈ standing upright, ~90° ≈ lying flat in the image plane.
// Averages left (5–11) and right (6–12) when both pass confidence; one side if only one passes.
// Returns std::nullopt if neither side is usable.
std::optional<float> computeTorsoAngle(const std::vector<Keypoint>& kpts,
                                       float minConf = 0.5f);

// Temporal torso-angle delta between consecutive frames for one track:
// angle_change = current_torso_angle - previous_torso_angle.
// Returns std::nullopt if either angle is unavailable or dt is invalid (<= 0 or non-finite).
std::optional<float> computeAngleChange(const std::optional<float>& currentTorsoAngle,
                                        const std::optional<float>& previousTorsoAngle,
                                        float dt);

// Temporal torso angular velocity:
// torso_angular_velocity = angle_change / dt (degrees/second).
// Returns std::nullopt if angle_change is unavailable or dt is invalid (<= 0 or non-finite).
std::optional<float> computeTorsoAngularVelocity(const std::optional<float>& angleChange,
                                                 float dt);

// Hip height feature: hip_center_y / pose height.
// Returns std::nullopt when hip center is unavailable.
std::optional<float> computeHipHeight(const PoseResult& pose,
                                      float minConf = 0.3f);

// Temporal hip-height delta between consecutive frames for one track:
// hip_height_change = current_hip_height - previous_hip_height.
// Returns std::nullopt if either hip_height is unavailable or dt is invalid (<= 0 or non-finite).
std::optional<float> computeHipHeightChange(const std::optional<float>& currentHipHeight,
                                            const std::optional<float>& previousHipHeight,
                                            float dt);

// Aspect ratio from keypoint-estimated bbox: estimated_width / pose height.
// Estimated width uses visible keypoints with conf >= minConf and finite x/y.
// Returns std::nullopt when width cannot be estimated.
std::optional<float> computeBboxAspectRatioFromKeypoints(const PoseResult& pose,
                                                         float minConf = 0.3f);

// Build a hip-centered, height-normalized pose vector.
// If hip center cannot be computed or height is invalid, returns std::nullopt.
std::optional<NormalizedPose> computeNormalizedPose(const PoseResult& pose);

// Compute a small set of high-signal pairwise distances, normalized by height.
// Returns std::nullopt if height is invalid.
std::optional<PairwiseDistances> computePairwiseDistances(const PoseResult& pose);

// Compute a compact set of joint angles (hips and knees).
JointAngles computeJointAngles(const std::vector<Keypoint>& kpts,
                               float minConf = 0.5f);

// Compute basic velocity features between two frames for a single tracked person.
//   prevPose / currPose: poses for the same person in consecutive frames
//   prevHip / currHip:   hip centers in image coordinates
//   dt:                  time delta in seconds
// All outputs are normalized by current bbox height.
// If dt <= 0 or height is invalid, all fields will be zero.
VelocityFeatures computeVelocityFeatures(const PoseResult& prevPose,
                                         const PoseResult& currPose,
                                         const cv::Point2f& prevHip,
                                         const cv::Point2f& currHip,
                                         float dt);

} // namespace pose_features

