#pragma once

#include "PoseEstimator.h"

#include <array>
#include <string>

// Converts one row of exported pose CSV data into a PoseResult suitable for
// pose_features helpers. CSV file I/O is not included — fill CsvPoseRow yourself.

struct CsvPoseRow {
    std::string video_name;
    int frame_index = 0;
    double timestamp = 0.0;
    int person_id = 0;

    // COCO order: 0 nose .. 16 R ankle (same as PoseEstimator / PoseFeatures).
    std::array<float, 17> kpt_x{};
    std::array<float, 17> kpt_y{};
    std::array<float, 17> kpt_conf{};
};

struct CsvPoseAdaptation {
    PoseResult pose;
    float estimated_bbox_height = 0.0f;
    bool valid_pose = false;
};

// Builds keypoints, estimates bbox height from visible joints, fills pose.box.height,
// and sets valid_pose when hip center and height are usable (see .cpp comments).
CsvPoseAdaptation adaptCsvRowToPose(const CsvPoseRow& row);
