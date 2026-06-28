#pragma once

// Minimal pose types for offline CSV feature extraction.
// The full fall_cpp PoseEstimator pulls in ONNX Runtime; this stub keeps the
// CSV tools buildable without inference dependencies.

#include <opencv2/core.hpp>
#include <vector>

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float conf = 0.0f;
};

struct PoseResult {
    int id = 0;
    float confidence = 0.0f;
    cv::Rect box{};
    std::vector<Keypoint> keypoints;
};
