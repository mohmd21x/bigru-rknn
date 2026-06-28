#include "CsvPoseAdapter.h"
#include "PoseFeatures.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kConfForHeight = 0.3f;
constexpr float kMinHeight = 1.0f;

std::vector<Keypoint> buildKeypoints(const CsvPoseRow& row) {
    std::vector<Keypoint> keypoints;
    keypoints.reserve(17);
    for (int i = 0; i < 17; ++i) {
        Keypoint kp;
        kp.x = row.kpt_x[static_cast<size_t>(i)];
        kp.y = row.kpt_y[static_cast<size_t>(i)];
        kp.conf = row.kpt_conf[static_cast<size_t>(i)];
        keypoints.push_back(kp);
    }
    return keypoints;
}

// Uses keypoints with conf >= threshold and finite x,y; height = max_y - min_y, then clamped.
float estimateBboxHeightFromKeypoints(const std::vector<Keypoint>& keypoints,
                                      float confMin) {
    bool any = false;
    float minY = 0.0f;
    float maxY = 0.0f;

    for (const Keypoint& kp : keypoints) {
        if (kp.conf < confMin) {
            continue;
        }
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            continue;
        }
        if (!any) {
            minY = kp.y;
            maxY = kp.y;
            any = true;
        } else {
            minY = std::min(minY, kp.y);
            maxY = std::max(maxY, kp.y);
        }
    }

    if (!any) {
        return 0.0f;
    }

    float span = maxY - minY;
    if (span < kMinHeight) {
        span = kMinHeight;
    }
    return span;
}

bool heightEstimateIsValid(float h) {
    return std::isfinite(h) && h >= kMinHeight;
}

} // namespace

CsvPoseAdaptation adaptCsvRowToPose(const CsvPoseRow& row) {
    CsvPoseAdaptation out;
    out.pose.confidence = 1.0f;
    out.pose.id = row.person_id;

    std::vector<Keypoint> keypoints = buildKeypoints(row);
    out.pose.keypoints = std::move(keypoints);

    out.estimated_bbox_height =
        estimateBboxHeightFromKeypoints(out.pose.keypoints, kConfForHeight);

    // PoseFeatures uses integer box.height; other fields stay 0.
    const float h = out.estimated_bbox_height;
    int heightInt = 0;
    if (heightEstimateIsValid(h)) {
        heightInt = static_cast<int>(std::lround(h));
        if (heightInt < 1) {
            heightInt = 1;
        }
    }
    out.pose.box = cv::Rect(0, 0, 0, heightInt);

    const bool haveSeventeen = (out.pose.keypoints.size() == 17);
    const bool hipOk =
        pose_features::hipCenter(out.pose.keypoints, kConfForHeight).has_value();
    const bool heightOk = heightEstimateIsValid(out.estimated_bbox_height);

    out.valid_pose = haveSeventeen && hipOk && heightOk;

    return out;
}
