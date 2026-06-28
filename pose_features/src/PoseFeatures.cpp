#include "PoseFeatures.h"
#include <cmath>

namespace pose_features {

namespace {

inline float torsoSideAngleDeg(const Keypoint& shoulder, const Keypoint& hip) {
    float dx = std::abs(shoulder.x - hip.x);
    float dy = std::abs(shoulder.y - hip.y);
    if (dy == 0.0f) {
        return 90.0f;
    }
    return std::atan2(dx, dy) * 180.0f / static_cast<float>(CV_PI);
}

inline float safeHeight(const PoseResult& pose) {
    float h = static_cast<float>(pose.box.height);
    if (h < 1.0f) h = 1.0f;
    return h;
}

inline float dist(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline std::optional<float> angleAtJoint(const cv::Point2f& a,
                                         const cv::Point2f& b,
                                         const cv::Point2f& c) {
    // Angle at point b formed by segments ba and bc.
    cv::Point2f v1 = a - b;
    cv::Point2f v2 = c - b;

    float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    if (len1 < 1e-5f || len2 < 1e-5f) {
        return std::nullopt;
    }

    float dot = v1.x * v2.x + v1.y * v2.y;
    float cosTheta = dot / (len1 * len2);
    // Clamp to valid range to avoid NaNs from numerical noise.
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    float theta = std::acos(cosTheta); // radians
    return theta * 180.0f / static_cast<float>(CV_PI);
}

inline std::optional<cv::Point2f> pointIfConf(const Keypoint& kp, float minConf) {
    if (kp.conf < minConf) return std::nullopt;
    return cv::Point2f(kp.x, kp.y);
}

} // namespace

std::optional<float> computeTorsoAngle(const std::vector<Keypoint>& kpts, float minConf) {
    if (kpts.size() != 17) {
        return std::nullopt;
    }

    // Per side: shoulder and hip must both satisfy conf > minConf (default minConf = 0.5 → legacy FallDetector).
    const bool leftOk =
        (kpts[5].conf > minConf && kpts[11].conf > minConf);
    const bool rightOk =
        (kpts[6].conf > minConf && kpts[12].conf > minConf);

    if (!leftOk && !rightOk) {
        return std::nullopt;
    }
    if (leftOk && rightOk) {
        return 0.5f * (torsoSideAngleDeg(kpts[5], kpts[11]) +
                       torsoSideAngleDeg(kpts[6], kpts[12]));
    }
    if (leftOk) {
        return torsoSideAngleDeg(kpts[5], kpts[11]);
    }
    return torsoSideAngleDeg(kpts[6], kpts[12]);
}

std::optional<float> computeAngleChange(const std::optional<float>& currentTorsoAngle,
                                        const std::optional<float>& previousTorsoAngle,
                                        float dt) {
    if (!currentTorsoAngle.has_value() || !previousTorsoAngle.has_value()) {
        return std::nullopt;
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return std::nullopt;
    }
    return *currentTorsoAngle - *previousTorsoAngle;
}

std::optional<float> computeTorsoAngularVelocity(const std::optional<float>& angleChange,
                                                 float dt) {
    if (!angleChange.has_value()) {
        return std::nullopt;
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return std::nullopt;
    }
    return *angleChange / dt;
}

std::optional<float> computeHipHeight(const PoseResult& pose, float minConf) {
    const auto hipOpt = hipCenter(pose.keypoints, minConf);
    if (!hipOpt.has_value()) {
        return std::nullopt;
    }

    const float h = safeHeight(pose);
    if (h <= 0.0f) {
        return std::nullopt;
    }

    return hipOpt->y / h;
}

std::optional<float> computeHipHeightChange(const std::optional<float>& currentHipHeight,
                                            const std::optional<float>& previousHipHeight,
                                            float dt) {
    if (!currentHipHeight.has_value() || !previousHipHeight.has_value()) {
        return std::nullopt;
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return std::nullopt;
    }
    return *currentHipHeight - *previousHipHeight;
}

std::optional<float> computeBboxAspectRatioFromKeypoints(const PoseResult& pose,
                                                         float minConf) {
    const auto& kpts = pose.keypoints;
    if (kpts.empty()) {
        return std::nullopt;
    }

    bool any = false;
    float minX = 0.0f;
    float maxX = 0.0f;
    for (const Keypoint& kp : kpts) {
        if (kp.conf < minConf) {
            continue;
        }
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            continue;
        }
        if (!any) {
            minX = kp.x;
            maxX = kp.x;
            any = true;
        } else {
            minX = std::min(minX, kp.x);
            maxX = std::max(maxX, kp.x);
        }
    }

    if (!any) {
        return std::nullopt;
    }

    float width = maxX - minX;
    if (width < 1.0f) {
        width = 1.0f;
    }

    const float h = safeHeight(pose);
    if (h <= 0.0f) {
        return std::nullopt;
    }

    return width / h;
}

std::optional<cv::Point2f> hipCenter(const std::vector<Keypoint>& kpts,
                                     float minConf) {
    if (kpts.size() <= 12) return std::nullopt;

    bool leftOk  = kpts[11].conf >= minConf;
    bool rightOk = kpts[12].conf >= minConf;

    if (!leftOk && !rightOk) {
        return std::nullopt;
    } else if (leftOk && rightOk) {
        return cv::Point2f(
            0.5f * (kpts[11].x + kpts[12].x),
            0.5f * (kpts[11].y + kpts[12].y)
        );
    } else if (leftOk) {
        return cv::Point2f(kpts[11].x, kpts[11].y);
    } else { // rightOk
        return cv::Point2f(kpts[12].x, kpts[12].y);
    }
}

std::optional<NormalizedPose> computeNormalizedPose(const PoseResult& pose) {
    const auto& kpts = pose.keypoints;
    if (kpts.size() != 17) return std::nullopt;

    auto hipOpt = hipCenter(kpts, 0.3f);
    if (!hipOpt) return std::nullopt;
    cv::Point2f hip = *hipOpt;

    float h = safeHeight(pose);
    if (h <= 0.0f) return std::nullopt;

    NormalizedPose out;
    for (int i = 0; i < 17; ++i) {
        float nx = (kpts[i].x - hip.x) / h;
        float ny = (kpts[i].y - hip.y) / h;
        out.values[2 * i + 0] = nx;
        out.values[2 * i + 1] = ny;
    }
    return out;
}

std::optional<PairwiseDistances> computePairwiseDistances(const PoseResult& pose) {
    const auto& kpts = pose.keypoints;
    if (kpts.size() != 17) return std::nullopt;

    float h = safeHeight(pose);
    if (h <= 0.0f) return std::nullopt;

    PairwiseDistances dists;

    // Helper lambdas to get points if reasonably confident.
    auto p = [&](int idx, float minConf) -> std::optional<cv::Point2f> {
        if (idx < 0 || idx >= static_cast<int>(kpts.size())) return std::nullopt;
        return pointIfConf(kpts[idx], minConf);
    };

    // Shoulder and hip widths.
    auto lShoulder = p(5, 0.3f);
    auto rShoulder = p(6, 0.3f);
    auto lHip      = p(11, 0.3f);
    auto rHip      = p(12, 0.3f);

    if (lShoulder && rShoulder) {
        dists.shoulderWidthH = dist(*lShoulder, *rShoulder) / h;
    }
    if (lHip && rHip) {
        dists.hipWidthH = dist(*lHip, *rHip) / h;
    }

    // Head–hip distance (nose to hip center).
    auto nose = p(0, 0.3f);
    auto hipC = hipCenter(kpts, 0.3f);
    if (nose && hipC) {
        dists.headHipH = dist(*nose, *hipC) / h;
    }

    // Legs: thighs and shins.
    auto lKnee  = p(13, 0.3f);
    auto rKnee  = p(14, 0.3f);
    auto lAnkle = p(15, 0.3f);
    auto rAnkle = p(16, 0.3f);

    if (lHip && lKnee) {
        dists.leftThighH = dist(*lHip, *lKnee) / h;
    }
    if (rHip && rKnee) {
        dists.rightThighH = dist(*rHip, *rKnee) / h;
    }
    if (lKnee && lAnkle) {
        dists.leftShinH = dist(*lKnee, *lAnkle) / h;
    }
    if (rKnee && rAnkle) {
        dists.rightShinH = dist(*rKnee, *rAnkle) / h;
    }

    // Arms / hands.
    auto lWrist = p(9, 0.3f);
    auto rWrist = p(10, 0.3f);
    if (lWrist && rWrist) {
        dists.handHandH = dist(*lWrist, *rWrist) / h;
    }
    if (lWrist && hipC) {
        dists.leftHandHipH = dist(*lWrist, *hipC) / h;
    }
    if (rWrist && hipC) {
        dists.rightHandHipH = dist(*rWrist, *hipC) / h;
    }

    return dists;
}

JointAngles computeJointAngles(const std::vector<Keypoint>& kpts,
                               float minConf) {
    JointAngles angles;
    if (kpts.size() != 17) return angles;

    auto p = [&](int idx) -> std::optional<cv::Point2f> {
        return pointIfConf(kpts[idx], minConf);
    };

    // Left hip: knee(13) - hip(11) - shoulder(5)
    auto lHip   = p(11);
    auto lKnee  = p(13);
    auto lShldr = p(5);
    if (lHip && lKnee && lShldr) {
        angles.leftHip = angleAtJoint(*lKnee, *lHip, *lShldr);
    }

    // Right hip: knee(14) - hip(12) - shoulder(6)
    auto rHip   = p(12);
    auto rKnee  = p(14);
    auto rShldr = p(6);
    if (rHip && rKnee && rShldr) {
        angles.rightHip = angleAtJoint(*rKnee, *rHip, *rShldr);
    }

    // Left knee: hip(11) - knee(13) - ankle(15)
    auto lAnkle = p(15);
    if (lHip && lKnee && lAnkle) {
        angles.leftKnee = angleAtJoint(*lHip, *lKnee, *lAnkle);
    }

    // Right knee: hip(12) - knee(14) - ankle(16)
    auto rAnkle = p(16);
    if (rHip && rKnee && rAnkle) {
        angles.rightKnee = angleAtJoint(*rHip, *rKnee, *rAnkle);
    }

    return angles;
}

VelocityFeatures computeVelocityFeatures(const PoseResult& prevPose,
                                         const PoseResult& currPose,
                                         const cv::Point2f& prevHip,
                                         const cv::Point2f& currHip,
                                         float dt) {
    VelocityFeatures v;
    if (dt <= 0.0f) {
        return v;
    }

    float h = safeHeight(currPose);
    if (h <= 0.0f) {
        return v;
    }

    float dx = currHip.x - prevHip.x;
    float dy = currHip.y - prevHip.y;

    v.vxH = (dx / dt) / h;
    v.vyH = (dy / dt) / h;
    v.speed = std::sqrt(v.vxH * v.vxH + v.vyH * v.vyH);

    // Limb velocities (wrists and ankles).
    auto limbSpeedH = [&](int idx) -> float {
        if (idx < 0 || idx >= 17) return 0.0f;
        const Keypoint& kPrev = prevPose.keypoints[idx];
        const Keypoint& kCurr = currPose.keypoints[idx];
        if (kPrev.conf < 0.3f || kCurr.conf < 0.3f) return 0.0f;
        float lx = kCurr.x - kPrev.x;
        float ly = kCurr.y - kPrev.y;
        float s  = std::sqrt(lx * lx + ly * ly) / (dt * h);
        return s;
    };

    // Wrists.
    float lw = limbSpeedH(9);
    float rw = limbSpeedH(10);
    v.handMaxSpeedH = std::max(lw, rw);

    // Ankles.
    float la = limbSpeedH(15);
    float ra = limbSpeedH(16);
    v.footMaxSpeedH = std::max(la, ra);

    return v;
}

} // namespace pose_features

