// Small test: read a pose CSV and print single-frame features for the first 3 rows.
//
// Usage: test_pose_csv_features <input.csv>

#include "CsvPoseAdapter.h"
#include "CsvPoseCsvReader.h"
#include "PoseFeatures.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>

namespace {

void printOptionalFloat(const char* label, const std::optional<float>& v) {
    std::cout << "  " << label << ": ";
    if (v.has_value()) {
        std::cout << std::fixed << std::setprecision(4) << *v << '\n';
    } else {
        std::cout << "NaN (not available)\n";
    }
}

void printFloatOrNan(const char* label, float v, bool available) {
    std::cout << "  " << label << ": ";
    if (!available) {
        std::cout << "NaN (not available)\n";
        return;
    }
    if (std::isnan(v)) {
        std::cout << "NaN\n";
    } else {
        std::cout << std::fixed << std::setprecision(4) << v << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << (argc > 0 ? argv[0] : "test_pose_csv_features")
                  << " <input.csv>\n";
        return 1;
    }

    const std::string path = argv[1];

    csv_pose::CsvPoseCsvReader reader(path);
    if (!reader.isOpen()) {
        std::cerr << "Error: cannot open file: " << path << '\n';
        return 1;
    }

    if (!reader.readHeader()) {
        std::cerr << "Error: file has no header line.\n";
        return 1;
    }

    constexpr int kMaxRows = 3;
    int processed = 0;

    CsvPoseRow row{};
    while (processed < kMaxRows && reader.readRow(row)) {
        ++processed;
        std::cout << "========== Row " << processed << " of " << kMaxRows
                  << " (data line) ==========\n";

        CsvPoseAdaptation adapt = adaptCsvRowToPose(row);

        std::cout << "  frame_index:           " << row.frame_index << '\n';
        std::cout << "  person_id:             " << row.person_id << '\n';
        std::cout << "  estimated_bbox_height: " << std::fixed << std::setprecision(4)
                  << adapt.estimated_bbox_height << '\n';
        std::cout << "  valid_pose:            " << (adapt.valid_pose ? "true" : "false")
                  << '\n';

        if (!adapt.valid_pose) {
            std::cout << "  (Skipping pose features: valid_pose is false)\n\n";
            continue;
        }

        const std::optional<pose_features::NormalizedPose> normOpt =
            pose_features::computeNormalizedPose(adapt.pose);
        const std::optional<pose_features::PairwiseDistances> distOpt =
            pose_features::computePairwiseDistances(adapt.pose);
        const pose_features::JointAngles angles =
            pose_features::computeJointAngles(adapt.pose.keypoints);

        std::cout << "  Sample normalized coords (hip-centered, / height):\n";
        std::cout << std::fixed << std::setprecision(6);
        if (normOpt.has_value()) {
            const auto& v = normOpt->values;
            // kpt i -> indices 2*i, 2*i+1
            std::cout << "    norm_kpt0_x:  " << v[0] << '\n';
            std::cout << "    norm_kpt0_y:  " << v[1] << '\n';
            std::cout << "    norm_kpt11_x: " << v[22] << '\n';
            std::cout << "    norm_kpt11_y: " << v[23] << '\n';
        } else {
            std::cout << "    norm_kpt0_x:  NaN (computeNormalizedPose returned no value)\n";
            std::cout << "    norm_kpt0_y:  NaN (computeNormalizedPose returned no value)\n";
            std::cout << "    norm_kpt11_x: NaN (computeNormalizedPose returned no value)\n";
            std::cout << "    norm_kpt11_y: NaN (computeNormalizedPose returned no value)\n";
        }

        std::cout << "  Pairwise distances (normalized by height):\n";
        if (distOpt.has_value()) {
            const pose_features::PairwiseDistances& d = *distOpt;
            printFloatOrNan("dist_shoulder_width", d.shoulderWidthH, true);
            printFloatOrNan("dist_hip_width", d.hipWidthH, true);
        } else {
            printFloatOrNan("dist_shoulder_width", 0.0f, false);
            printFloatOrNan("dist_hip_width", 0.0f, false);
        }

        std::cout << "  Joint angles (degrees):\n";
        printOptionalFloat("angle_left_hip", angles.leftHip);
        printOptionalFloat("angle_right_knee", angles.rightKnee);

        std::cout << '\n';
    }

    if (processed == 0) {
        std::cout << "No data rows read (file may be empty after header).\n";
    }

    return 0;
}
