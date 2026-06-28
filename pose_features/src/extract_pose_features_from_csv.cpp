// extract_pose_features_from_csv
// Reads pose CSV (header + rows), appends single-frame features, writes output CSV.
// Usage:
//   extract_pose_features_from_csv <input.csv> <output.csv>
//   extract_pose_features_from_csv <input_directory> <output_directory>
//
// Output columns (in order):
//   video_name, frame_index, timestamp, person_id,
//   norm_kpt0_x, norm_kpt0_y, ..., norm_kpt16_x, norm_kpt16_y,
//   dist_*, angle_* (joint), torso_angle, torso_angle_std, angle_change, torso_angular_velocity, hip_height, min_hip_height_over_window, hip_height_change, bbox_aspect_ratio,
//   vel_hip_vx, vel_hip_vy, rolling_mean_vertical_velocity, acceleration, vel_hip_speed, vel_max_wrist_speed, vel_max_ankle_speed,
//   valid_prev_pose,
//   estimated_bbox_height, valid_pose
//
// Per-track previous-row state is kept in memory. When preconditions hold, velocity is computed and
// written after joint angles (vel_* and valid_prev_pose), then helpers (estimated_bbox_height, valid_pose).

#include "CsvPoseAdapter.h"
#include "CsvPoseCsvReader.h"
#include "PoseFeatures.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

using csv_pose::kNumKeypoints;

bool hasCsvExtension(const fs::path& p) {
    const std::string e = p.extension().string();
    return e == ".csv" || e == ".CSV";
}

constexpr int kCsvFloatPrecision = 6;

// Last processed row for each (video_name, person_id), in file order.
// Used later for frame-to-frame velocity; we refresh this after every output row.
struct TrackPrevState {
    PoseResult pose;
    bool valid_pose = false;
    double timestamp = 0.0;
    // From pose_features::hipCenter when hips are confident enough; nullopt otherwise.
    std::optional<cv::Point2f> hip_center;
    // Up to last 5 valid vel_hip_vy values for this track.
    std::deque<float> recent_vy_history;
    // Up to last 5 valid hip_height values for this track.
    std::deque<float> recent_hip_height_history;
    // Up to last 10 valid torso_angle values for this track.
    std::deque<float> recent_torso_angle_history;
    // Previous row's rolling mean vertical velocity for this track.
    std::optional<float> last_rolling_mean_vy;
};

using TrackKey = std::pair<std::string, int>; // (video_name, person_id)
constexpr std::size_t kRollingVyWindow = 5;
constexpr std::size_t kHipHeightWindow = 5;
constexpr std::size_t kTorsoAngleWindow = 10;

void writeCsvEscaped(std::ostream& os, const std::string& s) {
    os << s;
}

std::ostream& writeFloatCell(std::ostream& os, float v) {
    if (std::isnan(v)) {
        os << "nan";
    } else {
        os << std::fixed << std::setprecision(kCsvFloatPrecision) << v;
    }
    return os;
}

std::ostream& writeDoubleCell(std::ostream& os, double v) {
    if (std::isnan(v)) {
        os << "nan";
    } else {
        os << std::fixed << std::setprecision(kCsvFloatPrecision) << v;
    }
    return os;
}

void writeOutputHeader(std::ostream& os) {
    os << "video_name,frame_index,timestamp,person_id";

    for (int i = 0; i < kNumKeypoints; ++i) {
        os << ",norm_kpt" << i << "_x,norm_kpt" << i << "_y";
    }

    os << ",dist_shoulder_width"
          ",dist_hip_width"
          ",dist_nose_to_hip"
          ",dist_left_thigh"
          ",dist_right_thigh"
          ",dist_left_shin"
          ",dist_right_shin"
          ",dist_hand_to_hand"
          ",dist_left_hand_to_hip"
          ",dist_right_hand_to_hip";

    os << ",angle_left_hip"
          ",angle_right_hip"
          ",angle_left_knee"
          ",angle_right_knee"
          ",torso_angle"
          ",torso_angle_std"
          ",angle_change"
          ",torso_angular_velocity"
          ",hip_height"
          ",min_hip_height_over_window"
          ",hip_height_change"
          ",bbox_aspect_ratio";

    os << ",vel_hip_vx"
          ",vel_hip_vy"
          ",rolling_mean_vertical_velocity"
          ",acceleration"
          ",vel_hip_speed"
          ",vel_max_wrist_speed"
          ",vel_max_ankle_speed"
          ",valid_prev_pose"
          ",estimated_bbox_height,valid_pose\n";
}

// 34 norm + 10 dist + 4 joint angles + torso_angle + torso_angle_std +
// angle_change + torso_angular_velocity +
// hip_height + min_hip_height_over_window + hip_height_change + bbox_aspect_ratio
// (invalid pose: no single-frame features).
void writeNanSingleFrameCells(std::ostream& os) {
    const int normCount = kNumKeypoints * 2;
    for (int i = 0; i < normCount; ++i) {
        os << ",nan";
    }
    for (int i = 0; i < 10 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1; ++i) {
        os << ",nan";
    }
}

void appendOptionalAngle(std::ostream& os, const std::optional<float>& ang) {
    os << ',';
    if (ang.has_value()) {
        writeFloatCell(os, *ang);
    } else {
        os << "nan";
    }
}

// Seven velocity columns + valid_prev_pose when velocity could not be computed.
void writeNanVelocityAndValidPrev(std::ostream& os) {
    os << ",nan,nan,nan,nan,nan,nan,nan,0";
}

// Writes vel_* from computeVelocityFeatures when present; otherwise NaN and valid_prev_pose 0.
void writeVelocityColumns(std::ostream& os,
                          const std::optional<pose_features::VelocityFeatures>& vf,
                          const std::optional<float>& rollingMeanVy,
                          const std::optional<float>& acceleration) {
    if (vf.has_value()) {
        const pose_features::VelocityFeatures& v = *vf;
        os << ',';
        writeFloatCell(os, v.vxH);
        os << ',';
        writeFloatCell(os, v.vyH);
        os << ',';
        if (rollingMeanVy.has_value()) {
            writeFloatCell(os, *rollingMeanVy);
        } else {
            os << "nan";
        }
        os << ',';
        if (acceleration.has_value()) {
            writeFloatCell(os, *acceleration);
        } else {
            os << "nan";
        }
        os << ',';
        writeFloatCell(os, v.speed);
        os << ',';
        writeFloatCell(os, v.handMaxSpeedH);
        os << ',';
        writeFloatCell(os, v.footMaxSpeedH);
        os << ',';
        os << '1';
    } else {
        writeNanVelocityAndValidPrev(os);
    }
}

// Snapshot current row into previous-state for this track (for a future velocity step).
TrackPrevState makeTrackPrevState(const CsvPoseAdaptation& adapt,
                                  double timestamp,
                                  std::deque<float> recentVyHistory,
                                  std::deque<float> recentHipHeightHistory,
                                  std::deque<float> recentTorsoAngleHistory,
                                  std::optional<float> rollingMeanVy) {
    TrackPrevState s;
    s.pose = adapt.pose;
    s.valid_pose = adapt.valid_pose;
    s.timestamp = timestamp;
    s.hip_center = pose_features::hipCenter(adapt.pose.keypoints);
    s.recent_vy_history = std::move(recentVyHistory);
    s.recent_hip_height_history = std::move(recentHipHeightHistory);
    s.recent_torso_angle_history = std::move(recentTorsoAngleHistory);
    s.last_rolling_mean_vy = rollingMeanVy;
    return s;
}

// Collects *.csv / *.CSV in one directory (non-recursive), sorted by path.
std::vector<fs::path> collectCsvFilesInDirectory(const fs::path& dir) {
    std::vector<fs::path> csvFiles;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (hasCsvExtension(entry.path())) {
            csvFiles.push_back(entry.path());
        }
    }
    std::sort(csvFiles.begin(), csvFiles.end());
    return csvFiles;
}

// Output name: <stem>_features.csv  e.g. foo_keypoints.csv -> foo_keypoints_features.csv
fs::path makeOutputPathForInputCsv(const fs::path& inputCsvFile, const fs::path& outputDir) {
    const std::string name = inputCsvFile.stem().string() + "_features.csv";
    return outputDir / name;
}

// Core pipeline: one input CSV -> one output CSV (same logic as original single-file main).
// If printSummary is true, prints row counts and output path (single-file mode).
bool processPoseFeaturesFile(const std::string& inputPathStr,
                             const std::string& outputPath,
                             bool printSummary) {
    csv_pose::CsvPoseCsvReader reader(inputPathStr);
    if (!reader.isOpen()) {
        std::cerr << "Error: cannot open input file: " << inputPathStr << '\n';
        return false;
    }

    if (!reader.readHeader()) {
        std::cerr << "Error: missing header line in input file: " << inputPathStr << '\n';
        return false;
    }

    std::ofstream out(outputPath);
    if (!out) {
        std::cerr << "Error: cannot open output file: " << outputPath << '\n';
        return false;
    }

    out << std::fixed;
    writeOutputHeader(out);

    std::size_t totalRows = 0;
    std::size_t validPoseRows = 0;
    std::size_t invalidRows = 0;

    std::map<TrackKey, TrackPrevState> trackPrev;

    CsvPoseRow row{};
    while (reader.readRow(row)) {
        const CsvPoseAdaptation adapt = adaptCsvRowToPose(row);
        const TrackKey key = std::make_pair(row.video_name, row.person_id);

        ++totalRows;
        if (adapt.valid_pose) {
            ++validPoseRows;
        } else {
            ++invalidRows;
        }

        std::optional<pose_features::VelocityFeatures> velocity_features;
        std::optional<float> rolling_mean_vertical_velocity;
        std::optional<float> acceleration;
        std::deque<float> next_vy_history;
        std::deque<float> next_hip_height_history;
        std::deque<float> next_torso_angle_history;
        if (const auto prevIt = trackPrev.find(key); prevIt != trackPrev.end()) {
            const TrackPrevState& prev = prevIt->second;
            next_vy_history = prev.recent_vy_history;
            next_hip_height_history = prev.recent_hip_height_history;
            next_torso_angle_history = prev.recent_torso_angle_history;
            const std::optional<cv::Point2f> currHip =
                pose_features::hipCenter(adapt.pose.keypoints);
            const double dt = row.timestamp - prev.timestamp;
            const bool can_compute_velocity =
                prev.valid_pose && adapt.valid_pose && prev.hip_center.has_value() &&
                currHip.has_value() && std::isfinite(dt) && dt > 0.0;
            if (can_compute_velocity) {
                velocity_features = pose_features::computeVelocityFeatures(
                    prev.pose,
                    adapt.pose,
                    *prev.hip_center,
                    *currHip,
                    static_cast<float>(dt));
            }
        }
        if (velocity_features.has_value()) {
            next_vy_history.push_back(velocity_features->vyH);
            if (next_vy_history.size() > kRollingVyWindow) {
                next_vy_history.pop_front();
            }
            if (!next_vy_history.empty()) {
                const float sum = std::accumulate(next_vy_history.begin(),
                                                  next_vy_history.end(),
                                                  0.0f);
                rolling_mean_vertical_velocity =
                    sum / static_cast<float>(next_vy_history.size());
            }
        }
        if (const auto prevIt = trackPrev.find(key); prevIt != trackPrev.end()) {
            const TrackPrevState& prev = prevIt->second;
            const float dt = static_cast<float>(row.timestamp - prev.timestamp);
            if (rolling_mean_vertical_velocity.has_value() &&
                prev.last_rolling_mean_vy.has_value() &&
                std::isfinite(dt) &&
                dt > 0.0f) {
                acceleration =
                    (*rolling_mean_vertical_velocity - *prev.last_rolling_mean_vy) / dt;
            }
        }

        writeCsvEscaped(out, row.video_name);
        out << ',';
        out << row.frame_index;
        out << ',';
        writeDoubleCell(out, row.timestamp);
        out << ',';
        out << row.person_id;

        if (!adapt.valid_pose) {
            writeNanSingleFrameCells(out);
            writeVelocityColumns(
                out,
                velocity_features,
                rolling_mean_vertical_velocity,
                acceleration);
            out << ',';
            writeFloatCell(out, adapt.estimated_bbox_height);
            out << ',';
            out << (adapt.valid_pose ? '1' : '0');
            out << '\n';

            trackPrev[key] = makeTrackPrevState(
                adapt,
                row.timestamp,
                std::move(next_vy_history),
                std::move(next_hip_height_history),
                std::move(next_torso_angle_history),
                rolling_mean_vertical_velocity);
            continue;
        }

        const std::optional<pose_features::NormalizedPose> normOpt =
            pose_features::computeNormalizedPose(adapt.pose);
        const std::optional<pose_features::PairwiseDistances> distOpt =
            pose_features::computePairwiseDistances(adapt.pose);
        const pose_features::JointAngles angles =
            pose_features::computeJointAngles(adapt.pose.keypoints);
        const std::optional<float> torsoAngle =
            pose_features::computeTorsoAngle(adapt.pose.keypoints);
        std::optional<float> torsoAngleStd;
        if (torsoAngle.has_value()) {
            next_torso_angle_history.push_back(*torsoAngle);
            if (next_torso_angle_history.size() > kTorsoAngleWindow) {
                next_torso_angle_history.pop_front();
            }
            if (next_torso_angle_history.size() >= 2) {
                const float n = static_cast<float>(next_torso_angle_history.size());
                const float mean =
                    std::accumulate(next_torso_angle_history.begin(),
                                    next_torso_angle_history.end(),
                                    0.0f) / n;
                float var = 0.0f;
                for (float a : next_torso_angle_history) {
                    const float d = a - mean;
                    var += d * d;
                }
                torsoAngleStd = std::sqrt(var / n);
            }
        }
        const std::optional<float> hipHeight =
            pose_features::computeHipHeight(adapt.pose);
        std::optional<float> minHipHeightOverWindow;
        if (hipHeight.has_value()) {
            next_hip_height_history.push_back(*hipHeight);
            if (next_hip_height_history.size() > kHipHeightWindow) {
                next_hip_height_history.pop_front();
            }
            if (!next_hip_height_history.empty()) {
                minHipHeightOverWindow =
                    *std::min_element(next_hip_height_history.begin(),
                                      next_hip_height_history.end());
            }
        }
        std::optional<float> angleChange;
        std::optional<float> torsoAngularVelocity;
        std::optional<float> hipHeightChange;
        if (const auto prevIt = trackPrev.find(key); prevIt != trackPrev.end()) {
            const TrackPrevState& prev = prevIt->second;
            const std::optional<float> prevTorsoAngle =
                pose_features::computeTorsoAngle(prev.pose.keypoints);
            const std::optional<float> prevHipHeight =
                pose_features::computeHipHeight(prev.pose);
            const float dt = static_cast<float>(row.timestamp - prev.timestamp);
            angleChange = pose_features::computeAngleChange(torsoAngle, prevTorsoAngle, dt);
            torsoAngularVelocity =
                pose_features::computeTorsoAngularVelocity(angleChange, dt);
            hipHeightChange =
                pose_features::computeHipHeightChange(hipHeight, prevHipHeight, dt);
        }

        if (normOpt.has_value()) {
            for (int i = 0; i < kNumKeypoints * 2; ++i) {
                out << ',';
                writeFloatCell(out, normOpt->values[static_cast<size_t>(i)]);
            }
        } else {
            for (int i = 0; i < kNumKeypoints * 2; ++i) {
                out << ",nan";
            }
        }

        if (distOpt.has_value()) {
            const pose_features::PairwiseDistances& d = *distOpt;
            out << ',';
            writeFloatCell(out, d.shoulderWidthH);
            out << ',';
            writeFloatCell(out, d.hipWidthH);
            out << ',';
            writeFloatCell(out, d.headHipH);
            out << ',';
            writeFloatCell(out, d.leftThighH);
            out << ',';
            writeFloatCell(out, d.rightThighH);
            out << ',';
            writeFloatCell(out, d.leftShinH);
            out << ',';
            writeFloatCell(out, d.rightShinH);
            out << ',';
            writeFloatCell(out, d.handHandH);
            out << ',';
            writeFloatCell(out, d.leftHandHipH);
            out << ',';
            writeFloatCell(out, d.rightHandHipH);
        } else {
            for (int i = 0; i < 10; ++i) {
                out << ",nan";
            }
        }

        appendOptionalAngle(out, angles.leftHip);
        appendOptionalAngle(out, angles.rightHip);
        appendOptionalAngle(out, angles.leftKnee);
        appendOptionalAngle(out, angles.rightKnee);

        appendOptionalAngle(out, torsoAngle);
        appendOptionalAngle(out, torsoAngleStd);
        appendOptionalAngle(out, angleChange);
        appendOptionalAngle(out, torsoAngularVelocity);
        appendOptionalAngle(out, hipHeight);
        appendOptionalAngle(out, minHipHeightOverWindow);
        appendOptionalAngle(out, hipHeightChange);
        appendOptionalAngle(out, pose_features::computeBboxAspectRatioFromKeypoints(adapt.pose));

        writeVelocityColumns(
            out,
            velocity_features,
            rolling_mean_vertical_velocity,
            acceleration);

        out << ',';
        writeFloatCell(out, adapt.estimated_bbox_height);
        out << ',';
        out << (adapt.valid_pose ? '1' : '0');
        out << '\n';

        trackPrev[key] = makeTrackPrevState(
            adapt,
            row.timestamp,
            std::move(next_vy_history),
            std::move(next_hip_height_history),
            std::move(next_torso_angle_history),
            rolling_mean_vertical_velocity);
    }

    if (printSummary) {
        std::cout << "total rows:      " << totalRows << '\n';
        std::cout << "valid_pose rows: " << validPoseRows << '\n';
        std::cout << "invalid rows:    " << invalidRows << '\n';
        std::cout << "output path:     " << outputPath << '\n';
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        const char* prog = (argc > 0) ? argv[0] : "extract_pose_features_from_csv";
        std::cerr << "Usage:\n"
                  << "  " << prog << " <input.csv> <output.csv>\n"
                  << "  " << prog << " <input_directory> <output_directory>\n";
        return 1;
    }

    const fs::path inputPath = argv[1];
    const fs::path outputPathArg = argv[2];
    std::error_code ec;

    if (fs::is_directory(inputPath, ec)) {
        fs::create_directories(outputPathArg, ec);
        if (ec) {
            std::cerr << "Error: could not create output directory: " << outputPathArg.string()
                      << " (" << ec.message() << ")\n";
            return 1;
        }

        const std::vector<fs::path> csvFiles = collectCsvFilesInDirectory(inputPath);
        const std::size_t n = csvFiles.size();
        for (std::size_t i = 0; i < n; ++i) {
            const fs::path& inFile = csvFiles[i];
            const fs::path outFile = makeOutputPathForInputCsv(inFile, outputPathArg);
            std::cout << '[' << (i + 1) << '/' << n << "] processing " << inFile.filename().string()
                      << '\n';
            if (!processPoseFeaturesFile(inFile.string(), outFile.string(), false)) {
                return 1;
            }
        }
        std::cout << "Done processing " << n << " files\n";
        return 0;
    }

    if (!fs::is_regular_file(inputPath, ec)) {
        std::cerr << "Error: input path is not a file or directory: " << inputPath.string() << '\n';
        return 1;
    }

    if (!processPoseFeaturesFile(inputPath.string(), outputPathArg.string(), true)) {
        return 1;
    }

    return 0;
}
