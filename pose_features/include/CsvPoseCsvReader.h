#pragma once

#include "CsvPoseAdapter.h"

#include <fstream>
#include <string>
#include <vector>

// Fixed-schema CSV reader for pose rows (no quotes, comma-separated).
// Header line: video_name, frame_index, timestamp, person_id,
//              kpt0_x, kpt0_y, kpt0_conf, ..., kpt16_x, kpt16_y, kpt16_conf

namespace csv_pose {

constexpr int kNumKeypoints = 17;
// 4 metadata columns + 17 * (x, y, conf)
constexpr int kExpectedColumnCount = 4 + kNumKeypoints * 3;

// Reads one input file. The first line is treated as a header and skipped.
// Float cells may be empty or "nan" (any case) -> quiet NaN.
// On success returns true and clears/replaces `rows`. On failure returns false
// and sets `error` if non-null.
bool loadPoseCsv(const std::string& path,
                 std::vector<CsvPoseRow>& rows,
                 std::string* error = nullptr);

// Stream one row at a time (lower memory use than loadPoseCsv).
class CsvPoseCsvReader {
public:
    explicit CsvPoseCsvReader(const std::string& path);

    bool isOpen() const;

    // Read the first line (header). Call once before readRow.
    // Optionally stores the raw header string in `headerOut`.
    bool readHeader(std::string* headerOut = nullptr);

    // Read the next data line into `row`. Returns false at EOF or on read error.
    // Empty lines are still parsed (missing cells become NaN for floats).
    bool readRow(CsvPoseRow& row);

private:
    std::ifstream in_;
};

} // namespace csv_pose
