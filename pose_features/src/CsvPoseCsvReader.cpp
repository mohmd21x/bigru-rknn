#include "CsvPoseCsvReader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>

namespace csv_pose {
namespace {

std::string trim(const std::string& s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) {
        ++a;
    }
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
        --b;
    }
    return s.substr(a, b - a);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

bool equalsIgnoreCase(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

float parseFloatCell(const std::string& raw) {
    const std::string s = trim(raw);
    if (s.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (equalsIgnoreCase(s, "nan")) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    try {
        size_t idx = 0;
        float v = std::stof(s, &idx);
        if (idx != s.size()) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return v;
    } catch (...) {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

double parseDoubleCell(const std::string& raw) {
    const std::string s = trim(raw);
    if (s.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (equalsIgnoreCase(s, "nan")) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    try {
        size_t idx = 0;
        double v = std::stod(s, &idx);
        if (idx != s.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return v;
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

int parseIntCell(const std::string& raw) {
    const std::string s = trim(raw);
    if (s.empty()) {
        return 0;
    }
    if (equalsIgnoreCase(s, "nan")) {
        return 0;
    }
    try {
        size_t idx = 0;
        long long v = std::stoll(s, &idx, 10);
        if (idx != s.size()) {
            return 0;
        }
        return static_cast<int>(v);
    } catch (...) {
        return 0;
    }
}

void cellsToCsvPoseRow(const std::vector<std::string>& cells, CsvPoseRow& row) {
    if (cells.size() > 0) {
        row.video_name = cells[0];
    }
    if (cells.size() > 1) {
        row.frame_index = parseIntCell(cells[1]);
    }
    if (cells.size() > 2) {
        row.timestamp = parseDoubleCell(cells[2]);
    }
    if (cells.size() > 3) {
        row.person_id = parseIntCell(cells[3]);
    }

    for (int i = 0; i < kNumKeypoints; ++i) {
        const int base = 4 + i * 3;
        float x = std::numeric_limits<float>::quiet_NaN();
        float y = std::numeric_limits<float>::quiet_NaN();
        float c = std::numeric_limits<float>::quiet_NaN();
        if (static_cast<int>(cells.size()) > base + 0) {
            x = parseFloatCell(cells[static_cast<size_t>(base + 0)]);
        }
        if (static_cast<int>(cells.size()) > base + 1) {
            y = parseFloatCell(cells[static_cast<size_t>(base + 1)]);
        }
        if (static_cast<int>(cells.size()) > base + 2) {
            c = parseFloatCell(cells[static_cast<size_t>(base + 2)]);
        }
        row.kpt_x[static_cast<size_t>(i)] = x;
        row.kpt_y[static_cast<size_t>(i)] = y;
        row.kpt_conf[static_cast<size_t>(i)] = c;
    }
}

} // namespace

CsvPoseCsvReader::CsvPoseCsvReader(const std::string& path) : in_(path) {}

bool CsvPoseCsvReader::isOpen() const {
    return in_.is_open();
}

bool CsvPoseCsvReader::readHeader(std::string* headerOut) {
    std::string line;
    if (!std::getline(in_, line)) {
        return false;
    }
    if (headerOut != nullptr) {
        *headerOut = line;
    }
    return true;
}

bool CsvPoseCsvReader::readRow(CsvPoseRow& row) {
    std::string line;
    if (!std::getline(in_, line)) {
        return false;
    }

    std::vector<std::string> cells = splitCsvLine(line);
    if (static_cast<int>(cells.size()) < kExpectedColumnCount) {
        cells.resize(static_cast<size_t>(kExpectedColumnCount));
    }

    row = CsvPoseRow{};
    cellsToCsvPoseRow(cells, row);
    return true;
}

bool loadPoseCsv(const std::string& path,
                 std::vector<CsvPoseRow>& rows,
                 std::string* error) {
    rows.clear();

    CsvPoseCsvReader reader(path);
    if (!reader.isOpen()) {
        if (error != nullptr) {
            *error = "cannot open file: " + path;
        }
        return false;
    }

    if (!reader.readHeader()) {
        if (error != nullptr) {
            *error = "missing or empty header line";
        }
        return false;
    }

    CsvPoseRow row;
    while (reader.readRow(row)) {
        rows.push_back(row);
    }

    return true;
}

} // namespace csv_pose
