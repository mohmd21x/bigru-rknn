cmake_minimum_required(VERSION 3.10)
project(pose_features_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Local include/PoseEstimator.h provides Keypoint/PoseResult for offline CSV tools.
# fall_cpp is only needed for live inference builds (not required here).
set(FALL_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fall_cpp" CACHE PATH
    "Optional fall_cpp checkout for live-inference builds")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(EXISTS "${FALL_CPP_ROOT}/include/PoseEstimator.h")
    include_directories(${FALL_CPP_ROOT}/include)
    message(STATUS "fall_cpp found at ${FALL_CPP_ROOT}")
else()
    message(STATUS "Using local PoseEstimator.h stub (CSV feature extraction only)")
endif()

set(POSE_FEATURES_LIB_SOURCES
    src/PoseFeatures.cpp
    src/CsvPoseAdapter.cpp
    src/CsvPoseCsvReader.cpp
)

add_executable(extract_pose_features_from_csv
    ${POSE_FEATURES_LIB_SOURCES}
    src/extract_pose_features_from_csv.cpp
)

add_executable(test_pose_csv_features
    ${POSE_FEATURES_LIB_SOURCES}
    src/test_pose_csv_features.cpp
)

set(POSE_FEATURES_EXECUTABLES extract_pose_features_from_csv test_pose_csv_features)

foreach(tgt IN LISTS POSE_FEATURES_EXECUTABLES)
    target_link_libraries(${tgt} PRIVATE ${OpenCV_LIBS})
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -O2)
    endif()
endforeach()
