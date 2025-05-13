cmake_minimum_required(VERSION 3.10)
project(twinlite_infer)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to specify ONNXRUNTIME_ROOT if not in standard locations
if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(STATUS "ONNXRUNTIME_ROOT not set, using /usr/local or default system paths")
    set(ONNXRUNTIME_ROOT "/home/ridebuddy/Documents/libraries/onnxruntime-linux-x64-1.21.0")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include
        /usr/local/include
        /usr/include
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /usr/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not find ONNX Runtime. Please set ONNXRUNTIME_ROOT to your install location.")
endif()

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIBRARY}")

# Include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# Source
add_executable(twinlite_infer twinlite_infer.cpp)

target_link_libraries(twinlite_infer
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
)
