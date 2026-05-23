cmake_minimum_required(VERSION 3.14)
project(rt_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ====== 查找依赖 ======
find_package(OpenCV REQUIRED)

# ONNX Runtime — 支持 brew / conda / 手动安装 / third_party
# 优先使用项目内 third_party 预编译包
set(ONNXRUNTIME_THIRD_PARTY "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-osx-arm64-1.17.1")

if(EXISTS ${ONNXRUNTIME_THIRD_PARTY})
    set(ONNXRUNTIME_ROOT ${ONNXRUNTIME_THIRD_PARTY})
elseif(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
elseif(DEFINED ENV{CONDA_PREFIX})
    set(ONNXRUNTIME_ROOT $ENV{CONDA_PREFIX})
endif()

# 查找头文件
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/homebrew/include/onnxruntime
)

# 查找库文件
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /opt/homebrew/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found!\n"
        "Set ONNXRUNTIME_ROOT or install via:\n"
        "  brew install onnxruntime\n"
        "  conda install -c conda-forge onnxruntime-cpp\n"
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ====== 构建目标 ======
add_executable(rt_detection
    src/main.cpp
    src/preprocessor.cpp
    src/detector.cpp
    src/postprocessor.cpp
)

target_include_directories(rt_detection PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(rt_detection PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

# macOS: 设置 rpath 以便运行时找到动态库
if(APPLE)
    target_link_libraries(rt_detection PRIVATE "-framework Foundation")
    # ONNX Runtime 动态库路径
    get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
    set_target_properties(rt_detection PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR};${OpenCV_DIR}/../../.."
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR};${OpenCV_DIR}/../../.."
    )
endif()

# 优化编译选项
target_compile_options(rt_detection PRIVATE
    -O2
    -Wall
    -Wextra
    -Wpedantic
)

# 可选：OpenMP 支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rt_detection PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()
