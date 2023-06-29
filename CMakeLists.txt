cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)

# defining onnx package
set(ONNX_PACKAGE_VERSION "1.12.1")
set(ONNX_PACKAGE_NAME "onnxruntime-win-x64-${ONNX_PACKAGE_VERSION}")

# downloading and extracting the onnx package
set(package_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_PACKAGE_VERSION}/${ONNX_PACKAGE_NAME}.zip")

# Check if the package directory exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/${ONNX_PACKAGE_NAME}")
    message(STATUS "ONNX package not found. Downloading and extracting...")
    
    # Download and extract the package
    file(DOWNLOAD ${package_url} "${CMAKE_CURRENT_SOURCE_DIR}/external/${ONNX_PACKAGE_NAME}.zip")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_SOURCE_DIR}/external/${ONNX_PACKAGE_NAME}.zip"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external"
    )
    file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/external/${ONNX_PACKAGE_NAME}.zip")
    
else()
    message(STATUS "Package already exists. Skipping download.")
endif()

# this project uses different versions of submodules please reffer to https://github.com/pengzhendong/silero-vad/tree/master/src/cmake
# for onnxruntime there was a hacky solution inspired by https://github.com/developer239/yolo-cmake-opencv-onnx-cpp/tree/feat/windows
if (WIN32 OR MSVC)
    set(CMAKE_PREFIX_PATH "C:/dev/vcpkg/installed/x64-windows/share")
endif ()
project(silero_vad VERSION 0.1)

set(TEST_TARGET ${PROJECT_NAME}_test)
set(TARGET ${PROJECT_NAME})

find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
#set(onnxruntime_BUILD_SHARED_LIB ON)
#add_subdirectory(onnxruntime/cmake)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/${ONNX_PACKAGE_NAME}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ONNXRUNTIME REQUIRED)

# for rnnoise
option(BUILD_FOR_RELEASE "Additional optimizations and steps may be taken for release" ON)
option(BUILD_TESTS "" OFF)
option(BUILD_VST_PLUGIN "If the VST2 plugin should be built" OFF)
option(BUILD_VST3_PLUGIN "If the VST3 plugin should be built" OFF)
option(BUILD_LV2_PLUGIN "If the LV2 plugin should be built" OFF)
option(BUILD_LADSPA_PLUGIN "If the LADSPA plugin should be built" OFF)
option(BUILD_AU_PLUGIN "If the AU plugin should be built (macOS only)" OFF)
option(BUILD_AUV3_PLUGIN "If the AUv3 plugin should be built (macOS only)" OFF)
add_subdirectory(noise-suppression-for-voice)

add_subdirectory(libsamplerate)

add_library(${TARGET} STATIC "vad/vad_model.cc" "vad/onnx_model.cc" "front_vad/denoiser.cc" "front_vad/resampler.cc" "front_vad/sample_queue.cc")
add_executable(${TEST_TARGET} "main.cc")
target_include_directories(${TARGET} PUBLIC .)
target_include_directories(${TARGET} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

target_link_libraries(${TARGET} gflags::gflags glog::glog RnNoise samplerate ${ONNXRUNTIME_LIBRARY})
target_link_libraries(${TEST_TARGET} ${TARGET})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("Compiler is Clang")
    if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
        file(COPY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/out/build/x64-Clang-Release/")
    else()
        file(COPY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/out/build/x64-Clang-Debug/")
    endif()
else()
    message("Compiler is not Clang please switch to clang")
endif()

