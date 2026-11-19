cmake_minimum_required(VERSION 3.28)
project(pyfastnoise2 LANGUAGES CXX)

# FastNoise2 v1.1.1. A new pin needs the matching version in pyproject.toml.
set(FASTNOISE2_COMMIT 903c1f2d2f9d53ddce94cd223f32727d9ab3aeaa)

option(PYFASTNOISE2_ALLOW_GCC "Build with GCC, whose FastNoise2 builds are about 1.8x slower" OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT PYFASTNOISE2_ALLOW_GCC)
  message(FATAL_ERROR
    "pyfastnoise2: the C++ compiler is GCC (${CMAKE_CXX_COMPILER}). FastNoise2 built with GCC "
    "runs about 1.8x slower than built with Clang. Select Clang, for example with the "
    "environment variable CXX=clang++, or accept the slower library with "
    "-DPYFASTNOISE2_ALLOW_GCC=ON (pip: -C cmake.define.PYFASTNOISE2_ALLOW_GCC=ON).")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(BUILD_SHARED_LIBS ON)
# Cache entries: FastNoise2 declares these with option(), which ignores normal variables
# under the policies of its minimum CMake version.
set(FASTNOISE2_TOOLS OFF CACHE BOOL "" FORCE)
set(FASTNOISE2_TESTS OFF CACHE BOOL "" FORCE)
set(FASTNOISE2_UTILITY OFF CACHE BOOL "" FORCE)
# Bit-identical results on every SIMD level and with every compiler.
set(FASTNOISE2_STRICT_FP ON CACHE BOOL "" FORCE)

include(FetchContent)
# EXCLUDE_FROM_ALL drops FastNoise2's own install rules (headers, CMake package files).
FetchContent_Declare(
  FastNoise2
  GIT_REPOSITORY https://github.com/Auburn/FastNoise2.git
  GIT_TAG ${FASTNOISE2_COMMIT}
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(FastNoise2)

# The wheel ships LICENSES/ for the code compiled into the library.
foreach(dependency IN ITEMS FastNoise2 FastSIMD)
  file(SHA256 "${${dependency}_SOURCE_DIR}/LICENSE" fetched_license)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/${dependency}-LICENSE" shipped_license)
  if(NOT fetched_license STREQUAL shipped_license)
    message(FATAL_ERROR
      "LICENSES/${dependency}-LICENSE differs from the LICENSE of the ${dependency} sources "
      "at ${${dependency}_SOURCE_DIR}; copy the new text.")
  endif()
endforeach()

# The binding loads the library by its release name, next to the Python sources.
set_target_properties(FastNoise PROPERTIES EXCLUDE_FROM_ALL FALSE DEBUG_POSTFIX "")
install(TARGETS FastNoise LIBRARY DESTINATION pyfastnoise2)
