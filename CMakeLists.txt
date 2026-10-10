# Modified for fngine from cinference:CMakeLists.txt @ 383e5db. See NOTICE.
cmake_minimum_required(VERSION 3.28)

# fngine is compiled only for sm_120a (RTX 5090). Set and validate the
# architecture before CUDA compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "fngine supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(fngine LANGUAGES C CXX CUDA)

option(FNGINE_BUILD_TESTS "Build fngine tests" OFF)
option(FNGINE_BUILD_BENCH "Build fngine-bench" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS fngine_link=1)
  set(CMAKE_JOB_POOL_LINK fngine_link)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(apps)
if(FNGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
