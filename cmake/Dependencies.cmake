# Modified for fngine from cinference:cmake/Dependencies.cmake @ 383e5db. See NOTICE.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(fngine::json INTERFACE IMPORTED GLOBAL)
target_include_directories(fngine::json INTERFACE ${PROJECT_SOURCE_DIR}/third_party)

add_library(fngine::httplib INTERFACE IMPORTED GLOBAL)
target_include_directories(fngine::httplib INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)

# Internal headers: public API in include/, implementation in src/.
function(fngine_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
endfunction()

function(fngine_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
endfunction()

# Host code on the CPU expert path uses AVX-512 F/BW/DQ/VL with VNNI, BF16, VBMI, BITALG and
# VPOPCNTDQ (Zen 4/5, Intel Sapphire Rapids and later). The default compiles for the building
# machine; set FNGINE_CPU_ARCH (a -march value such as znver5) to target another CPU.
set(FNGINE_CPU_ARCH "native" CACHE STRING "-march target for the CPU expert path")
set(FNGINE_CPU_FLAGS -O3 -march=${FNGINE_CPU_ARCH})
# Test oracles: no reassociation or contraction.
set(FNGINE_ORACLE_FLAGS -fno-fast-math -ffp-contract=off)
