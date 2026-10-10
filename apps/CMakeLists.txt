add_executable(fngine-probe fngine_probe.cu)
fngine_cuda_target(fngine-probe)
fngine_internal_includes(fngine-probe)
target_compile_options(fngine-probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=${FNGINE_CPU_ARCH}>)
target_link_libraries(fngine-probe PRIVATE CUDA::cudart Threads::Threads PkgConfig::LIBURING)

# Teacher-forced validation against the Hugging Face golden data.
add_executable(fngine-validate validate/main.cpp)
fngine_internal_includes(fngine-validate)
target_link_libraries(fngine-validate PRIVATE fngine_model fngine::json spdlog::spdlog)

# Engine-independent option parsing / report code (linked by the apps and their tests).
add_library(fngine_cli_options STATIC cli/options.cpp)
fngine_internal_includes(fngine_cli_options)
target_include_directories(fngine_cli_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cli)
target_link_libraries(fngine_cli_options PUBLIC fngine_serve_protocol)

add_library(fngine_bench_support STATIC bench/fngine_bench_support.cpp)
fngine_internal_includes(fngine_bench_support)
target_include_directories(fngine_bench_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(fngine_bench_support PUBLIC fngine_serve_protocol)

# The executables need the Engine implementation.
if(TARGET fngine_engine)
  add_executable(fngine cli/main.cpp)
  fngine_internal_includes(fngine)
  target_link_libraries(fngine PRIVATE fngine_cli_options fngine_engine)

  add_executable(fngine-serve serve/main.cpp)
  fngine_internal_includes(fngine-serve)
  target_link_libraries(fngine-serve PRIVATE fngine_serve)

  if(FNGINE_BUILD_BENCH)
    add_executable(fngine-bench bench/fngine_bench.cpp)
    fngine_internal_includes(fngine-bench)
    target_link_libraries(fngine-bench PRIVATE fngine_bench_support fngine_engine CUDA::cudart)
  endif()
endif()
