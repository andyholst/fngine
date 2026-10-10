# Linear / GR kernel tests (owner: Linear agent).
set(_linear_dir ${CMAKE_CURRENT_LIST_DIR}/linear)
fngine_add_test(kernels_linear ORACLE
  SOURCES ${_linear_dir}/test_linear.cpp
  LIBRARIES fngine_kernels_linear)
fngine_add_test(kernels_gr ORACLE
  SOURCES ${_linear_dir}/test_gr.cpp
  LIBRARIES fngine_kernels_gr)

# Kernel microbenchmark (not a test): per-shape path sweep, T=1 GB/s, T=4096 TFLOP/s.
add_executable(linear_bench ${_linear_dir}/bench_linear.cpp)
fngine_internal_includes(linear_bench)
target_include_directories(linear_bench PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(linear_bench PRIVATE fngine_kernels_linear)
