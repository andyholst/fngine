# QSA kernels (src/kernels/qsa): FP64-oracle tests and a timing tool.
set(_qsa_dir ${CMAKE_CURRENT_LIST_DIR}/qsa)
fngine_add_test(kernels_qsa_norm_rope SOURCES ${_qsa_dir}/test_qsa_norm_rope.cu
  LIBRARIES fngine_kernels_qsa ORACLE)
fngine_add_test(kernels_qsa_kv_append SOURCES ${_qsa_dir}/test_qsa_kv_append.cu
  LIBRARIES fngine_kernels_qsa ORACLE)
fngine_add_test(kernels_qsa_indexer SOURCES ${_qsa_dir}/test_qsa_indexer.cu
  LIBRARIES fngine_kernels_qsa ORACLE)
fngine_add_test(kernels_qsa_attention SOURCES ${_qsa_dir}/test_qsa_attention.cu
  LIBRARIES fngine_kernels_qsa ORACLE)
fngine_add_test(kernels_qsa_determinism SOURCES ${_qsa_dir}/test_qsa_determinism.cu
  LIBRARIES fngine_kernels_qsa ORACLE)
# Timing tool (not a ctest): build/tests/qsa_bench
add_executable(qsa_bench ${_qsa_dir}/bench_qsa.cu)
fngine_internal_includes(qsa_bench)
target_include_directories(qsa_bench PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(qsa_bench PRIVATE fngine_kernels_qsa)
fngine_cuda_target(qsa_bench)
