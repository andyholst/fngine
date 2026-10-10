add_library(fngine_kernels_qsa STATIC
  attention.cu
  indexer.cu
  kv_append.cu
  qk_norm_rope.cu
)
fngine_cuda_target(fngine_kernels_qsa)
fngine_internal_includes(fngine_kernels_qsa)
target_link_libraries(fngine_kernels_qsa PUBLIC fngine_core)
