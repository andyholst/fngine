add_library(fngine_kernels_gdn STATIC
  storage.cpp
  conv.cu
  norm.cu
  recurrent.cu
  prefill.cu
)
fngine_cuda_target(fngine_kernels_gdn)
fngine_internal_includes(fngine_kernels_gdn)
target_link_libraries(fngine_kernels_gdn PUBLIC fngine_core)
