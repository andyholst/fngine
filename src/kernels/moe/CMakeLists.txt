add_library(fngine_kernels_moe STATIC
  moe_decode.cu
  moe_format.cpp
  moe_grouped_gemm.cu
  moe_prefill.cu
)
fngine_cuda_target(fngine_kernels_moe)
fngine_internal_includes(fngine_kernels_moe)
target_link_libraries(fngine_kernels_moe PUBLIC fngine_core)
