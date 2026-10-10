# Dense FP8-row and BF16 projections with fused epilogues (see linear.h).
add_library(fngine_kernels_linear STATIC
  linear_fp8.cpp
  linear_bf16.cu
  fp8/fp8_a8.cu
  fp8/shapes/n16384_k2560.cu
  fp8/shapes/n13312_k2560.cu
  fp8/shapes/n2560_k6144.cu
  fp8/shapes/n1280_k2560.cu
  fp8/shapes/n2560_k640.cu
  fp8/shapes/n320_k10240.cu
  fp8/shapes/n10240_k384.cu
  fp8/shapes/n10240_k2560.cu
  fp8/shapes/n2560_k2560.cu
  fp8/shapes/n248320_k2560.cu
  fp8/shapes/n32768_k2560.cu
)
fngine_cuda_target(fngine_kernels_linear)
fngine_internal_includes(fngine_kernels_linear)
target_link_libraries(fngine_kernels_linear PUBLIC fngine_core CUDA::cublasLt)
