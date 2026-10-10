add_library(fngine_kernels_head STATIC
  argmax.cu
  embed_gather.cu
  mtp_round.cu
  sampling.cu
  speculative_round.cu
)
fngine_cuda_target(fngine_kernels_head)
fngine_internal_includes(fngine_kernels_head)
target_link_libraries(fngine_kernels_head PUBLIC fngine_core)
