# Gated-residual stream norm + inject (see gr.h).
add_library(fngine_kernels_gr STATIC gr.cu)
fngine_cuda_target(fngine_kernels_gr)
fngine_internal_includes(fngine_kernels_gr)
target_link_libraries(fngine_kernels_gr PUBLIC fngine_core)
