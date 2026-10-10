# fngine_model: config, dense weights, expert placement, per-sequence state, workspace, the
# forward composition and the runtime that owns them.
set(FNGINE_MODEL_SOURCES
  adaptive_experts.cpp
  config.cpp
  forward.cpp
  placement.cpp
  l2_prefetch.cu
  prefill_stream.cpp
  routing_counts.cu
  runtime.cpp
  state.cpp
  step_graph.cpp
  step_kernels.cu
  weights.cpp
  workspace.cpp
)

add_library(fngine_model STATIC ${FNGINE_MODEL_SOURCES})
fngine_cuda_target(fngine_model)
fngine_internal_includes(fngine_model)
target_link_libraries(fngine_model PUBLIC fngine_core fngine_io fngine_kernels fngine_cpu spdlog::spdlog)
