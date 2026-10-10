add_library(fngine_core STATIC
  arena.cu
  cpu_topology.cpp
  decode_graph.cpp
  device.cu
  dtype.cpp
  exchange.cpp
  host_memory.cpp
  tensor.cpp
)
fngine_cuda_target(fngine_core)
fngine_internal_includes(fngine_core)
target_include_directories(fngine_core PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fngine_core PUBLIC CUDA::cudart Threads::Threads spdlog::spdlog)
