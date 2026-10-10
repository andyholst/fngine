# MoE kernel tests (owner: MoeGpu agent).
set(_moe_dir ${CMAKE_CURRENT_LIST_DIR}/moe)
fngine_add_test(kernels_moe_experts ORACLE
  SOURCES ${_moe_dir}/test_moe_experts.cpp
  LIBRARIES fngine_kernels_moe ggml_iq)
fngine_add_test(kernels_moe_exchange ORACLE
  SOURCES ${_moe_dir}/test_moe_exchange.cpp
  LIBRARIES fngine_kernels_moe)
fngine_add_test(kernels_moe_ablate ORACLE
  SOURCES ${_moe_dir}/test_moe_ablate.cpp
  LIBRARIES fngine_kernels_moe)
