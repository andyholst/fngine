# CPU expert path tests (owner: CpuIq). Real-blob parts need FNGINE_TEST_PACK.
fngine_add_test(cpu_q8_quant SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_q8_quant.cpp
  LIBRARIES fngine_cpu ggml_iq ORACLE)
fngine_add_test(cpu_iq_kernels SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_iq_kernels.cpp
  LIBRARIES fngine_cpu ggml_iq fngine::json ORACLE)
fngine_add_test(cpu_expert_pool SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_expert_pool.cpp
  LIBRARIES fngine_cpu ggml_iq fngine::json ORACLE)
fngine_add_test(cpu_doorbell SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_doorbell.cu
  LIBRARIES fngine_cpu ggml_iq)
