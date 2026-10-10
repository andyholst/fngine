fngine_add_test(kernels_ple SOURCES ${CMAKE_CURRENT_LIST_DIR}/ple/test_ple.cpp
  LIBRARIES fngine_kernels_ple ggml_iq ORACLE)
