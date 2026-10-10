fngine_add_test(test_pack_load SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_pack_load.cpp
  LIBRARIES fngine_model fngine_io fngine_core)
fngine_add_test(test_expert_swap SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_expert_swap.cpp
  LIBRARIES fngine_model fngine_io fngine_core ggml_iq fngine::json ORACLE)
fngine_add_test(test_routing_stats SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_routing_stats.cu
  LIBRARIES fngine_model fngine_core)
fngine_add_test(test_prefill_stream SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_prefill_stream.cpp
  LIBRARIES fngine_model fngine_io fngine_core)
add_test(NAME test_prefill_stream_pageable COMMAND test_prefill_stream --pageable)
set_tests_properties(test_prefill_stream_pageable PROPERTIES SKIP_RETURN_CODE 77)
fngine_add_test(test_elastic_experts SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_elastic_experts.cpp
  LIBRARIES fngine_model fngine_io fngine_core)
fngine_add_test(test_context_capacity SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_context_capacity.cpp
  LIBRARIES fngine_model fngine_io fngine_core)
