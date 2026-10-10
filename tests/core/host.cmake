fngine_add_test(test_host_memory SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_host_memory.cpp
  LIBRARIES fngine_core fngine_io)
fngine_add_test(test_cpu_topology SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_cpu_topology.cpp
  LIBRARIES fngine_core)
