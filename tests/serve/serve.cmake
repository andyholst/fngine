# Serving-layer, CLI and bench tests (engine-independent).
fngine_add_test(test_openai_schema
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_openai_schema.cpp LIBRARIES fngine_serve_protocol)
fngine_add_test(test_serve_options
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_serve_options.cpp LIBRARIES fngine_serve_protocol)
fngine_add_test(test_http_transport
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_http_transport.cpp LIBRARIES fngine_serve_protocol)
fngine_add_test(test_cli_options
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_cli_options.cpp LIBRARIES fngine_cli_options)
fngine_add_test(test_bench_support
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_bench_support.cpp LIBRARIES fngine_bench_support)
