# Modified for fngine from cinference:src/serve/CMakeLists.txt @ 383e5db. See NOTICE.
# fngine_serve_protocol: engine-independent OpenAI wire handling, option parsing, transport and
# app helpers (tests and the apps link it).
add_library(fngine_serve_protocol STATIC
  cli_common.cpp
  serve_options.cpp
  request_validation.cpp
  openai_common.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  translate.cpp
  generation_outcome.cpp
  http_transport.cpp
  app_support.cpp
)
fngine_internal_includes(fngine_serve_protocol)
target_include_directories(fngine_serve_protocol PUBLIC
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(fngine_serve_protocol
  PUBLIC fngine_text Threads::Threads fngine::json fngine::httplib spdlog::spdlog)

# fngine_serve: the HTTP server and generation service over the Engine.
if(TARGET fngine_engine)
  add_library(fngine_serve STATIC
    generation_service.cpp
    http_server.cpp
    openai_chat_http.cpp
  )
  fngine_internal_includes(fngine_serve)
  target_link_libraries(fngine_serve PUBLIC fngine_serve_protocol fngine_engine)
endif()
