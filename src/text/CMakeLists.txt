# Modified for fngine from cinference:src/text/CMakeLists.txt @ 383e5db. See NOTICE.
# fngine_unicode: utf8proc-backed helpers shared with the vendored llama-jinja runtime.
add_library(fngine_unicode STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
fngine_internal_includes(fngine_unicode)
target_compile_definitions(fngine_unicode PUBLIC UTF8PROC_STATIC)

# llama-jinja (third_party, EXCLUDE_FROM_ALL target ninfer_jinja) uses text/unicode.h and nlohmann.
fngine_internal_includes(ninfer_jinja)
target_link_libraries(ninfer_jinja PRIVATE fngine_unicode fngine::json)

add_library(fngine_text STATIC
  jinja.cpp
  tokenizer.cpp
  chat_template.cpp
  prompt_layout.cpp
  tool_call_parser.cpp
  output_session.cpp
  prompt_lookup.cpp
  frontend.cpp
)
fngine_internal_includes(fngine_text)
target_include_directories(fngine_text PUBLIC
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(fngine_text PUBLIC fngine_unicode ninfer_jinja fngine::json)
