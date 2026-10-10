# Text frontend tests. Tokenizer/template/session tests need FNGINE_TEST_PACK (pack tokenizer/).
fngine_add_test(test_tokenizer
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_tokenizer.cpp LIBRARIES fngine_text)
fngine_add_test(test_chat_template
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp LIBRARIES fngine_text)
fngine_add_test(text_output_session
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_output_session.cpp LIBRARIES fngine_text)
fngine_add_test(text_tool_call_parser
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_tool_call_parser.cpp LIBRARIES fngine_text)
fngine_add_test(text_prompt_lookup
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_prompt_lookup.cpp LIBRARIES fngine_text)
