# Engine tests. test_engine_real needs FNGINE_TEST_PACK and ~25 GB VRAM / ~30 GB RAM.
if(TARGET fngine_engine)
  fngine_add_test(test_engine_session
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_engine_session.cpp LIBRARIES fngine_engine)
  fngine_add_test(test_engine_real
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp LIBRARIES fngine_engine)
  fngine_add_test(test_adaptive_experts
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_adaptive_experts.cpp LIBRARIES fngine_engine)
endif()
