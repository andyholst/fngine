# Speculative decoding equivalence / acceptance / rollback on the real model (FNGINE_TEST_PACK).
fngine_add_test(test_speculative_real
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_speculative_real.cpp
  LIBRARIES fngine_model fngine_engine)
