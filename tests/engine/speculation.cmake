# Built from sources directly so the test does not depend on src/engine/CMakeLists.txt.
fngine_add_test(test_speculation SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_speculation.cpp
  ${PROJECT_SOURCE_DIR}/src/engine/speculation.cpp)
