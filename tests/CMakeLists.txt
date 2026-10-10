# fngine tests. Each component area owns one or more `tests/<component>/<area>.cmake` files
# that call fngine_add_test(). Tests that need a GPU or real data return 77 to skip.
#   Real-data environment: FNGINE_TEST_PACK (pack dir), FNGINE_TEST_REF (golden dir).
#   Kernel tests are named kernels_<name> so `ctest -R kernels_` selects them.

# fngine_add_test(<name> SOURCES ... [LIBRARIES ...] [ORACLE])
#   ORACLE: compile host code with -fno-fast-math -ffp-contract=off (FP64 oracles).
function(fngine_add_test name)
  cmake_parse_arguments(T "ORACLE" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  fngine_internal_includes(${name})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  target_link_libraries(${name} PRIVATE ${T_LIBRARIES})
  target_compile_definitions(${name} PRIVATE FNGINE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  if(T_ORACLE)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FNGINE_ORACLE_FLAGS}>)
  endif()
  get_target_property(_srcs ${name} SOURCES)
  foreach(_s ${_srcs})
    if(_s MATCHES "\\.cu$")
      fngine_cuda_target(${name})
      break()
    endif()
  endforeach()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

file(GLOB FNGINE_TEST_FRAGMENTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cmake)
list(SORT FNGINE_TEST_FRAGMENTS)
foreach(fragment ${FNGINE_TEST_FRAGMENTS})
  include(${fragment})
endforeach()
