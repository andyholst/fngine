# fngine_engine: the public Engine (include/fngine/engine.h) over the model runtime.
include(${CMAKE_CURRENT_SOURCE_DIR}/speculation.cmake)

add_library(fngine_engine STATIC
  engine.cpp
  session.cpp
)
fngine_internal_includes(fngine_engine)
target_include_directories(fngine_engine PUBLIC
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(fngine_engine
  PUBLIC fngine_text Threads::Threads
  PRIVATE fngine_model fngine_speculation spdlog::spdlog CUDA::cudart)
