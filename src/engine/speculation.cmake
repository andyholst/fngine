# fngine_speculation: cost-model speculation controller (pure host logic).
# Included by src/engine/CMakeLists.txt.
add_library(fngine_speculation STATIC ${CMAKE_CURRENT_LIST_DIR}/speculation.cpp)
fngine_internal_includes(fngine_speculation)
target_include_directories(fngine_speculation PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
