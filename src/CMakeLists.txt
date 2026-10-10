add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/ggml-iq ${PROJECT_BINARY_DIR}/third_party/ggml-iq)

# Each component directory owns its CMakeLists.txt and defines the targets named here.
#   core     -> fngine_core
#   kernels  -> fngine_kernels (INTERFACE over fngine_kernels_<area>)
#   cpu      -> fngine_cpu
#   io       -> fngine_io
#   text     -> fngine_text
#   model    -> fngine_model
#   engine   -> fngine_engine
#   serve    -> fngine_serve
foreach(component core kernels cpu io text model engine serve)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${component}/CMakeLists.txt)
    add_subdirectory(${component})
  endif()
endforeach()
