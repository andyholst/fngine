# Kernel areas. Each area directory defines fngine_kernels_<area> (STATIC, CUDA) and owns its
# sources. fngine_kernels links every area that exists.
add_library(fngine_kernels INTERFACE)
foreach(area linear gr gdn qsa ple moe head mtp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${area}/CMakeLists.txt)
    add_subdirectory(${area})
    target_link_libraries(fngine_kernels INTERFACE fngine_kernels_${area})
  endif()
endforeach()
