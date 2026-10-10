# GDN kernel tests (owner: Gdn agent).
set(_gdn_dir ${CMAKE_CURRENT_LIST_DIR}/gdn)
fngine_add_test(kernels_gdn_recurrent ORACLE
  SOURCES ${_gdn_dir}/test_gdn_recurrent.cpp ${_gdn_dir}/gdn_test_device.cu
  LIBRARIES fngine_kernels_gdn)
fngine_add_test(kernels_gdn_fold ORACLE
  SOURCES ${_gdn_dir}/test_gdn_fold.cpp ${_gdn_dir}/gdn_test_device.cu
  LIBRARIES fngine_kernels_gdn)
fngine_add_test(kernels_gdn_ops ORACLE
  SOURCES ${_gdn_dir}/test_gdn_ops.cpp ${_gdn_dir}/gdn_test_device.cu
  LIBRARIES fngine_kernels_gdn)
