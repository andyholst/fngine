# fngine_io: pack manifest loader, O_DIRECT staging/upload, n-gram store.
# Each owner appends its sources to FNGINE_IO_SOURCES.
set(FNGINE_IO_SOURCES
  pack.cpp
  direct_io.cpp
  ngram_store.cpp
)

add_library(fngine_io STATIC ${FNGINE_IO_SOURCES})
fngine_internal_includes(fngine_io)
target_link_libraries(fngine_io PUBLIC fngine_core fngine::json spdlog::spdlog Threads::Threads)
target_link_libraries(fngine_io PUBLIC PkgConfig::LIBURING)
