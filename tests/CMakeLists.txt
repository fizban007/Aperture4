add_executable(tests "test.cpp")
target_sources(tests PUBLIC
  "test_buffer.cpp"
)
target_link_libraries(tests Aperture)

add_executable(mem_layout "mem_layout.cpp")
target_link_libraries(mem_layout Aperture)
