add_custom_target(check)

add_executable(tests "test.cpp")
target_sources(tests PUBLIC
  "test_buffer.cpp"
)
target_link_libraries(tests Aperture)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

add_executable(mem_layout "mem_layout.cpp")
target_link_libraries(mem_layout Aperture)
