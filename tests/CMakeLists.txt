add_custom_target(check)

add_executable(tests "test.cpp")
target_sources(tests PUBLIC
  "test_buffer.cpp"
  "test_particles.cpp"
  "test_multi_array.cpp"
  "test_params.cpp"
  "test_fields.cpp"
  "test_interp.cpp"
  "test_ptc_updater.cpp"
  )
if (use_cuda AND CMAKE_CUDA_COMPILER)
  target_sources(tests PUBLIC
    "test_multi_array.cu"
    "test_vec.cu"
    "test_params.cu"
    "test_grid.cu"
    )
else()
  target_sources(tests PUBLIC
    # "test_ptc_updater.cpp"
    )
endif()

target_link_libraries(tests Aperture)

add_custom_target(
  run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

# add_executable(mem_layout "mem_layout.cpp")
# target_link_libraries(mem_layout Aperture)

# add_executable(playground "playground.cpp")
# target_link_libraries(playground Aperture)

if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(playground_gpu "playground.cu")
  target_link_libraries(playground_gpu Aperture)
endif()
