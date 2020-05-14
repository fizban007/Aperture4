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
  "test_exporter.cpp"
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

if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(test_mpi2d "test_mpi2d.cpp")
  target_link_libraries(test_mpi2d Aperture)

  add_executable(perf_pusher "perf_pusher.cpp")
  target_link_libraries(perf_pusher Aperture)
  add_executable(perf_pusher3d "perf_pusher3d.cpp")
  target_link_libraries(perf_pusher3d Aperture)
  add_executable(perf_deposit "perf_deposit.cpp")
  target_link_libraries(perf_deposit Aperture)
  add_executable(perf_deposit3d "perf_deposit3d.cpp")
  target_link_libraries(perf_deposit3d Aperture)
endif()
