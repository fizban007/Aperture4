if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(single_stream
    "single_stream.cu"
    "initial_condition.cu"
    )
  set_target_properties(single_stream PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(single_stream Aperture)
  add_dependencies(problems single_stream)

  add_executable(phase_space_test
    "phase_space.cu")
  target_link_libraries(phase_space_test Aperture)
  add_dependencies(problems phase_space_test)

  add_executable(IC_test
    "IC_test.cu")
  target_link_libraries(IC_test Aperture)
  add_dependencies(problems IC_test)

  add_executable(IC_single
    "IC_single.cu")
  target_link_libraries(IC_single Aperture)
  add_dependencies(problems IC_single)
endif()
