add_aperture_executable(pml_test "pml_test.cpp")
add_dependencies(problems pml_test)

add_aperture_executable(gca_test "gca_test.cpp")
add_dependencies(problems gca_test)

if (use_cuda OR use_hip)
  add_aperture_executable(single_stream
    "single_stream.cu"
    "initial_condition.cu"
    )
  add_dependencies(problems single_stream)

  add_aperture_executable(phase_space_test
    "phase_space.cu")
  add_dependencies(problems phase_space_test)

  add_aperture_executable(IC_test
    "IC_test.hip.cpp")
  add_dependencies(problems IC_test)

  add_aperture_executable(IC_single
    "IC_single.cu")
  add_dependencies(problems IC_single)

  # add_executable(pml_test
  #   "pml_test.cu")
  # target_link_libraries(pml_test Aperture Aperture_dev)
  # add_dependencies(problems pml_test)

  add_aperture_executable(gca_test_compare
    "test_gca_compare.hip.cpp")
  add_dependencies(problems gca_test_compare)
endif()

