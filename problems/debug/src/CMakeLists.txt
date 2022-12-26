file(GLOB debug_src_files *.cu *.hip.cpp)
message(${debug_src_files})
if (use_cuda AND CMAKE_CUDA_COMPILER)
  set_source_files_properties(
    ${debug_src_files}
    PROPERTIES LANGUAGE CUDA)
elseif (use_hip AND CMAKE_HIP_COMPILER)
  set_source_files_properties(
    ${debug_src_files}
    PROPERTIES LANGUAGE HIP)
endif()

if (use_cuda OR use_hip)
  add_executable(single_stream
    "single_stream.cu"
    "initial_condition.cu"
    )
  target_link_libraries(single_stream Aperture Aperture_dev)
  add_dependencies(problems single_stream)

  add_executable(phase_space_test
    "phase_space.cu")
  target_link_libraries(phase_space_test Aperture Aperture_dev)
  add_dependencies(problems phase_space_test)

  add_executable(IC_test
    "IC_test.hip.cpp")
  target_link_libraries(IC_test Aperture Aperture_dev)
  add_dependencies(problems IC_test)

  add_executable(IC_single
    "IC_single.cu")
  target_link_libraries(IC_single Aperture Aperture_dev)
  add_dependencies(problems IC_single)

  add_executable(pml_test
    "pml_test.cu")
  target_link_libraries(pml_test Aperture Aperture_dev)
  add_dependencies(problems pml_test)

  add_executable(gca_test
    "test_gca.hip.cpp")
  target_link_libraries(gca_test Aperture Aperture_dev)
  add_dependencies(problems gca_test)

  add_executable(gca_test_compare
    "test_gca_compare.hip.cpp")
  target_link_libraries(gca_test_compare Aperture Aperture_dev)
  add_dependencies(problems gca_test_compare)
else()
  add_executable(pml_test
    "pml_test.cpp")
  target_link_libraries(pml_test Aperture)
  add_dependencies(problems pml_test)
endif()
