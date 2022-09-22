if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(alfven_quasi_1d
    "main.cpp"
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  set_target_properties(alfven_quasi_1d PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(alfven_quasi_1d Aperture Aperture_dev)
  add_dependencies(problems alfven_quasi_1d)

  add_executable(alfven_quasi_1d_single_ptc
    "single_ptc_test.cpp")
  target_link_libraries(alfven_quasi_1d_single_ptc Aperture Aperture_dev)
  add_dependencies(problems alfven_quasi_1d_single_ptc)
endif()
