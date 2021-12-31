if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(wave_scattering
    "main.cpp"
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  set_target_properties(wave_scattering PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(wave_scattering Aperture)
  add_dependencies(problems wave_scattering)

  add_executable(polar_wave
    "polar.cu"
    )
  target_link_libraries(polar_wave Aperture)
  add_dependencies(problems polar_wave)

  # add_executable(alfven_quasi_1d_single_ptc
  #   "single_ptc_test.cpp")
  # target_link_libraries(alfven_quasi_1d_single_ptc Aperture)
  # add_dependencies(problems alfven_quasi_1d_single_ptc)
endif()
