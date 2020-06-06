if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(plasma_wave "main.cpp"
    "systems/boundary_condition.cu"
    "systems/initial_condition.cu"
    )
  set_target_properties(plasma_wave PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(plasma_wave Aperture)
  add_dependencies(problems plasma_wave)
endif()
