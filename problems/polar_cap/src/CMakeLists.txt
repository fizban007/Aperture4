if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(polar_cap_3d
    "initial_condition.cu"
    # "systems/boundary_condition.cu"
    )
  target_link_libraries(polar_cap_3d Aperture)
  add_dependencies(problems polar_cap_3d)

  add_executable(sim_polar_cap "main.cu")
  target_link_libraries(sim_polar_cap polar_cap_3d)
  add_dependencies(problems sim_polar_cap)

endif()
