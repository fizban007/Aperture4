if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(polar_cap_3d
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    "systems/field_solver_frame_dragging.cu"
    )
  target_link_libraries(polar_cap_3d Aperture)
  add_dependencies(problems polar_cap_3d)

  add_executable(sim_polar_cap "main.cu")
  target_link_libraries(sim_polar_cap polar_cap_3d)
  add_dependencies(problems sim_polar_cap)

  add_executable(sim_polar_cap_boris "main_boris.cu")
  target_link_libraries(sim_polar_cap_boris polar_cap_3d)
  add_dependencies(problems sim_polar_cap_boris)

  add_executable(vacuum_test "vacuum_test.cu")
  target_link_libraries(vacuum_test polar_cap_3d)
  add_dependencies(problems vacuum_test)

  add_executable(ptc_test "ptc_test.cu")
  target_link_libraries(ptc_test polar_cap_3d)
  add_dependencies(problems ptc_test)

  add_executable(radiation_test "radiation_test.cu")
  target_link_libraries(radiation_test polar_cap_3d)
  add_dependencies(problems radiation_test)

endif()
