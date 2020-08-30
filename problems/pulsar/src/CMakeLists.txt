if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(pulsar "main.cu"
    "systems/boundary_condition.cu"
    "systems/ptc_injector_pulsar.cu"
    "systems/ptc_updater_pulsar.cu"
    )
  set_target_properties(pulsar PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(pulsar Aperture)
  add_dependencies(problems pulsar)

  add_executable(pulsar_single_ptc "single_particle.cu"
    "systems/ptc_updater_pulsar.cu"
    )
  set_target_properties(pulsar_single_ptc PROPERTIES OUTPUT_NAME "single_ptc")
  target_link_libraries(pulsar_single_ptc Aperture)
  add_dependencies(problems pulsar_single_ptc)

  add_executable(debug "debug.cu"
    "systems/ptc_updater_pulsar.cu"
    "systems/boundary_condition.cu"
    )
  target_link_libraries(debug Aperture)
  add_dependencies(problems debug)
endif()
