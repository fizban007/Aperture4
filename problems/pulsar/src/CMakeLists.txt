if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(pulsar "main.cu"
    "systems/boundary_condition.cu"
    "systems/ptc_injector_pulsar.cu"
    "systems/ptc_updater_pulsar.cu"
    )
  set_target_properties(pulsar PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(pulsar Aperture)
  add_dependencies(problems pulsar)
endif()
