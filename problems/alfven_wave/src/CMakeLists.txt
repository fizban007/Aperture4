if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(alfven_wave "main.cu"
    "systems/boundary_condition.cu"
    "systems/initial_condition.cu"
    # "systems/ptc_injector_mult.cu"
    )
  set_target_properties(alfven_wave PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(alfven_wave Aperture Aperture_dev)
  add_dependencies(problems alfven_wave)
endif()
