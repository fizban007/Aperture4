if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(alfven_charge_starve
    "main.cpp"
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  set_target_properties(alfven_charge_starve PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(alfven_charge_starve Aperture)
  add_dependencies(problems alfven_charge_starve)
endif()
