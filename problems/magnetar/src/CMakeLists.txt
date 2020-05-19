if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(magnetar "main.cpp"
    "systems/boundary_condition.cu"
    "systems/ptc_updater_magnetar.cu"
    )
  set_target_properties(magnetar PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(magnetar Aperture)
endif()
