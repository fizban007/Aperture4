if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(scaling
    "initial_condition.cu"
    )
  target_link_libraries(scaling Aperture Aperture_dev)
  add_dependencies(problems scaling)

  add_executable(scaling_2d "scaling_2d.cpp")
  target_link_libraries(scaling_2d scaling)
  add_dependencies(problems scaling_2d)

  add_executable(scaling_3d "scaling_3d.cpp")
  target_link_libraries(scaling_3d scaling)
  add_dependencies(problems scaling_3d)

  # add_executable(scaling_single "scaling_single.cpp")
  # target_link_libraries(scaling_single scaling)
  # add_dependencies(problems scaling_single)

endif()
