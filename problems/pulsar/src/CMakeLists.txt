if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(pulsar "main.cpp"
    "systems/boundary_condition.cu"
    )
  set_target_properties(pulsar PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(pulsar Aperture)
endif()
