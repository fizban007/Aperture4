if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(two_stream
    "main.cpp"
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  set_target_properties(two_stream PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(two_stream Aperture)
  add_dependencies(problems two_stream)
endif()
