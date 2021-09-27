if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(single_stream
    "single_stream.cpp"
    "initial_condition.cu"
    )
  set_target_properties(single_stream PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(single_stream Aperture)
  add_dependencies(problems single_stream)
endif()
