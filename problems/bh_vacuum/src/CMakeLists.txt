if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(bh_vacuum
    "initial_condition.cu"
    "main.cpp"
    )
  set_target_properties(bh_vacuum PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(bh_vacuum Aperture)
  add_dependencies(problems bh_vacuum)
endif()
