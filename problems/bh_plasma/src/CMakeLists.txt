if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(bh_plasma
    "injector.cu"
    "initial_condition.cu"
    "main.cpp"
    )
  set_target_properties(bh_plasma PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(bh_plasma Aperture)
  add_dependencies(problems bh_plasma)
endif()
