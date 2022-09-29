if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(sph_wave_test
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  target_link_libraries(sph_wave_test Aperture Aperture_dev)
  # set_target_properties(sph_wave_test PROPERTIES OUTPUT_NAME "aperture")

  add_executable(sph_wave_plasma "sph_wave_plasma.cpp")
  set_target_properties(sph_wave_plasma PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(sph_wave_plasma sph_wave_test)
  add_dependencies(problems sph_wave_plasma)
endif()
