if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(bh_plasma
    "injector.cu"
    "initial_condition.cu"
    )
  target_link_libraries(bh_plasma Aperture)

  add_executable(grpic "main.cpp")
  set_target_properties(grpic PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(grpic bh_plasma)
  add_dependencies(problems grpic)

  add_executable(gr_single_ptc "test_particle.cpp")
  target_link_libraries(gr_single_ptc bh_plasma)
  add_dependencies(problems gr_single_ptc)
endif()
