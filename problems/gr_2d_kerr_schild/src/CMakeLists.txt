if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(gr_2d_kerr_schild
    "injector.cu"
    "initial_condition.cu"
    "systems/radiation/gr_inverse_compton.cu"
    )
  target_link_libraries(gr_2d_kerr_schild Aperture)

  add_executable(grpic "main.cpp")
  set_target_properties(grpic PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(grpic gr_2d_kerr_schild)
  add_dependencies(problems grpic)

  add_executable(gr_vacuum "vacuum.cpp")
  target_link_libraries(gr_vacuum gr_2d_kerr_schild)
  add_dependencies(problems gr_vacuum)

  add_executable(gr_single_ptc "test_particle.cpp")
  target_link_libraries(gr_single_ptc gr_2d_kerr_schild)
  add_dependencies(problems gr_single_ptc)
endif()
