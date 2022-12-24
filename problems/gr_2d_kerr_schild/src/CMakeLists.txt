if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(gr_2d_kerr_schild
    "injector.cu"
    "initial_condition.cu"
    # "systems/radiation/gr_inverse_compton.cu"
    )
  target_link_libraries(gr_2d_kerr_schild Aperture Aperture_dev)

  add_executable(grpic "main.hip.cpp")
  set_target_properties(grpic PROPERTIES
    OUTPUT_NAME "aperture")
  target_link_libraries(grpic gr_2d_kerr_schild)
  add_dependencies(problems grpic)

  add_executable(gr_vacuum "vacuum.cpp")
  target_link_libraries(gr_vacuum gr_2d_kerr_schild)
  add_dependencies(problems gr_vacuum)

  add_executable(gr_monopole "monopole_vacuum.cpp")
  target_link_libraries(gr_monopole gr_2d_kerr_schild)
  add_dependencies(problems gr_monopole)

  add_executable(gr_discharge "monopole_discharge.cpp")
  target_link_libraries(gr_discharge gr_2d_kerr_schild)
  add_dependencies(problems gr_discharge)

  add_executable(gr_single_ptc "test_particle.cpp")
  target_link_libraries(gr_single_ptc gr_2d_kerr_schild)
  add_dependencies(problems gr_single_ptc)
endif()
