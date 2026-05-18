# if (use_cuda AND CMAKE_CUDA_COMPILER)
#   add_library(gr_2d_kerr_schild
#     "injector.cu"
#     "initial_condition.cu"
#     # "systems/radiation/gr_inverse_compton.cu"
#     )
#   target_link_libraries(gr_2d_kerr_schild Aperture Aperture_dev)

#   add_executable(grpic "main.hip.cpp")
#   set_target_properties(grpic PROPERTIES
#     OUTPUT_NAME "aperture")
#   target_link_libraries(grpic gr_2d_kerr_schild)
#   add_dependencies(problems grpic)

#   add_executable(gr_vacuum "vacuum.cpp")
#   target_link_libraries(gr_vacuum gr_2d_kerr_schild)
#   add_dependencies(problems gr_vacuum)

#   add_executable(gr_discharge "monopole_discharge.cpp")
#   target_link_libraries(gr_discharge gr_2d_kerr_schild)
#   add_dependencies(problems gr_discharge)

#   add_executable(gr_single_ptc "test_particle.cpp")
#   target_link_libraries(gr_single_ptc gr_2d_kerr_schild)
#   add_dependencies(problems gr_single_ptc)
# endif()

add_aperture_library(gr_3d_kerr_schild
  "injector.cpp"
  "density_floor_injector.cpp"
  "initial_condition.cpp")

# add_aperture_executable(grpic "main.cpp")
# target_link_libraries(grpic gr_2d_kerr_schild)
# add_dependencies(problems grpic)
# 
add_aperture_executable(vacuum_monopole_3d "monopole_vacuum.cpp")
target_link_libraries(vacuum_monopole_3d gr_3d_kerr_schild)
add_dependencies(problems vacuum_monopole_3d)

add_aperture_executable(monopole_3d "main_monopole.cpp")
target_link_libraries(monopole_3d gr_3d_kerr_schild)
add_dependencies(problems monopole_3d)

add_aperture_executable(wald_3d "main_wald.cpp")
target_link_libraries(wald_3d gr_3d_kerr_schild)
add_dependencies(problems wald_3d)
# 
# add_aperture_executable(gap "main_gap.cpp")
# target_link_libraries(gap gr_2d_kerr_schild)
# add_dependencies(problems gap)

#add_aperture_executable(fluxhold "main_fluxhold.cpp")
#target_link_libraries(fluxhold gr_2d_kerr_schild)
#add_dependencies(problems fluxhold)

add_aperture_executable(vacuum_wald_3d "vacuum.cpp")
target_link_libraries(vacuum_wald_3d gr_3d_kerr_schild)
add_dependencies(problems vacuum_wald_3d)

# add_aperture_executable(gr_single_ptc "test_particle.cpp")
# target_link_libraries(gr_single_ptc gr_2d_kerr_schild)
# add_dependencies(problems gr_single_ptc)


# add_aperture_executable(torus "main_torus.cpp")
# target_link_libraries(torus gr_2d_kerr_schild)
# add_dependencies(problems torus)
# 
# add_aperture_executable(torus_vertB "main_torus_vertB.cpp")
# target_link_libraries(torus_vertB gr_2d_kerr_schild)
# add_dependencies(problems torus_vertB)

add_aperture_executable(bondi_3d "main_bondi.cpp")
target_link_libraries(bondi_3d gr_3d_kerr_schild)
add_dependencies(problems bondi_3d)
