# if (use_cuda AND CMAKE_CUDA_COMPILER)
#   add_library(lib_magnetar
#     "systems/boundary_condition.cu"
#     "systems/ptc_updater_magnetar.cu"
#     "systems/rt_magnetar.cu"
#     )
#   target_link_libraries(lib_magnetar Aperture)
#   add_executable(magnetar "main.cpp")
#   set_target_properties(magnetar PROPERTIES OUTPUT_NAME "aperture")
#   target_link_libraries(magnetar lib_magnetar)
#   add_dependencies(problems magnetar)

#   add_executable(magnetar_single_ptc "single_ptc.cpp")
#   set_target_properties(magnetar_single_ptc PROPERTIES OUTPUT_NAME "single_ptc")
#   target_link_libraries(magnetar_single_ptc lib_magnetar)
#   add_dependencies(problems magnetar_single_ptc)
# endif()
add_aperture_executable(magnetar "main.cpp")
add_dependencies(problems magnetar)

add_aperture_executable(magnetar_single_ptc "single_ptc")
add_dependencies(problems magnetar_single_ptc)

add_aperture_executable(magnetar_flux_tube "flux_tube")
add_dependencies(problems magnetar_flux_tube)