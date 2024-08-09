# add_aperture_library(Kink_Instability "initial_condition.cpp")
add_aperture_executable(kink "main.cpp")
add_dependencies(problems kink)


# file(GLOB reconnection_src_files *.cu *.hip.cpp)
# if (use_cuda AND CMAKE_CUDA_COMPILER)
#   set_source_files_properties(
#     ${reconnection_src_files}
#     PROPERTIES LANGUAGE CUDA)
# elseif (use_hip AND CMAKE_HIP_COMPILER)
#   set_source_files_properties(
#     ${reconnection_src_files}
#     PROPERTIES LANGUAGE HIP)
# endif()

# # if (use_cuda OR use_hip)
# if ((use_cuda AND CMAKE_CUDA_COMPILER) OR (use_hip AND CMAKE_HIP_COMPILER))
# set(reconnection_src
#   "initial_condition.cu"
#   "systems/boundary_condition.cu"
#   )
# add_library(reconnection ${reconnection_src})
# target_link_libraries(reconnection Aperture_dev)
# add_dependencies(problems reconnection)


# add_executable(IC_cooling "main_IC.cu")
# target_link_libraries(IC_cooling reconnection)
# add_dependencies(problems IC_cooling)

# add_executable(IC_cooling_3d "main_IC_3D.cu")
# target_link_libraries(IC_cooling_3d reconnection)
# add_dependencies(problems IC_cooling_3d)

# add_executable(double_cs "no_cooling.cu")
# target_link_libraries(double_cs reconnection)
# add_dependencies(problems double_cs)

# # add_executable(boosted_cs "boosted_reconnection.cpp")
# # target_link_libraries(boosted_cs reconnection_2d)
# # add_dependencies(problems boosted_cs)

# # add_executable(upstream_cooling "upstream_cooling.cu")
# # target_link_libraries(upstream_cooling reconnection_2d)
# # add_dependencies(problems upstream_cooling)

# add_executable(sync_cooling_test_2d "sync_cooling_test.cu")
# target_link_libraries(sync_cooling_test_2d Aperture Aperture_dev)
# add_dependencies(problems sync_cooling_test_2d)
# endif()
