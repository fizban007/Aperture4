add_custom_target(check)

# add_executable(tests "test.cpp")
add_executable(tests
  "test_util_functions.cpp"
  "test_buffer.cpp"
  "test_particles.cpp"
  "test_multi_array.cpp"
  "test_params.cpp"
  "test_fields.cpp"
  "test_interp.cpp"
  # "test_ptc_updater.cpp"
  "test_exporter.cpp"
  "test_config.cpp"
  "test_gaussian_quadrature.cpp"
  "test_singleton.cpp"
  "test_mem_layout.cpp"
  "test_lorentz.cpp"
  "test_rng.cpp"
  )
# target_link_libraries(tests Aperture Aperture_dev Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC
  Aperture
  $<TARGET_NAME_IF_EXISTS:Aperture_dev>
  Catch2::Catch2WithMain)

# add_executable(tests_dev "test.cpp")
if (use_cuda OR use_hip)
if (use_cuda AND CMAKE_CUDA_COMPILER)
  # target_sources(tests PUBLIC
  set(test_dev_src
    "test_multi_array.cu"
    "test_vec.cu"
    "test_params.cu"
    "test_grid.cu"
    "test_math.cu"
    "test_cusparse.cu"
    "test_rng.cu"
    # "compare_new_pusher_sph.cpp"
    )
  set_source_files_properties(${test_dev_src} PROPERTIES LANGUAGE CUDA)
elseif(use_hip)
  set(test_dev_src
    "test_multi_array.cu"
    "test_vec.cu"
    "test_params.cu"
    "test_grid.cu"
    "test_math.cu"
    "test_rng.cu"
    # "compare_new_pusher_sph.cpp"
    )
  set_source_files_properties(${test_dev_src} PROPERTIES
    LANGUAGE HIP
    # COMPILE_FLAGS "-save-temps"
    )
  # target_sources(tests_dev PUBLIC
endif()
  add_executable(tests_dev
    ${test_dev_src})
  # target_link_libraries(tests_dev Aperture_dev Catch2::Catch2WithMain)
  target_link_libraries(tests_dev PUBLIC
    Aperture
    $<TARGET_NAME_IF_EXISTS:Aperture_dev>
    Catch2::Catch2WithMain)
endif()

add_custom_target(
  run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)
if(use_cuda OR use_hip)
  add_dependencies(run_tests tests_dev)
  add_custom_command(
    TARGET run_tests POST_BUILD
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests_dev"
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

if (use_cuda AND CMAKE_CUDA_COMPILER)
  # add_executable(test_summit_mpi "test_summit_mpi.cu")
  # target_link_libraries(test_summit_mpi Aperture_dev)

  # add_executable(test_mpi3d "test_mpi3d.cpp")
  # target_link_libraries(test_mpi3d Aperture)

  # add_executable(perf_ptc_update "perf_ptc_update.cpp")
  # target_link_libraries(perf_ptc_update Aperture)

  # add_executable(perf_pusher "perf_pusher.cpp")
  # target_link_libraries(perf_pusher Aperture_dev)
  # add_executable(perf_pusher3d "perf_pusher3d.cpp")
  # target_link_libraries(perf_pusher3d Aperture_dev)
  # add_executable(perf_deposit "perf_deposit.cpp")
  # target_link_libraries(perf_deposit Aperture_dev)
  # add_executable(perf_deposit3d "perf_deposit3d.cpp")
  # target_link_libraries(perf_deposit3d Aperture_dev)
  # add_executable(perf_gr_update "perf_gr_update.cpp")
  # target_link_libraries(perf_gr_update Aperture_dev)

  # add_executable(mock_loop "mock_loop.cu")
  # target_link_libraries(mock_loop Aperture_dev)

  # add_executable(test_IC_cooling "test_IC_cooling.cu")
  # target_link_libraries(test_IC_cooling Aperture_dev)

  # add_executable(test_cyclotron_cooling "test_cyclotron_cooling.cu")
  # target_link_libraries(test_cyclotron_cooling Aperture_dev)

  # add_executable(test_gca_pusher "test_gca_pusher.cu")
  # target_link_libraries(test_gca_pusher Aperture Aperture_dev)

else()
  # add_executable(perf_ptc_update_cpu "perf_ptc_update_cpu.cpp")
  # target_link_libraries(perf_ptc_update_cpu Aperture_dev)

  # add_executable(perf_ptc_update_mpi "perf_ptc_update_mpi.cpp")
  # target_link_libraries(perf_ptc_update_mpi Aperture_dev)

endif()

# if (use_cuda OR use_hip)

# if (use_cuda AND CMAKE_CUDA_COMPILER)
#   set_source_files_properties("test_exporter_snapshot.hip.cpp" PROPERTIES LANGUAGE CUDA)
# elseif(use_hip)
#   set_source_files_properties("test_exporter_snapshot.hip.cpp" PROPERTIES LANGUAGE HIP)
# endif()

# add_executable(test_exporter_snapshot "test_exporter_snapshot.hip.cpp")
# # target_link_libraries(test_exporter_snapshot Aperture_dev Catch2::Catch2)
# target_link_libraries(test_exporter_snapshot
#   PUBLIC
#   Aperture
#   $<TARGET_NAME_IF_EXISTS:Aperture_dev>
#   Catch2::Catch2)

# endif()

add_aperture_executable(test_exporter_snapshot "test_exporter_snapshot.cpp")
target_link_libraries(test_exporter_snapshot Catch2::Catch2)


add_executable(test_exporter_mpi "test_exporter_mpi.cpp")
# target_link_libraries(test_exporter_mpi Aperture Aperture_dev)
target_link_libraries(test_exporter_mpi
  PUBLIC
  Aperture
  $<TARGET_NAME_IF_EXISTS:Aperture_dev>
  )

add_executable(test_mpi2d "test_mpi2d.cpp")
# target_link_libraries(test_mpi2d Aperture Aperture_dev)
target_link_libraries(test_mpi2d
  PUBLIC
  Aperture
  $<TARGET_NAME_IF_EXISTS:Aperture_dev>
  )

add_aperture_executable(test_sync_polarization "test_sync_polarization.cpp")
# add_executable(gr_pusher "test_gr_pusher.cpp")
# target_link_libraries(gr_pusher Aperture_dev)

# add_executable(test_sync_cooling "test_sync_cooling.cpp")
# target_link_libraries(test_sync_cooling Aperture_dev)

# add_executable(test_sync_emission "test_sync_emission.cpp")
# target_link_libraries(test_sync_emission Aperture_dev)

# # add_executable(test_omp "test_omp.cpp")

# add_executable(test_inverse_compton "test_inverse_compton.cpp")
# target_link_libraries(test_inverse_compton Aperture_dev)
