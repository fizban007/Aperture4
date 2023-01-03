set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/mpi_helper.cpp"
  "utils/hdf_wrapper.cpp"
  "framework/environment.cpp"
  "framework/params_store.cpp"
  "data/fields.cpp"
  "data/rng_states.cpp"
  "core/buffer_impl.cpp"
  "core/detail/multi_array_helpers.cpp"
  "core/particles.cpp"
  "core/particles_functions.cpp"
  "data/particle_data.cpp"
  "data/tracked_ptc.cpp"
  "systems/data_exporter.cpp"
  "systems/domain_comm.cpp"
  "systems/grid.cpp"
  "systems/grid_sph.cpp"
  "systems/grid_polar.cpp"
  "systems/grid_ks.cpp"
  "systems/compute_moments.cpp"
  "systems/gather_momentum_space.cpp"
  "systems/gather_tracked_ptc.cpp"
  "systems/inverse_compton.cpp"
  "systems/field_solver_cartesian.cpp"
  "systems/ptc_updater.cpp"
  "systems/radiative_transfer.cpp"
  "systems/sync_curv_emission.cpp"
  )
set_source_files_properties(${Aperture_src} PROPERTIES LANGUAGE CXX)
add_library(Aperture STATIC ${Aperture_src})
target_link_libraries(Aperture fmt::fmt ${MPI_LIBRARIES}
  ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

if (use_cuda OR use_hip)
  set(Aperture_dev_src
    "core/buffer_impl.hip.cpp"
    "core/constant_mem.hip.cpp"
    "core/detail/multi_array_helpers.hip.cpp"
    "core/particles_functions.hip.cpp"
    "data/rng_states.hip.cpp"
    "systems/data_exporter.hip.cpp"
    "systems/domain_comm.hip.cpp"
    "systems/field_solver_cartesian.hip.cpp"
    # "systems/field_solver_sph.hip.cpp"
    # "systems/field_solver_polar.hip.cpp"
    # "systems/field_solver_gr_ks.hip.cpp"
    "systems/compute_moments.hip.cpp"
    "systems/gather_momentum_space.hip.cpp"
    "systems/gather_tracked_ptc.hip.cpp"
    "systems/ptc_updater.hip.cpp"
    "systems/radiative_transfer.hip.cpp"
    )
  if (use_cuda AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${Aperture_dev_src} PROPERTIES LANGUAGE CUDA)
  elseif(use_hip)
    set_source_files_properties(${Aperture_dev_src} PROPERTIES LANGUAGE HIP)
  endif()

  add_library(Aperture_dev ${Aperture_dev_src})
  target_link_libraries(Aperture_dev PUBLIC Aperture)
  if (use_cuda AND CMAKE_CUDA_COMPILER)
    target_link_libraries(Aperture_dev PUBLIC ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
    # target_link_libraries(Aperture ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
  endif()

  if (use_hip AND CMAKE_HIP_COMPILER)
    # target_link_libraries(Aperture_dev hip::host hip::device roc::rocthrust roc::rocrand)
    target_link_libraries(Aperture_dev PUBLIC hip::host hip::device roc::rocthrust)
    # target_link_libraries(Aperture hip::host hip::device roc::rocthrust)
  endif()
endif()
# else()
  # list(APPEND Aperture_src
  # set(Aperture_dev_src ""
    # "systems/ptc_updater_simd.cpp"
    # "systems/ffe_solver_EZ.cpp"
    # )
# endif()

# set_property(TARGET Aperture PROPERTY CUDA_ARCHITECTURES 60 61)




# add_executable(aperture "main.cpp")
# target_link_libraries(aperture Aperture)
