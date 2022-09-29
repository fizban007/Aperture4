set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/mpi_helper.cpp"
  "utils/hdf_wrapper.cpp"
  "framework/environment.cpp"
  "framework/params_store.cpp"
  "data/fields.cpp"
  "core/buffer_impl.cpp"
  "core/detail/multi_array_helpers.cpp"
  "data/particle_data.cpp"
  "systems/data_exporter.cpp"
  "systems/domain_comm.cpp"
  "systems/grid.cpp"
  "systems/grid_sph.cpp"
  "systems/grid_polar.cpp"
  "systems/grid_ks.cpp"
  "systems/field_solver.cpp"
  "systems/legacy/ptc_updater_old.cpp"
  "systems/compute_lorentz_factor.cpp"
  "systems/gather_momentum_space.cpp"
  "systems/inverse_compton.cpp"
  "systems/sync_curv_emission.cpp"
  )
set_source_files_properties(${Aperture_src} PROPERTIES LANGUAGE CXX)

if (use_cuda AND CMAKE_CUDA_COMPILER)
  set(Aperture_dev_src
    "core/buffer_impl.hip.cpp"
    "core/constant_mem.hip.cpp"
    "core/particles.hip.cpp"
    "core/detail/multi_array_helpers.hip.cpp"
    "data/rng_states.hip.cpp"
    "systems/field_solver.hip.cpp"
    "systems/field_solver_sph.hip.cpp"
    "systems/field_solver_polar.hip.cpp"
    "systems/field_solver_gr_ks.hip.cpp"
    "systems/compute_lorentz_factor.hip.cpp"
    "systems/gather_momentum_space.hip.cpp"
    "systems/ptc_updater_base.hip.cpp"
    "systems/radiative_transfer.hip.cpp"
    )
  set_source_files_properties(${Aperture_dev_src} PROPERTIES LANGUAGE CUDA)
elseif(use_hip)
  set(Aperture_dev_src
    "core/buffer_impl.hip.cpp"
    "core/constant_mem.hip.cpp"
    "core/particles.hip.cpp"
    "core/detail/multi_array_helpers.hip.cpp"
    "data/rng_states.hip.cpp"
    "systems/field_solver.hip.cpp"
    "systems/field_solver_sph.hip.cpp"
    "systems/field_solver_polar.hip.cpp"
    "systems/field_solver_gr_ks.hip.cpp"
    "systems/compute_lorentz_factor.hip.cpp"
    "systems/gather_momentum_space.hip.cpp"
    "systems/ptc_updater_base.hip.cpp"
    "systems/radiative_transfer.hip.cpp"
    )
  set_source_files_properties(${Aperture_dev_src} PROPERTIES LANGUAGE HIP)
else()
  # list(APPEND Aperture_src
  set(Aperture_dev_src
    "core/particles.cpp"
    "data/rng_states.cpp"
    "systems/ptc_updater_base.cpp"
    "systems/ptc_updater_simd.cpp"
    "systems/radiative_transfer.cpp"
    # "systems/ffe_solver_EZ.cpp"
    )
endif()

add_library(Aperture STATIC ${Aperture_src})
# set_property(TARGET Aperture PROPERTY CUDA_ARCHITECTURES 60 61)
target_link_libraries(Aperture fmt::fmt ${MPI_LIBRARIES}
  ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_library(Aperture_dev STATIC ${Aperture_dev_src})
target_link_libraries(Aperture_dev Aperture)

if (use_cuda AND CMAKE_CUDA_COMPILER)
  target_link_libraries(Aperture_dev ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
endif()

if (use_hip AND CMAKE_HIP_COMPILER)
  # target_link_libraries(Aperture_dev hip::host hip::device roc::rocthrust roc::rocrand)
  target_link_libraries(Aperture_dev hip::host hip::device roc::rocthrust)
endif()


# add_executable(aperture "main.cpp")
# target_link_libraries(aperture Aperture)
