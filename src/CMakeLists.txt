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

if (use_cuda AND CMAKE_CUDA_COMPILER)
  set(Aperture_cuda_src
    "core/buffer_impl.hip.cpp"
    "core/constant_mem.hip.cpp"
    "core/particles.hip.cpp"
    "core/detail/multi_array_helpers.hip.cpp"
    "data/rng_states.hip.cpp"
    # "systems/ffe_solver_EZ.cu"
    "systems/field_solver.hip.cpp"
    "systems/field_solver_sph.hip.cpp"
    "systems/field_solver_polar.hip.cpp"
    "systems/field_solver_gr_ks.hip.cpp"
    # "systems/legacy/ptc_updater_old.cu"
    # "systems/legacy/ptc_updater_old_sph.cu"
    # "systems/legacy/ptc_updater_old_gr_ks.cu"
    # "systems/ptc_injector.cu"
    #"systems/domain_comm_async.cpp"
    "systems/compute_lorentz_factor.hip.cpp"
    "systems/gather_momentum_space.hip.cpp"
    "systems/ptc_updater_base.cu"
    "systems/radiative_transfer.cu"
    )
  set_source_files_properties(${Aperture_cuda_src} PROPERTIES LANGUAGE CUDA)
  list(APPEND Aperture_src
    ${Aperture_cuda_src})
elseif(use_hip)
  set(Aperture_hip_src
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
    # "systems/ptc_updater_base.cu"
    # "systems/radiative_transfer.cu"
    )
  set_source_files_properties(${Aperture_hip_src} PROPERTIES LANGUAGE HIP)
  list(APPEND Aperture_src
    ${Aperture_hip_src})
else()
  list(APPEND Aperture_src
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

if (use_cuda AND CMAKE_CUDA_COMPILER)
  target_link_libraries(Aperture ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
endif()



# add_executable(aperture "main.cpp")
# target_link_libraries(aperture Aperture)
