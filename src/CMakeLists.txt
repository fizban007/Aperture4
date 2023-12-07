set(Aperture_src
<<<<<<< HEAD
  "${CMAKE_CURRENT_SOURCE_DIR}/core/buffer_impl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/detail/multi_array_helpers.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/particles_functions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/data/rng_states.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/logger.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/timer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/mpi_helper.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/hdf_wrapper.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/framework/environment.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/framework/params_store.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/data/fields.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/data/tracked_ptc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/data/particle_data.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/particles.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/inverse_compton.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/grid.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/grid_sph.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/grid_polar.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/grid_ks.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/sync_curv_emission.cpp"
  PARENT_SCOPE
)
set(Aperture_host_src
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/data_exporter.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/domain_comm.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/compute_moments.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/gather_momentum_space.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/gather_tracked_ptc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_cartesian.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_gr_ks.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_sph.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/ptc_updater.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/radiative_transfer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/systems/vlasov_solver.cpp"
  PARENT_SCOPE)
if (NOT build_for_frontier)
  set_source_files_properties(${Aperture_src} ${Aperture_host_src} PROPERTIES LANGUAGE CXX)
  add_library(Aperture STATIC ${Aperture_src} ${Aperture_host_src})
  target_link_libraries(Aperture fmt::fmt ${MPI_LIBRARIES}
    ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
endif()

if (use_cuda OR use_hip)
  set(Aperture_dev_src
    "${CMAKE_CURRENT_SOURCE_DIR}/core/buffer_impl.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/constant_mem.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/detail/multi_array_helpers.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/particles_functions.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/data/rng_states.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/data_exporter.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/domain_comm.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/compute_moments.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/gather_momentum_space.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/gather_tracked_ptc.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_cartesian.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_sph.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/field_solver_gr_ks.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/vlasov_solver.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/ptc_updater.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/systems/radiative_transfer.hip.cpp"
    PARENT_SCOPE)
  set_compile_gpu(${Aperture_dev_src})

  if (NOT build_for_frontier)
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
