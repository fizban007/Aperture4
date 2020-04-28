set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/buffer_impl.cpp"
  "utils/mpi_helper.cpp"
  "utils/hdf_wrapper.cpp"
  "framework/environment.cpp"
  "framework/params_store.cpp"
  "data/fields.cpp"
  "core/detail/multi_array_helpers.cpp"
  "data/particle_data.cpp"
  "systems/data_exporter.cpp"
  "systems/domain_comm.cpp"
  "systems/grid.cpp"
  "systems/grid_sph.cpp"
  "systems/grid_logsph.cpp"
  "systems/ptc_updater.cpp"
  )

if (use_cuda AND CMAKE_CUDA_COMPILER)
  list(APPEND Aperture_src
    "utils/buffer_impl.cu"
    "core/constant_mem.cu"
    "core/particles.cu"
    "core/detail/multi_array_helpers.cu"
    "data/curand_states.cu"
    # "systems/ffe_solver_EZ.cu"
    "systems/field_solver_default.cu"
    "systems/field_solver_logsph.cu"
    "systems/ptc_updater.cu"
    "systems/ptc_updater_logsph.cu"
    )
else()
  list(APPEND Aperture_src
    "core/particles.cpp"
    # "systems/ffe_solver_EZ.cpp"
    )
endif()

add_library(Aperture ${Aperture_src})
target_link_libraries(Aperture fmt::fmt ${MPI_LIBRARIES}
  ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

if (use_cuda AND CMAKE_CUDA_COMPILER)
  target_link_libraries(Aperture ${CUDART_LIBRARY})
endif()

# add_executable(aperture "main.cpp")
# target_link_libraries(aperture Aperture)
