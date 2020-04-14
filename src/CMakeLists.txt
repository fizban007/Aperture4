set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/buffer_impl.cpp"
  "utils/mpi_helper.cpp"
  # "core/params.cpp"
  "framework/environment.cpp"
  "framework/params_store.cpp"
  "systems/domain_comm.cpp"
  "systems/grid.cpp"
  # "systems/grid_logsph.cpp"
  "data/fields.cpp"
  "data/particle_data.cpp"
  )

if (use_cuda AND CMAKE_CUDA_COMPILER)
  list(APPEND Aperture_src
    "utils/buffer_impl.cu"
    "core/constant_mem.cu"
    "core/particles.cu"
    # "systems/ffe_solver_EZ.cu"
    "systems/field_solver_default.cu"
    # "systems/field_solver_logsph.cu"
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

add_executable(aperture "main.cpp")
target_link_libraries(aperture Aperture)
