set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/buffer_impl.cpp"
  "core/params.cpp"
  "framework/environment.cpp"
  "framework/param_store.cpp"
  )

if (use_cuda AND CMAKE_CUDA_COMPILER)
  list(APPEND Aperture_src
    "utils/buffer_impl.cu"
    "core/constant_mem.cu"
    "core/particles.cu"
    )
else()
  list(APPEND Aperture_src
    "core/particles.cpp"
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
