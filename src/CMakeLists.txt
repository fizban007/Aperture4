set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "core/particles.cpp")

add_library(Aperture ${Aperture_src})
target_link_libraries(Aperture fmt::fmt)

if (use_cuda AND CMAKE_CUDA_COMPILER)
  target_link_libraries(Aperture ${CUDART_LIBRARY})
endif()
