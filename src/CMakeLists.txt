set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp")

add_library(Aperture ${Aperture_src})
target_link_libraries(Aperture fmt::fmt)
