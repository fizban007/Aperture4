add_aperture_executable(fast_wave_2d "main.cpp")
add_dependencies(problems fast_wave_2d)

add_aperture_executable(fast_wave_2d_cooling "main_cooling.cpp")
add_dependencies(problems fast_wave_2d_cooling)