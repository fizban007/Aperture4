add_aperture_executable(two_stream "main.cpp")
add_dependencies(problems two_stream)

add_aperture_executable(two_stream_1d "main_1d.cpp")
add_dependencies(problems two_stream_1d)

add_aperture_executable(single_stream "single_stream.cpp")
add_dependencies(problems single_stream)
