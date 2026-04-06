add_aperture_executable(prismatic_dipole "main.cpp")
add_dependencies(problems prismatic_dipole)

add_aperture_executable(vacuum_dipole "vacuum_dipole.cpp")
add_dependencies(problems vacuum_dipole)

add_aperture_executable(streaming_test "streaming_test.cpp")
add_dependencies(problems streaming_test)

# add_aperture_executable(slice_fields "slice_fields.cpp")
# add_dependencies(problems slice_fields)
