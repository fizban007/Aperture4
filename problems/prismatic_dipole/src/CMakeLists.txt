add_aperture_executable(prismatic_dipole "main.cpp")
add_dependencies(problems prismatic_dipole)

add_aperture_executable(vacuum_dipole "vacuum_dipole.cpp")
add_dependencies(problems vacuum_dipole)

add_aperture_executable(cavity_resonator "cavity_resonator.cpp")
add_dependencies(problems cavity_resonator)

add_aperture_executable(streaming_test "streaming_test.cpp")
add_dependencies(problems streaming_test)

# add_aperture_executable(slice_fields "slice_fields.cpp")
# add_dependencies(problems slice_fields)

add_aperture_executable(ns_rotator "ns_rotator.cpp")
add_dependencies(problems ns_rotator)
