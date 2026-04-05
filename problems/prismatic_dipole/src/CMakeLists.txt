add_aperture_executable(prismatic_dipole "main.cpp")
add_dependencies(problems prismatic_dipole)

add_aperture_executable(slice_fields "slice_fields.cpp")
add_dependencies(problems slice_fields)
