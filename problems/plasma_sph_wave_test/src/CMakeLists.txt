add_executable(plasma_wave "main.cpp" "systems/boundary_condition.cu")
set_target_properties(plasma_wave PROPERTIES OUTPUT_NAME "aperture")
target_link_libraries(plasma_wave Aperture)
