add_executable(vacuum_wave "main.cpp" "systems/boundary_condition.cu")
set_target_properties(vacuum_wave PROPERTIES OUTPUT_NAME "aperture")
target_link_libraries(vacuum_wave Aperture)
