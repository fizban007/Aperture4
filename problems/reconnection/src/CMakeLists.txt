if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(reconnection_2d
    "main.cpp"
    "initial_condition.cu"
    )
  set_target_properties(reconnection_2d PROPERTIES OUTPUT_NAME "aperture")
  target_link_libraries(reconnection_2d Aperture)
  add_dependencies(problems reconnection_2d)

  add_executable(sync_cooling_test_2d "sync_cooling_test.cu")
  target_link_libraries(sync_cooling_test_2d Aperture)
  add_dependencies(problems sync_cooling_test_2d)
endif()
