if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_library(reconnection_2d
    "initial_condition.cu"
    "systems/boundary_condition.cu"
    )
  target_link_libraries(reconnection_2d Aperture)
  add_dependencies(problems reconnection_2d)

  add_executable(single_cs "main.cpp")
  target_link_libraries(single_cs reconnection_2d)
  add_dependencies(problems single_cs)

  add_executable(boosted_cs "boosted_reconnection.cpp")
  target_link_libraries(boosted_cs reconnection_2d)
  add_dependencies(problems boosted_cs)

  add_executable(double_cs "double_sheet.cpp")
  target_link_libraries(double_cs reconnection_2d)
  add_dependencies(problems double_cs)

  add_executable(upstream_cooling "upstream_cooling.cu")
  target_link_libraries(upstream_cooling reconnection_2d)
  add_dependencies(problems upstream_cooling)

  add_executable(sync_cooling_test_2d "sync_cooling_test.cu")
  target_link_libraries(sync_cooling_test_2d Aperture)
  add_dependencies(problems sync_cooling_test_2d)
endif()
