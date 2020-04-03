cmake_minimum_required(VERSION 3.8)
project(Aperture LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-DUSE_BMI2")
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost -no-prec-div -DUSE_BMI2")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DUSE_BMI2")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Set include directories
set(FMT_INCLUDE_DIR deps/fmt/include)
set(MORTON_INCLUDE_DIR deps/mortonlib/include)

include_directories(
  ${FMT_INCLUDE_DIR}
  ${MORTON_INCLUDE_DIR}
  src)

add_subdirectory(deps/fmt)
add_subdirectory(src)
add_subdirectory(tests)
