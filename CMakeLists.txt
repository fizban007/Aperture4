cmake_minimum_required(VERSION 3.8)
project(Aperture LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

option(use_cuda "Include the GPU part of the code" OFF)
option(use_double "Use double precision" OFF)
option(build_docs "Build the doxygen documentation" OFF)
option(build_tests "Build the test suite" OFF)
option(lib_only "Only build the core Aperture library" ON)

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

# Detect C++ compiler features
# message("Your C++ compiler supports these C++ features:")
# foreach(i ${CMAKE_CXX_COMPILE_FEATURES})
#   message("${i}")
# endforeach()

# Set customized module path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-DUSE_BMI2")
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -xHost -no-prec-div -DUSE_BMI2 ${CXX_EXTRA_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DUSE_BMI2 ${CXX_EXTRA_FLAGS}")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# Find required packages
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

# Set include directories
set(FMT_INCLUDE_DIR deps/fmt/include)
set(CATCH_INCLUDE_DIR deps/catch/single_include/catch2)
set(VISIT_STRUCT_INCLUDE_DIR deps/visit_struct/include)
set(CPPTOML_INCLUDE_DIR deps/cpptoml/include)
set(CXXOPT_INCLUDE_DIR deps/cxxopts/include)
set(GSL_INCLUDE_DIR deps/gsl/include)

include_directories(
  ${MPI_INCLUDE_PATH}
  ${FMT_INCLUDE_DIR}
  ${CATCH_INCLUDE_DIR}
  ${VISIT_STRUCT_INCLUDE_DIR}
  ${CPPTOML_INCLUDE_DIR}
  ${CXXOPT_INCLUDE_DIR}
  ${GSL_INCLUDE_DIR}
  src)

# Check for cuda support
if(use_cuda)
  message("Checking cuda support")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DCUDA_ENABLED)
    message(
      "Enabling Cuda, ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS}"
    )
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                        ${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS
        "--expt-extended-lambda --expt-relaxed-constexpr\
      -lcublas -lcurand -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored \
      -Xcudafe --diag_suppress=code_is_unreachable")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # message(${CMAKE_CXX_COMPILER})
    find_library(CUDART_LIBRARY cudart
                 PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()

if(use_double)
  add_definitions(-DUSE_DOUBLE)
endif()

add_subdirectory(deps/fmt)
add_subdirectory(deps/catch)
add_subdirectory(src)
add_subdirectory(problems)

if(build_tests)
  add_subdirectory(tests)
endif()

if(build_docs)
  add_subdirectory(docs)
endif()
