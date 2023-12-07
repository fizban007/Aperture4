cmake_minimum_required(VERSION 3.8)
project(Aperture LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

option(use_cuda "Include the Cuda part of the code" OFF)
option(use_double "Use double precision" OFF)
option(use_hip "Include the HIP part of the code" OFF)
option(build_tests "Build the test suite" ON)
option(build_docs "Build the documentation" OFF)
option(lib_only "Only build the core Aperture library" OFF)
option(use_boost_filesystem "Use boost filesystem instead of std::filesystem" OFF)
option(use_libcpp "Use libc++ instead of libstdc++" OFF)
option(build_for_frontier "Building for frontier" OFF)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
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
set(CXX_EXTRA_FLAGS "-DNO_WARN_X86_INTRINSICS -Wno-inconsistent-missing-override")
if(use_boost_filesystem)
set(CXX_EXTRA_FLAGS "${CXX_EXTRA_FLAGS} -D__USE_BOOST_FILESYSTEM__")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D__USE_BOOST_FILESYSTEM__")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_EXTRA_FLAGS}")
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -xHost -no-prec-div")
else()
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native ${CXX_EXTRA_FLAGS}")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(use_libcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()
message(${CMAKE_CXX_COMPILER})
# Find required packages
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  message(${OpenMP_CXX_FLAGS})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if (BOOST_FOUND)
  message(${Boost_INCLUDE_DIRS})
endif()

# Set include directories
set(BOOST_PREPROCESSOR_INCLUDE_DIR deps)
set(FMT_INCLUDE_DIR deps/fmt/include)
set(CATCH_INCLUDE_DIR deps/catch/src)
set(VISIT_STRUCT_INCLUDE_DIR deps/visit_struct/include)
set(CPPTOML_INCLUDE_DIR deps/cpptoml/include)
set(CXXOPT_INCLUDE_DIR deps/cxxopts/include)
set(GSL_INCLUDE_DIR deps/gsl/include)
set(VECTORCLASS_INCLUDE_DIR deps/vectorclass1)

include_directories(
  ${BOOST_PREPROCESSOR_INCLUDE_DIR}
  ${MPI_INCLUDE_PATH}
  ${HDF5_INCLUDE_DIR}
  ${FMT_INCLUDE_DIR}
  ${CATCH_INCLUDE_DIR}
  ${VISIT_STRUCT_INCLUDE_DIR}
  ${CPPTOML_INCLUDE_DIR}
  ${CXXOPT_INCLUDE_DIR}
  ${GSL_INCLUDE_DIR}
  ${VECTORCLASS_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  src)

# Check for cuda support
if(use_cuda)
  # if(POLICY CMP0104)
  #   cmake_policy(SET CMP0104 NEW)
  # endif()

  message("Checking cuda support")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_compile_definitions(CUDA_ENABLED)
    message(
      "Enabling Cuda, ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS}"
    )
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                        ${CUDA_INCLUDE_DIRS})
    #set(CUDA_EXTRA_FLAGS "-DNO_WARN_X86_INTRINSICS")
    set(CUDA_EXTRA_FLAGS "-Wno-unused-variable")
    set(CMAKE_CUDA_FLAGS
        "--expt-extended-lambda --expt-relaxed-constexpr\
      -lcusparse -lcublas -lcurand -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored \
      -Xcudafe --diag_suppress=code_is_unreachable ${CUDA_EXTRA_FLAGS}")
    # set_property(TARGET myTarget PROPERTY CUDA_ARCHITECTURES 35 50 72)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # message(${CMAKE_CXX_COMPILER})
    find_library(CUDART_LIBRARY cudart
                 PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUSPARSE_LIBRARY cusparse
                 PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()

if(use_double)
  add_compile_definitions(USE_DOUBLE)
endif()

# if(use_simd)
#   add_compile_definitions(USE_SIMD)
# endif()

add_subdirectory(deps/fmt)
add_subdirectory(deps/catch)
link_libraries(stdc++fs)

if(use_hip)
  # Search for rocm in common locations
  # list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  # Find hip
  # find_package(hip REQUIRED CONFIG)
  # find_package(hip MODULE REQUIRED)
  message("Checking hip support")
  include(CheckLanguage)
  check_language(HIP)
  if (CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    add_compile_definitions(HIP_ENABLED)
    message(
      "Enabling HIP, include path is ${HSA_HEADER}"
    )
    include_directories(${HSA_HEADER})
    # find_package(HIP)
    # find_package(hip)
    find_package(rocthrust)
    find_package(rocrand)
    find_package(rocsparse)
    find_package(hipcub)
    set(MPICH_DIR $ENV{MPICH_DIR})
    message("mpich_dir is ${MPICH_DIR}")
    include_directories("${MPICH_DIR}/include")
    set(HDF5_DIR $ENV{HDF5_DIR})
    include_directories("${HDF5_DIR}/include")
    set(CRAY_XPMEM_POST_LINK_OPTS $ENV{CRAY_XPMEM_POST_LINK_OPTS})
    message(${CRAY_XPMEM_POST_LINK_OPTS})
    # set(PE_MPICH_GTL_DIR_amd_gfx90a $ENV{PE_MPICH_GTL_DIR_amd_gfx90a})
    STRING(REGEX REPLACE "-L([^ ]*)" "\\1" PE_MPICH_GTL_DIR_amd_gfx90a $ENV{PE_MPICH_GTL_DIR_amd_gfx90a})
    message(${PE_MPICH_GTL_DIR_amd_gfx90a})
    link_directories(${PE_MPICH_GTL_DIR_amd_gfx90a})
    set(CMAKE_HIP_COMPILER CC)
    # set(PE_MPICH_GTL_LIBS_amd_gfx90a $ENV{PE_MPICH_GTL_LIBS_amd_gfx90a})
    # message(${PE_MPICH_GTL_LIBS_amd_gfx90a})
    set(ROCM_PATH $ENV{ROCM_PATH})
    include_directories("${ROCM_PATH}/include")
    # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fgpu-rdc -Wno-inconsistent-missing-override -Wno-deprecated-register -munsafe-fp-atomics -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a ${CRAY_XPMEM_POST_LINK_OPTS}")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fgpu-rdc -Wno-inconsistent-missing-override -Wno-deprecated-register -munsafe-fp-atomics --rocm-path=${ROCM_PATH} --offload-arch=gfx90a")
    set(CMAKE_HIP_LINK_FLAGS "${CMAKE_HIP_LINK_FLAGS} --rocm-path=${ROCM_PATH} ${CRAY_XPMEM_POST_LINK_OPTS}")
    set(CMAKE_HIP_LINK_EXECUTABLE "CC <FLAGS> <CMAKE_HIP_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
    # set(CMAKE_CXX_LINK_EXECUTABLE "hipcc <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
    if(use_libcpp)
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -stdlib=libc++")
    endif()

    if($ENV{MPICH_GPU_SUPPORT_ENABLED} EQUAL 1)
      message("Using gpu-aware mpich")
      add_compile_definitions(GPU_AWARE_MPICH)
    endif()

    if(NOT DEFINED ENV{HIP_PLATFORM})
      set(ENV{HIP_PLATFORM} "amd")
    endif()

    if($ENV{HIP_PLATFORM} STREQUAL "amd")
      message("Using amd platform")
      add_compile_definitions(__HIP_PLATFORM_AMD__)
    elseif($ENV{HIP_PLATFORM} STREQUAL "nvidia")
      message("Using nvidia platform")
      add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
    endif()
  endif()
endif()

function(set_compile_gpu)
  if (use_cuda AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  elseif (use_hip AND CMAKE_HIP_COMPILER)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
  endif()
endfunction()

function(add_aperture_executable target_name)
  set_compile_gpu(${ARGN})
  add_executable(${target_name} ${ARGN})
  if (NOT build_for_frontier)
    target_link_libraries(${target_name}
      Aperture
      $<TARGET_NAME_IF_EXISTS:Aperture_dev>)
  endif()
endfunction()

function(add_aperture_library target_name)
  set_compile_gpu(${ARGN})
  add_library(${target_name} ${ARGN})
  if (NOT build_for_frontier)
    target_link_libraries(${target_name}
      Aperture
      $<TARGET_NAME_IF_EXISTS:Aperture_dev>)
  endif()
endfunction()

function(link_dependent_libraries target_name)
  target_link_libraries(${target_name} PUBLIC
    fmt::fmt ${MPI_LIBRARIES}
    ${HDF5_LIBRARIES} ${Boost_LIBRARIES}
  )
  if (use_cuda AND CMAKE_CUDA_COMPILER)
    target_link_libraries(${target_name} PUBLIC ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
    # target_link_libraries(Aperture ${CUDART_LIBRARY} ${CUSPARSE_LIBRARY})
  endif()

  if (use_hip AND CMAKE_HIP_COMPILER)
    # target_link_libraries(Aperture_dev hip::host hip::device roc::rocthrust roc::rocrand)
    target_link_libraries(${target_name} PUBLIC hip::host hip::device roc::rocthrust amdhip64 mpi xpmem mpi_gtl_hsa)
    # target_link_libraries(Aperture hip::host hip::device roc::rocthrust)
  endif()
endfunction()

add_subdirectory(src)
add_subdirectory(problems)

if(build_tests)
  add_subdirectory(tests)
endif()

if(build_docs)
  add_subdirectory(docs)
endif()

