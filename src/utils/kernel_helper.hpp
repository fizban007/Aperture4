/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __KERNEL_HELPER_H_
#define __KERNEL_HELPER_H_

#include "core/gpu_error_check.h"
#include "core/gpu_translation_layer.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

#ifdef CUDA_ENABLED
#include <cuda_occupancy.h>
#endif

#include <iostream>
#include <map>
#include <mutex>

// All of these kernel helprs are taken from hemi:
// https://github.com/harrism/hemi.git

namespace Aperture {

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

class kernel_exec_policy {
 public:
  enum class config_state : uint32_t {
    Automatic = 0,
    SharedMem = 1,
    BlockSize = 2,
    GridSize = 4,
    FullManual = GridSize | BlockSize | SharedMem
  };

  kernel_exec_policy()
      : m_state(0),
        m_grid_size(0),
        m_block_size(0),
        m_shared_mem_bytes(0),
        m_stream((gpuStream_t)0) {}

  kernel_exec_policy(int gridSize, int blockSize) : m_state(0), m_stream(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
    set_shared_mem_bytes(0);
  }

  kernel_exec_policy(int gridSize, int blockSize, size_t sharedMemBytes)
      : m_state(0), m_stream(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
    set_shared_mem_bytes(sharedMemBytes);
  }

  kernel_exec_policy(int gridSize, int blockSize, size_t sharedMemBytes,
                     gpuStream_t stream)
      : m_state(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
    set_shared_mem_bytes(sharedMemBytes);
    set_stream(stream);
  }

  ~kernel_exec_policy() {}

  uint32_t get_config_state() const { return m_state; }

  int get_grid_size() const { return m_grid_size; }
  int get_block_size() const { return m_block_size; }
  int get_max_block_size() const { return m_max_block_size; }
  size_t get_shared_mem_bytes() const { return m_shared_mem_bytes; }
  gpuStream_t get_stream() const { return m_stream; }

  void set_grid_size(int arg) {
    m_grid_size = arg;
    if (m_grid_size > 0)
      set_flag(m_state, config_state::GridSize);
    else
      clear_flag(m_state, config_state::GridSize);
  }

  void set_block_size(int arg) {
    m_block_size = arg;
    if (m_block_size > 0)
      set_flag(m_state, config_state::BlockSize);
    else
      clear_flag(m_state, config_state::BlockSize);
  }

  void set_max_block_size(int arg) { m_max_block_size = arg; }

  void set_shared_mem_bytes(size_t arg) {
    m_shared_mem_bytes = arg;
    set_flag(m_state, config_state::SharedMem);
  }

  void set_stream(gpuStream_t stream) { m_stream = stream; }

 private:
  uint32_t m_state;
  int m_grid_size;
  int m_block_size;
  int m_max_block_size;
  size_t m_shared_mem_bytes;
  gpuStream_t m_stream;
};

class dev_property_cache {
 public:
  // Return a reference to a gpuDeviceProp_t for the current device
  static gpuDeviceProp_t &get() {
    static dev_property_cache instance;

    int devId;
    gpuError_t status = gpuGetDevice(&devId);
    if (status != gpuSuccess) throw status;

    std::lock_guard<std::mutex> guard(instance.mtx);

    if (instance.dpcache.find(devId) == instance.dpcache.end()) {
      // cache miss
      instance.dpcache[devId] = gpuDeviceProp_t();
      status = gpuGetDeviceProperties(&instance.dpcache[devId], devId);
      if (status != gpuSuccess) throw status;
    }
    return instance.dpcache[devId];
  }

 private:
  std::map<int, gpuDeviceProp_t> dpcache;
  std::mutex mtx;
};

#ifdef CUDA_ENABLED
inline size_t
available_shared_bytes_per_block(size_t sharedMemPerMultiprocessor,
                                 size_t sharedSizeBytesStatic, int blocksPerSM,
                                 int smemAllocationUnit) {
  size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM,
                              smemAllocationUnit) -
                 smemAllocationUnit;
  return bytes - sharedSizeBytesStatic;
}
#endif

// template <typename... Args>
// gpuError_t
// configure_grid(kernel_exec_policy &p, void (*k) (Args... args)) {
//   return configure_grid(p, (const void*)k);
// }

template <typename KernelFunc>
gpuError_t
configure_grid(kernel_exec_policy &p, KernelFunc k) {
  uint32_t configState = p.get_config_state();

  if (configState == (uint32_t)kernel_exec_policy::config_state::FullManual)
    return gpuSuccess;

  gpuDeviceProp_t *props;
  try {
    props = &dev_property_cache::get();
  } catch (gpuError_t status) {
    return status;
  }

  gpuFuncAttributes attribs;
  gpuError_t status =
      gpuFuncGetAttributes(&attribs, reinterpret_cast<const void *>(k));
  // gpuError_t status = gpuFuncGetAttributes(&attribs, k);
  if (status != gpuSuccess) return status;

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
  cudaOccDeviceProp occProp(*props);
  cudaOccFuncAttributes occAttrib(attribs);

  cudaFuncCache cacheConfig;
  status = cudaDeviceGetCacheConfig(&cacheConfig);
  if (status != cudaSuccess) return status;
  cudaOccDeviceState occState;
  occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

  int numSMs = props->multiProcessorCount;

  if (!check_flag(configState, kernel_exec_policy::config_state::BlockSize)) {
    int bsize = 0, minGridSize = 0;
    cudaOccError occErr = cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState,
        p.get_shared_mem_bytes());
    if (occErr != CUDA_OCC_SUCCESS || bsize < 0)
      return cudaErrorInvalidConfiguration;
    p.set_block_size(bsize);
  }

  // if ((configState & kernel_exec_policy::GridSize) == 0) {
  if (!check_flag(configState, kernel_exec_policy::config_state::GridSize)) {
    cudaOccResult result;
    cudaOccError occErr = cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, p.get_block_size(),
        p.get_shared_mem_bytes());
    if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
    p.set_grid_size(result.activeBlocksPerMultiprocessor * numSMs);
    if (p.get_grid_size() < numSMs) return cudaErrorInvalidConfiguration;
  }

  // if ((configState & kernel_exec_policy::SharedMem) == 0) {
  if (!check_flag(configState, kernel_exec_policy::config_state::SharedMem)) {
    int smemGranularity = 0;
    cudaOccError occErr =
        cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
    if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
    size_t sbytes = available_shared_bytes_per_block(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(p.get_grid_size(), numSMs), smemGranularity);
    p.set_shared_mem_bytes(sbytes);
  }
#elif defined(HIP_ENABLED) && defined(__HIPCC__)
  int numSMs = props->multiProcessorCount;

  // Find optimal block size if the user doesn't specify one
  if (!check_flag(configState, kernel_exec_policy::config_state::BlockSize) ||
      !check_flag(configState, kernel_exec_policy::config_state::GridSize)) {
    int blockSize = 0, gridSize = 0;
    GpuSafeCall(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, k,
                                                  p.get_shared_mem_bytes(),
                                                  attribs.maxThreadsPerBlock));

    p.set_block_size(blockSize);
    p.set_grid_size(gridSize);
  }

//   if (!check_flag(configState, kernel_exec_policy::config_state::SharedMem))
//   {
//     int smemGranularity = 0;
//     cudaOccError occErr =
//         cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
//     if (occErr != CUDA_OCC_SUCCESS)
//       return cudaErrorInvalidConfiguration;
//     size_t sbytes = available_shared_bytes_per_block(
//         props->sharedMemPerBlock, attribs.sharedSizeBytes,
//         __occDivideRoundUp(p.get_grid_size(), numSMs), smemGranularity);
//     p.set_shared_mem_bytes(sbytes);
//   }
#endif

  // printf("%d %d %ld\n", p.getBlockSize(), p.getGridSize(),
  // p.getSharedMemBytes());
  // Logger::print_debug("block_size: {}, grid_size: {}, shared_mem: {}",
  //                     p.get_block_size(), p.get_grid_size(),
  //                     p.get_shared_mem_bytes());

  return gpuSuccess;
}

#if defined(__CUDACC__) || defined(__HIPCC__)

template <typename Func, typename... Args>
__global__ void
generic_kernel(Func f, Args... args) {
  f(args...);
}

// template <typename... Args>
// __global__ void
// generic_kernel(void (*f) (Args... args), Args... args) {
//   f(args...);
// }

template <typename Func, typename... Args>
void
configure_grid(kernel_exec_policy& policy, Func f, Args... args) {
  GpuSafeCall(configure_grid(policy, generic_kernel<Func, Args...>));
}

// template <typename... Args>
// gpuError_t
// configure_grid(kernel_exec_policy& policy, void (*f) (Args... args)) {
//   GpuSafeCall(configure_grid(policy, f));
// }

// template <typename Func, typename... Args>
// void
// kernel_launch(const kernel_exec_policy &policy, Func f, Args... args) {
//   kernel_exec_policy p = policy;
//   GpuSafeCall(configure_grid(p, generic_kernel<Func, Args...>));
// #ifdef __CUDACC__
//   generic_kernel<<<p.get_grid_size(), p.get_block_size(),
//                    p.get_shared_mem_bytes(), p.get_stream()>>>(f,
//                                                                args...);
// #else
//   hipLaunchKernelGGL(generic_kernel, dim3(p.get_grid_size()),
//                      dim3(p.get_block_size()), p.get_shared_mem_bytes(),
//                      p.get_stream(), args...);
// #endif
//   GpuCheckError();
// }

template <typename Func, typename... Args>
void
kernel_launch(Func f, Args... args) {
  kernel_exec_policy p;
  kernel_launch(p, f, args...);
}

template <typename Func, typename... Args>
void
kernel_launch(const kernel_exec_policy &policy, Func f, Args... args) {
  kernel_exec_policy p = policy;
  GpuSafeCall(configure_grid(p, generic_kernel<Func, Args...>));
  // std::cout << "gridSize: " << p.get_grid_size()
  //           << ", blockSize: " << p.get_block_size()
  //           << ", sharedMem: " << p.get_shared_mem_bytes()
  //           << ", stream: " << (size_t)p.get_stream() << std::endl;
#ifdef __CUDACC__
  generic_kernel<<<p.get_grid_size(), p.get_block_size(),
                   p.get_shared_mem_bytes(), p.get_stream()>>>(f, args...);
#else
  hipLaunchKernelGGL(generic_kernel, dim3(p.get_grid_size()),
                     dim3(p.get_block_size()), p.get_shared_mem_bytes(),
                     p.get_stream(), f, args...);
#endif
  GpuCheckError();
}

template <typename... Args>
void
kernel_launch(const kernel_exec_policy &policy, void (*f)(Args... args),
              Args... args) {
  kernel_exec_policy p = policy;
  GpuSafeCall(configure_grid(p, f));
#ifdef __CUDACC__
  f<<<p.get_grid_size(), p.get_block_size(), p.get_shared_mem_bytes(),
      p.get_stream()>>>(args...);
#else
  hipLaunchKernelGGL(f, dim3(p.get_grid_size()), dim3(p.get_block_size()),
                     p.get_shared_mem_bytes(), p.get_stream(), args...);
#endif
  GpuCheckError();
}

template <typename... Args>
void
kernel_launch(void (*f)(Args... args), Args... args) {
  kernel_exec_policy p;
  kernel_launch(p, f, args...);
}

#endif

#endif

}  // namespace Aperture

#endif  // __KERNEL_HELPER_H_
