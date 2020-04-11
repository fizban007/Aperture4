#ifndef __KERNEL_HELPER_H_
#define __KERNEL_HELPER_H_

#include "core/cuda_control.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <cuda_occupancy.h>
#include <map>
#include <mutex>

// All of these kernel helprs are taken from hemi:
// https://github.com/harrism/hemi.git

namespace Aperture {

#ifdef CUDA_ENABLED

class exec_policy {
 public:
  enum class config_state : uint32_t {
    Automatic = 0,
    SharedMem = 1,
    BlockSize = 2,
    GridSize = 4,
    FullManual = GridSize | BlockSize | SharedMem
  };

  exec_policy()
      : m_state(0),
        m_grid_size(0),
        m_block_size(0),
        m_shared_mem_bytes(0),
        m_stream((cudaStream_t)0) {}

  exec_policy(int gridSize, int blockSize)
      : m_state(0), m_stream(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
  }

  exec_policy(int gridSize, int blockSize, size_t sharedMemBytes)
      : m_state(0), m_stream(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
    set_shared_mem_bytes(sharedMemBytes);
  }

  exec_policy(int gridSize, int blockSize, size_t sharedMemBytes,
              cudaStream_t stream)
      : m_state(0) {
    set_grid_size(gridSize);
    set_block_size(blockSize);
    set_shared_mem_bytes(sharedMemBytes);
    set_stream(stream);
  }

  ~exec_policy() {}

  uint32_t get_config_state() const { return m_state; }

  int get_grid_size() const { return m_grid_size; }
  int get_block_size() const { return m_block_size; }
  int get_max_block_size() const { return m_max_block_size; }
  size_t get_shared_mem_bytes() const { return m_shared_mem_bytes; }
  cudaStream_t get_stream() const { return m_stream; }

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

  void set_stream(cudaStream_t stream) { m_stream = stream; }

 private:
  uint32_t m_state;
  int m_grid_size;
  int m_block_size;
  int m_max_block_size;
  size_t m_shared_mem_bytes;
  cudaStream_t m_stream;
};

class dev_property_cache {
 public:
  // Return a reference to a cudaDeviceProp for the current device
  static cudaDeviceProp &get() {
    static dev_property_cache instance;

    int devId;
    cudaError_t status = cudaGetDevice(&devId);
    if (status != cudaSuccess) throw status;

    std::lock_guard<std::mutex> guard(instance.mtx);

    if (instance.dpcache.find(devId) == instance.dpcache.end()) {
      // cache miss
      instance.dpcache[devId] = cudaDeviceProp();
      status = cudaGetDeviceProperties(&instance.dpcache[devId], devId);
      if (status != cudaSuccess) throw status;
    }
    return instance.dpcache[devId];
  }

 private:
  std::map<int, cudaDeviceProp> dpcache;
  std::mutex mtx;
};

inline size_t
available_shared_bytes_per_block(size_t sharedMemPerMultiprocessor,
                                 size_t sharedSizeBytesStatic,
                                 int blocksPerSM,
                                 int smemAllocationUnit) {
  size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM,
                              smemAllocationUnit) -
                 smemAllocationUnit;
  return bytes - sharedSizeBytesStatic;
}

template <typename KernelFunc>
cudaError_t
configure_grid(exec_policy &p, KernelFunc k) {
  uint32_t configState = p.get_config_state();

  if (configState == (uint32_t)exec_policy::config_state::FullManual)
    return cudaSuccess;

  cudaDeviceProp *props;
  try {
    props = &dev_property_cache::get();
  } catch (cudaError_t status) {
    return status;
  }

  cudaFuncAttributes attribs;
  cudaOccDeviceProp occProp(*props);

  cudaError_t status = cudaFuncGetAttributes(&attribs, k);
  if (status != cudaSuccess) return status;
  cudaOccFuncAttributes occAttrib(attribs);

  cudaFuncCache cacheConfig;
  status = cudaDeviceGetCacheConfig(&cacheConfig);
  if (status != cudaSuccess) return status;
  cudaOccDeviceState occState;
  occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

  int numSMs = props->multiProcessorCount;

  if (!check_flag(configState, exec_policy::config_state::BlockSize)) {
    int bsize = 0, minGridSize = 0;
    cudaOccError occErr = cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState,
        p.get_shared_mem_bytes());
    if (occErr != CUDA_OCC_SUCCESS || bsize < 0)
      return cudaErrorInvalidConfiguration;
    p.set_block_size(bsize);
  }

  // if ((configState & exec_policy::GridSize) == 0) {
  if (!check_flag(configState, exec_policy::config_state::GridSize)) {
    cudaOccResult result;
    cudaOccError occErr = cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, p.get_block_size(),
        p.get_shared_mem_bytes());
    if (occErr != CUDA_OCC_SUCCESS)
      return cudaErrorInvalidConfiguration;
    p.set_grid_size(result.activeBlocksPerMultiprocessor * numSMs);
    if (p.get_grid_size() < numSMs)
      return cudaErrorInvalidConfiguration;
  }

  // if ((configState & exec_policy::SharedMem) == 0) {
  if (!check_flag(configState, exec_policy::config_state::SharedMem)) {
    int smemGranularity = 0;
    cudaOccError occErr =
        cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
    if (occErr != CUDA_OCC_SUCCESS)
      return cudaErrorInvalidConfiguration;
    size_t sbytes = available_shared_bytes_per_block(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(p.get_grid_size(), numSMs), smemGranularity);
    p.set_shared_mem_bytes(sbytes);
  }

  // printf("%d %d %ld\n", p.getBlockSize(), p.getGridSize(),
  // p.getSharedMemBytes());
  Logger::print_debug("block_size: {}, grid_size: {}, shared_mem: {}",
                      p.get_block_size(), p.get_grid_size(),
                      p.get_shared_mem_bytes());

  return cudaSuccess;
}

#ifdef __CUDACC__

template <typename Func, typename... Args>
__global__ void
generic_kernel(Func f, Args... args) {
  f(args...);
}

template <typename Func, typename... Args>
void
kernel_launch(const exec_policy &policy, Func f, Args... args) {
  exec_policy p = policy;
  CudaSafeCall(configure_grid(p, generic_kernel<Func, Args...>));
  generic_kernel<<<p.get_grid_size(), p.get_block_size(),
                   p.get_shared_mem_bytes(), p.get_stream()>>>(f,
                                                               args...);
  CudaCheckError();
}

template <typename Func, typename... Args>
void
kernel_launch(Func f, Args... args) {
  exec_policy p;
  kernel_launch(p, f, args...);
}

template <typename... Args>
void
kernel_launch(const exec_policy &policy, void (*f)(Args... args),
              Args... args) {
  exec_policy p = policy;
  CudaSafeCall(configure_grid(p, f));
  f<<<p.get_grid_size(), p.get_block_size(), p.get_shared_mem_bytes(),
      p.get_stream()>>>(args...);
  CudaCheckError();
}

template <typename... Args>
void
kernel_launch(void (*f)(Args... args), Args... args) {
  exec_policy p;
  kernel_launch(p, f, args...);
}

#endif

#endif

}  // namespace Aperture

#endif  // __KERNEL_HELPER_H_
