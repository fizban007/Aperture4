#ifndef __BUFFER_H_
#define __BUFFER_H_

#include "cuda_control.h"
#include "typedefs_and_constants.h"
#include "enum_types.h"
#include "utils/logger.h"
#include <cstdlib>
#include <type_traits>

namespace Aperture {

/// A class for linear buffers that manages resources both on the host
/// and the device.
template <typename T, MemoryModel Model = default_memory_model>
class buffer_t {
 protected:
  size_t m_size = 0;

  mutable T* m_data_h = nullptr;
  mutable T* m_data_d = nullptr;
  // mutable bool m_host_valid = true;
  // mutable bool m_dev_valid = true;
  bool m_host_allocated = false;
  bool m_dev_allocated = false;

  void alloc_mem(size_t size) {
    if constexpr (Model == MemoryModel::host_only ||
                  Model == MemoryModel::host_device) {
      // Allocate host
      m_data_h = new T[size];
      m_host_allocated = true;
    }
#ifdef CUDA_ENABLED
    if constexpr (Model != MemoryModel::host_only) {
      if constexpr (Model == MemoryModel::device_managed) {
        CudaSafeCall(cudaMallocManaged(&m_data_d, size * sizeof(T)));
        m_data_h = m_data_d;
      } else {
        CudaSafeCall(cudaMalloc(&m_data_d, size * sizeof(T)));
      }
      m_dev_allocated = true;
    }
#endif
    m_size = size;
    Logger::print_debug("Allocated {} bytes", size * sizeof(T));
  }

  void free_mem() {
    if (m_host_allocated) {
      delete[] m_data_h;
      m_data_h = nullptr;
      m_host_allocated = false;
    }
#ifdef CUDA_ENABLED
    if (m_dev_allocated) {
      CudaSafeCall(cudaFree(m_data_d));
      m_data_d = nullptr;
      m_dev_allocated = false;
    }
#endif
  }

 public:
  typedef buffer_t<T, Model> self_type;

  buffer_t() {}
  buffer_t(size_t size) { alloc_mem(size); }
  buffer_t(const self_type& other) = delete;
  buffer_t(self_type&& other) {
    *this = std::move(other);
  }

  ~buffer_t() { free_mem(); }

  self_type& operator=(const self_type& other) = delete;
  self_type& operator=(self_type&& other) {
    m_size = other.m_size;
    other.m_size = 0;

    m_host_allocated = other.m_host_allocated;
    m_dev_allocated = other.m_dev_allocated;
    other.m_host_allocated = false;
    other.m_dev_allocated = false;

    m_data_h = other.m_data_h;
    m_data_d = other.m_data_d;
    other.m_data_h = nullptr;
    other.m_data_d = nullptr;

    return *this;
  }

  template <MemoryModel M = Model>
  inline std::enable_if_t<M != MemoryModel::device_only, T> operator[](
      size_t n) const {
    return host_ptr()[n];
  }

  template <MemoryModel M = Model>
  inline std::enable_if_t<M != MemoryModel::device_only, T&> operator[](
      size_t n) {
    return host_ptr()[n];
  }

  void resize(size_t size) {
    if (m_host_allocated || m_dev_allocated) {
      free_mem();
    }
    alloc_mem(size);
    // m_host_valid = true;
    // m_dev_valid = true;
  }

  bool host_allocated() const { return m_host_allocated; }
  bool dev_allocated() const { return m_dev_allocated; }
  size_t size() const { return m_size; }

  const T* data() const {
    if constexpr (Model == MemoryModel::host_only || Model == MemoryModel::host_device)
      return host_ptr();
    else
      return dev_ptr();
  }
  T* data() {
    if constexpr (Model == MemoryModel::host_only || Model == MemoryModel::host_device)
      return host_ptr();
    else
      return dev_ptr();
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::device_only, const T*> host_ptr()
      const {
    // if (!m_host_valid && m_dev_valid) copy_to_host();
    return m_data_h;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::device_only, T*> host_ptr() {
    // m_host_valid = true;
    // m_dev_valid = false;
    return m_data_h;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::host_only, const T*> dev_ptr()
      const {
    // if (!m_dev_valid && m_host_valid) copy_to_device();
    return m_data_d;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::host_only, T*> dev_ptr() {
    // m_dev_valid = true;
    // m_host_valid = false;
    return m_data_d;
  }

  void copy_to_host() {
    // m_host_valid = true;
    if constexpr (Model == MemoryModel::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_size * sizeof(T),
                              cudaMemcpyDeviceToHost));
#endif
    }
  }

  void copy_to_device() {
    // m_dev_valid = true;
    if constexpr (Model == MemoryModel::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_size * sizeof(T),
                              cudaMemcpyHostToDevice));
#endif
    }
  }
};

}  // namespace Aperture

#endif  // __BUFFER_H_
