#ifndef __BUFFER_H_
#define __BUFFER_H_

#include "buffer_impl.hpp"
#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include "utils/logger.h"
#include <cstdlib>
#include <type_traits>

#ifdef CUDA_ENABLED
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#endif

namespace Aperture {

/// A class for linear buffers that manages resources both on the host
/// and the device.
template <typename T, MemoryModel Model = default_memory_model>
// template <typename T>
class buffer_t {
 protected:
  size_t m_size = 0;

  mutable T* m_data_h = nullptr;
  mutable T* m_data_d = nullptr;
  // mutable bool m_host_valid = true;
  // mutable bool m_dev_valid = true;
  bool m_host_allocated = false;
  bool m_dev_allocated = false;
  // MemoryModel m_model = default_memory_model;

  void alloc_mem(size_t size) {
    if (Model == MemoryModel::host_only ||
        Model == MemoryModel::host_device) {
      // Allocate host
      m_data_h = new T[size];
      m_host_allocated = true;
    }
#ifdef CUDA_ENABLED
    if (Model != MemoryModel::host_only) {
      if (Model == MemoryModel::device_managed) {
        CudaSafeCall(cudaMallocManaged(&m_data_d, size * sizeof(T)));
        m_data_h = m_data_d;
      } else {
        CudaSafeCall(cudaMalloc(&m_data_d, size * sizeof(T)));
      }
      m_dev_allocated = true;
    }
#endif
    m_size = size;
    // Logger::print_debug("Allocated {} bytes", size * sizeof(T));
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
  static constexpr MemoryModel model() { return Model; }

  buffer_t() {}
  buffer_t(size_t size) { alloc_mem(size); }
  buffer_t(const self_type& other) = delete;
  buffer_t(self_type&& other) { *this = std::move(other); }

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
    // inline T operator[](size_t n) const { return host_ptr()[n]; }
    return host_ptr()[n];
  }

  template <MemoryModel M = Model>
  inline std::enable_if_t<M != MemoryModel::device_only, T&> operator[](
      size_t n) {
    // inline T& operator[](size_t n) { return host_ptr()[n]; }
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

  void assign(size_t start, size_t end, const T& value) {
    // Do not go further than the array size
    end = std::min(m_size, end);
    start = std::min(start, end);
    if (Model == MemoryModel::host_only) {
      if (m_host_allocated)
        ptr_assign(m_data_h, start, end, value);
    } else {
      if (m_dev_allocated)
        ptr_assign_dev(m_data_d, start, end, value);
    }
  }

  void assign(const T& value) { assign(0, m_size, value); }

  void copy_from(const self_type& other, size_t num, size_t src_pos = 0,
                 size_t dest_pos = 0) {
    // Sanitize input
    if (dest_pos + num > m_size) num = m_size - dest_pos;
    if (src_pos + num > other.m_size) num = other.m_size - src_pos;
    if (Model == MemoryModel::host_only) {
      if (m_host_allocated && other.m_host_allocated)
        ptr_copy(other.m_data_h, m_data_h, num, src_pos, dest_pos);
    } else {
      if (m_dev_allocated && other.m_dev_allocated)
        ptr_copy_dev(other.m_data_d, m_data_d, num, src_pos, dest_pos);
    }
  }

  void copy_from(const self_type& other) {
    copy_from(other, other.m_size, 0, 0);
  }

  bool host_allocated() const { return m_host_allocated; }
  bool dev_allocated() const { return m_dev_allocated; }
  size_t size() const { return m_size; }

  const T* data() const {
    if (Model == MemoryModel::host_only ||
        Model == MemoryModel::host_device)
      return m_data_h;
    else
      return m_data_d;
  }
  T* data() {
    if (Model == MemoryModel::host_only ||
        Model == MemoryModel::host_device)
      return m_data_h;
    else
      return m_data_d;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::device_only, const T*> host_ptr()
      const {
    // const T* host_ptr() const {
    // if (!m_host_valid && m_dev_valid) copy_to_host();
    return m_data_h;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::device_only, T*> host_ptr() {
    // T* host_ptr() {
    // m_host_valid = true;
    // m_dev_valid = false;
    return m_data_h;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::host_only, const T*> dev_ptr()
      const {
    // const T* dev_ptr() const {
    // if (!m_dev_valid && m_host_valid) copy_to_device();
    return m_data_d;
  }

  template <MemoryModel M = Model>
  std::enable_if_t<M != MemoryModel::host_only, T*> dev_ptr() {
    // T* dev_ptr() {
    // m_dev_valid = true;
    // m_host_valid = false;
    return m_data_d;
  }

  void copy_to_host() {
    // m_host_valid = true;
    if (Model == MemoryModel::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_size * sizeof(T),
                              cudaMemcpyDeviceToHost));
#endif
    }
  }

  void copy_to_device() {
    // m_dev_valid = true;
    if (Model == MemoryModel::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_size * sizeof(T),
                              cudaMemcpyHostToDevice));
#endif
    }
  }
};

}  // namespace Aperture

#endif  // __BUFFER_H_
