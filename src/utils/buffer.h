#ifndef __BUFFER_H_
#define __BUFFER_H_

#include "buffer_impl.hpp"
#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include "utils/logger.h"
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

#ifdef CUDA_ENABLED
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#endif

namespace Aperture {

/// A class for linear buffers that manages resources both on the host
/// and the device.
// template <typename T, MemType Model = default_mem_type>
template <typename T>
class buffer_t {
 protected:
  size_t m_size = 0;

  mutable T* m_data_h = nullptr;
  mutable T* m_data_d = nullptr;
  // mutable bool m_host_valid = true;
  // mutable bool m_dev_valid = true;
  bool m_host_allocated = false;
  bool m_dev_allocated = false;
  MemType m_model = default_mem_type;

  void alloc_mem(size_t size) {
    if (m_model == MemType::host_only || m_model == MemType::host_device) {
      // Allocate host
      m_data_h = new T[size];
      m_host_allocated = true;
    }
#ifdef CUDA_ENABLED
    if (m_model != MemType::host_only) {
      if (m_model == MemType::device_managed) {
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
  // typedef buffer_t<T, Model> self_type;
  typedef buffer_t<T> self_type;
  // static constexpr MemType model() { return Model; }
  MemType mem_type() const { return m_model; }
  void set_memtype(MemType type) { m_model = type; }

  buffer_t(MemType model = default_mem_type) : m_model(model) {}
  buffer_t(size_t size, MemType model = default_mem_type) : m_model(model) {
    alloc_mem(size);
  }
  buffer_t(const self_type& other) = delete;
  buffer_t(self_type&& other) { *this = std::move(other); }

  ~buffer_t() { free_mem(); }

  self_type& operator=(const self_type& other) = delete;
  self_type& operator=(self_type&& other) {
    m_model = other.m_model;
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

  ///  Subscript operator, only defined if this is not device_only
  // template <MemType M = Model>
  // inline std::enable_if_t<M != MemType::device_only, T> operator[](
  //     size_t n) const {
  T operator[](size_t n) const {
    // inline T operator[](size_t n) const { return host_ptr()[n]; }
    return m_data_h[n];
  }

  ///  Subscript operator, only defined if this is not device_only
  // template <MemType M = Model>
  // inline std::enable_if_t<M != MemType::device_only, T&> operator[](
  // size_t n) {
  T& operator[](size_t n) {
    // inline T& operator[](size_t n) { return host_ptr()[n]; }
    return m_data_h[n];
  }

  /// Resize the buffer to a given @size.
  void resize(size_t size) {
    if (m_host_allocated || m_dev_allocated) {
      free_mem();
    }
    alloc_mem(size);
    // m_host_valid = true;
    // m_dev_valid = true;
  }

  /// Assign a single value to part of the buffer, host version
  void assign_host(size_t start, size_t end, const T& value) {
    // Do not go further than the array size
    end = std::min(m_size, end);
    start = std::min(start, end);
    if (m_host_allocated) ptr_assign(m_data_h, start, end, value);
  }

  /// Assign a single value to part of the buffer, device version
  void assign_dev(size_t start, size_t end, const T& value) {
    // Do not go further than the array size
    end = std::min(m_size, end);
    start = std::min(start, end);
    if (m_dev_allocated) ptr_assign_dev(m_data_d, start, end, value);
  }

  /// Assign a single value to part of the buffer. Calls the host or device
  /// version depending on the memory location
  void assign(size_t start, size_t end, const T& value) {
    if (m_model == MemType::host_only) {
      assign_host(start, end, value);
    } else {
      assign_dev(start, end, value);
    }
  }

  /// Assign a value to the whole buffer. Calls the host or device version
  /// depending on the memory location
  void assign(const T& value) { assign(0, m_size, value); }

  /// Assign a value to the whole buffer. Host version
  void assign_host(const T& value) { assign_host(0, m_size, value); }

  /// Assign a value to the whole buffer. Device version
  void assign_dev(const T& value) { assign_dev(0, m_size, value); }

  ///  Copy a part from another buffer.
  ///  \param other  The other buffer that we are copying from
  ///  \param num    Number of elements to copy
  ///  \param src_pos   Starting position in the other buffer
  ///  \param dest_pos  Starting position in this buffer (the target)
  void copy_from(const self_type& other, size_t num, size_t src_pos = 0,
                 size_t dest_pos = 0) {
    // Sanitize input
    if (dest_pos + num > m_size) num = m_size - dest_pos;
    if (src_pos + num > other.m_size) num = other.m_size - src_pos;
    if (m_model == MemType::host_only) {
      if (m_host_allocated && other.m_host_allocated)
        ptr_copy(other.m_data_h, m_data_h, num, src_pos, dest_pos);
    } else {
      if (m_dev_allocated && other.m_dev_allocated)
        ptr_copy_dev(other.m_data_d, m_data_d, num, src_pos, dest_pos);
    }
  }

  ///  Copy from the whole other buffer
  void copy_from(const self_type& other) {
    copy_from(other, other.m_size, 0, 0);
  }

  ///  Place some values directly at and after @pos. Very useful for
  ///  initialization.
  void emplace(size_t pos, const std::initializer_list<T>& list) {
    if (m_model == MemType::device_only) return;
    for (auto& t : list) {
      if (pos >= m_size) break;
      m_data_h[pos] = t;
      pos += 1;
    }
  }

  bool host_allocated() const { return m_host_allocated; }
  bool dev_allocated() const { return m_dev_allocated; }
  size_t size() const { return m_size; }

  const T* data() const {
    if (m_model == MemType::host_only || m_model == MemType::host_device)
      return m_data_h;
    else
      return m_data_d;
  }
  T* data() {
    if (m_model == MemType::host_only || m_model == MemType::host_device)
      return m_data_h;
    else
      return m_data_d;
  }

  // template <MemType M = Model>
  // std::enable_if_t<M != MemType::device_only, const T*> host_ptr() const {
  const T* host_ptr() const { return m_data_h; }

  // template <MemType M = Model>
  // std::enable_if_t<M != MemType::device_only, T*> host_ptr() {
  T* host_ptr() { return m_data_h; }

  // template <MemType M = Model>
  // std::enable_if_t<M != MemType::host_only, const T*> dev_ptr() const {
  const T* dev_ptr() const { return m_data_d; }

  // template <MemType M = Model>
  // std::enable_if_t<M != MemType::host_only, T*> dev_ptr() {
  T* dev_ptr() { return m_data_d; }

  void copy_to_host() {
    if (m_model == MemType::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_size * sizeof(T),
                              cudaMemcpyDeviceToHost));
#endif
    }
  }

  void copy_to_device() {
    if (m_model == MemType::host_device) {
#ifdef CUDA_ENABLED
      CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_size * sizeof(T),
                              cudaMemcpyHostToDevice));
#endif
    }
  }
};

}  // namespace Aperture

#endif  // __BUFFER_H_
