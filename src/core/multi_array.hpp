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

#ifndef __MULTI_ARRAY_H_
#define __MULTI_ARRAY_H_

#include "core/buffer.hpp"
#include "ndptr.hpp"
#include "typedefs_and_constants.h"
#include "utils/index.hpp"
#include "utils/range.hpp"
#include "utils/type_traits.hpp"
#include "utils/vec.hpp"
#include <exception>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is the multi-dimensional array class of *Aperture*.
///
///  Since all underlying memory is linear, a multi-dimensional array is simply
///  a linear segment of memory with an indexing scheme. The structure of this
///  class reflect that. The multi_array class inherits the \ref buffer class
///  (so that it inherits the host-device copying functionality), extends it
///  with an indexing scheme, and keeps track of its n-dimensional size.
///
///  \tparam T      The datatype stored in the multi_array
///  \tparam Rank   The dimensionality of the multi_array
///  \tparam Idx_t  An indexing scheme for the multi_array
////////////////////////////////////////////////////////////////////////////////
template <typename T, int Rank, typename Idx_t = default_idx_t<Rank>>
class multi_array : public buffer<T> {
 private:
  extent_t<Rank> m_ext;

 public:
  typedef multi_array<T, Rank, Idx_t> self_type;
  typedef T value_t;
  typedef Idx_t idx_t;
  typedef ndptr<T, Rank, Idx_t> ptr_t;
  typedef ndptr_const<T, Rank, Idx_t> const_ptr_t;

  /// Default constructor, initialize an empty multi_array
  multi_array(MemType type = default_mem_type) : m_ext{}, buffer<T>(type) {}

  /// Initialize a multi_array specifying its dimensions using a number of
  /// integers.
  ///
  /// For example,
  ///
  ///     multi_array<2, float> v(32, 32);
  ///
  /// This will initialize a 2D `float` array of size 32x32.
  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  multi_array(Args... args)
      : m_ext(args...), buffer<T>(extent_t<Rank>(args...).size()) {
    check_dimension();
  }

  /// Initialize a multi_array specifying its dimensions using an extent_t
  /// object, and optionally specifying its memory location.
  ///
  /// For example,
  ///
  ///     multi_array<2, float> v(extent(32, 32), MemType::host_device);
  ///
  /// This will initialize a 2D `float` array of size 32x32, both allocating on
  /// the device and the host.
  multi_array(const extent_t<Rank>& extent, MemType type = default_mem_type)
      : m_ext(extent), buffer<T>(extent.size(), type) {
    check_dimension();
  }

  // Disallow copy
  multi_array(const self_type& other) = delete;

  // Only allow move
  multi_array(self_type&& other)
      : m_ext(other.m_ext), buffer<T>(std::move(other)) {
    other.m_ext = extent_t<Rank>{};
  }

  ~multi_array() {}

  using buffer<T>::assign;

  void copy_from(const self_type& other) { buffer<T>::copy_from(other); }

  void resize(const extent_t<Rank>& ext) {
    m_ext = ext;
    m_ext.get_strides();
    buffer<T>::resize(ext.size());
  }

  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  void resize(Args... args) {
    resize(extent(args...));
  }

  void check_dimension() {
    if (std::is_same<Idx_t, idx_zorder_t<Rank>>::value) {
      if (not_power_of_two(m_ext)) {
        throw std::range_error(
            "One of the dimensions is not a power of 2, can't use "
            "zorder "
            "indexing!");
      }
    }
  }

  self_type& operator=(const self_type& other) = delete;

  self_type& operator=(self_type&& other) {
    buffer<T>::operator=(std::move(other));
    m_ext = other.m_ext;
    other.m_ext = extent_t<Rank>{};
    return *this;
  }

  using buffer<T>::operator[];

  inline T operator[](const Idx_t& idx) const {
    return this->m_data_h[idx.linear];
  }
  inline T& operator[](const Idx_t& idx) { return this->m_data_h[idx.linear]; }

  inline T at(const index_t<Rank>& pos) const {
    return this->m_data_h[Idx_t(pos, m_ext).linear];
  }
  inline T& at(const index_t<Rank>& pos) {
    return this->m_data_h[Idx_t(pos, m_ext).linear];
  }

  HD_INLINE T at_dev(const index_t<Rank>& pos) const {
    return this->m_data_d[Idx_t(pos, m_ext).linear];
  }
  HD_INLINE T& at_dev(const index_t<Rank>& pos) {
    return this->m_data_d[Idx_t(pos, m_ext).linear];
  }

  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  inline T operator()(Args... args) const {
    auto idx = get_idx(args...);
    return this->m_data_h[idx.linear];
  }

  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  inline T& operator()(Args... args) {
    auto idx = get_idx(args...);
    return this->m_data_h[idx.linear];
  }

  template <typename... Args, typename = all_convertible_to<uint32_t, Args...>>
  inline Idx_t get_idx(Args... args) {
    auto idx = Idx_t(index_t<Rank>(args...), m_ext);
    return idx;
  }

  inline Idx_t get_idx(index_t<Rank> pos) const { return Idx_t(pos, m_ext); }

  inline Idx_t idx_at(uint64_t n) const { return Idx_t(n, m_ext); }

  inline ptr_t dev_ndptr() { return ptr_t(this->m_data_d); }
  inline ptr_t host_ndptr() { return ptr_t(this->m_data_h); }
  inline const_ptr_t dev_ndptr_const() const {
    return const_ptr_t(this->m_data_d);
  }
  inline const_ptr_t host_ndptr_const() const {
    return const_ptr_t(this->m_data_h);
  }

  const extent_t<Rank>& extent() const { return m_ext; }

  inline Idx_t begin() const { return idx_at(0); }
  inline Idx_t end() const { return idx_at(this->m_size); }
  inline range_proxy<Idx_t> indices() const { return range(begin(), end()); }

  class cref_t {
   public:
    typedef multi_array<T, Rank, Idx_t> multi_array_t;
    typedef T value_t;
    typedef Idx_t idx_t;

    cref_t() = default;
    cref_t(const multi_array_t& array)
        : m_ptr(array.host_ndptr_const()),
          m_dev_ptr(array.dev_ndptr_const()),
          m_ext(array.extent()) {}
    HOST_DEVICE cref_t(const cref_t& other) = default;
    HOST_DEVICE ~cref_t() {}

    // HD_INLINE const value_t& operator[](const idx_t& idx) const {
    HD_INLINE value_t operator[](const idx_t& idx) const {
#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__)
      return m_dev_ptr[idx];
#else
      return m_ptr[idx];
#endif
    }

    value_t at(const index_t<Rank>& pos) const {
      return m_ptr[Idx_t(pos, m_ext)];
    }
    HD_INLINE value_t at_dev(const index_t<Rank>& pos) const {
      return m_dev_ptr[Idx_t(pos, m_ext)];
    }

    HD_INLINE const extent_t<Rank>& ext() const { return m_ext; }
    const const_ptr_t& ptr() const { return m_ptr; }
    HD_INLINE const const_ptr_t& dev_ptr() const { return m_dev_ptr; }

   private:
    const_ptr_t m_ptr;
    const_ptr_t m_dev_ptr;
    extent_t<Rank> m_ext;
  };

  cref_t cref() const { return cref_t(*this); }

  class ref_t {
   public:
    typedef multi_array<T, Rank, Idx_t> multi_array_t;
    typedef T value_t;
    typedef Idx_t idx_t;

    ref_t() = default;
    ref_t(multi_array_t& array)
        : m_ptr(array.host_ndptr()),
          m_dev_ptr(array.dev_ndptr()),
          m_ext(array.extent()) {}
    HOST_DEVICE ref_t(const ref_t& other) = default;
    HOST_DEVICE ~ref_t() {}

    HD_INLINE value_t& operator[](const idx_t& idx) {
#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__)
      return m_dev_ptr[idx];
#else
      return m_ptr[idx];
#endif
    }
    value_t& at(const index_t<Rank>& pos) { return m_ptr[Idx_t(pos, m_ext)]; }
    HD_INLINE value_t& at_dev(const index_t<Rank>& pos) {
      return m_dev_ptr[Idx_t(pos, m_ext)];
    }

    const extent_t<Rank>& ext() const { return m_ext; }

   private:
    ptr_t m_ptr;
    ptr_t m_dev_ptr;
    extent_t<Rank> m_ext;
  };

  ref_t ref() { return ref_t(*this); }
};

/// Helper function to construct a multi_array.
/**
 *  This can be useful for initializing a multi_array without specifying
 *  template parameters (specifically the dimensionality, which is deduced from
 *  the supplied extent_t struct). For example:
 *
 *      auto array = make_multi_array<float>(extent(32, 32),
 * MemType::host_only);
 *
 *  This will create a 2D `float` multi_array with size 32x32 that lives on the
 *  host only.
 */
template <typename T, template <int> class Index_t = default_idx_t, int Rank>
auto
make_multi_array(const extent_t<Rank>& ext, MemType type = default_mem_type) {
  return multi_array<T, Rank, Index_t<Rank>>(ext, type);
}

template <typename T, int Rank, typename Idx_t>
struct host_adapter<multi_array<T, Rank, Idx_t>> {
  typedef ndptr<T, Rank, Idx_t> type;
  typedef ndptr_const<T, Rank, Idx_t> const_type;

  static inline type apply(multi_array<T, Rank, Idx_t>& array) {
    return array.host_ndptr();
  }
  static inline const_type apply(const multi_array<T, Rank, Idx_t>& array) {
    return array.host_ndptr_const();
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <typename T, int Rank, typename Idx_t>
struct gpu_adapter<multi_array<T, Rank, Idx_t>> {
  typedef ndptr<T, Rank, Idx_t> type;
  typedef ndptr_const<T, Rank, Idx_t> const_type;

  static inline type apply(multi_array<T, Rank, Idx_t>& array) {
    return array.dev_ndptr();
  }
  static inline const_type apply(const multi_array<T, Rank, Idx_t>& array) {
    return array.dev_ndptr_const();
  }
};

#endif

}  // namespace Aperture

#endif  // __MULTI_ARRAY_H_
