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

#ifndef __FIELDS_H_
#define __FIELDS_H_

#include "core/grid.hpp"
#include "core/ndptr.hpp"
#include "core/data_adapter.h"
#include "framework/data.h"
#include "utils/logger.h"
#include "utils/stagger.h"
#include <array>

namespace Aperture {

template <int N>
using greater_than_unity = std::enable_if_t<(N > 1), bool>;

template <typename Conf>
class grid_t;

////////////////////////////////////////////////////////////////////////////////
///  Enum type that encodes the stagger of the field.
////////////////////////////////////////////////////////////////////////////////
enum field_type : char {
  face_centered,
  edge_centered,
  cell_centered,
  vert_centered
};

////////////////////////////////////////////////////////////////////////////////
///  Data class that stores an N-component field on the simulation grid.
///
///  \tparam N    Number of components in the field
///  \tparam Conf The Config type for dimensionality, floating type, and other
///               things
////////////////////////////////////////////////////////////////////////////////
template <int N, typename Conf>
class field_t : public data_t {
 public:
  using Grid_t = typename Conf::grid_t;
  field_t(MemType memtype = default_mem_type) : m_memtype(memtype) {}
  field_t(const Grid_t& grid, MemType memtype = default_mem_type);
  field_t(const Grid_t& grid, const vec_t<stagger_t, N> st,
          MemType memtype = default_mem_type);
  field_t(const Grid_t& grid, field_type type,
          MemType memtype = default_mem_type);

  field_t(const field_t<N, Conf>& other) = delete;
  field_t(field_t<N, Conf>&& other) = default;

  field_t<N, Conf>& operator=(const field_t<N, Conf>& other) = delete;
  field_t<N, Conf>& operator=(field_t<N, Conf>&& other) = default;

  void init() override;

  void resize(const Grid_t& grid);
  void assign_dev(const typename Conf::value_t& value);
  void assign_host(const typename Conf::value_t& value);
  void assign(const typename Conf::value_t& value);

  template <typename Func>
  void set_values(int n, const Func& f) {
    if (n >= 0 && n < N) {
      // Logger::print_debug("data[{}] has extent {}x{}", n, m_data[n].extent()[0],
      //                     m_data[n].extent()[1]);
      for (auto idx : m_data[n].indices()) {
        auto pos = idx.get_pos();
        double x0 = m_grid->coord(0, pos[0], m_stagger[n][0]);
        double x1 =
            (Conf::dim > 1 ? m_grid->coord(1, pos[1], m_stagger[n][1]) : 0.0);
        double x2 =
            (Conf::dim > 2 ? m_grid->coord(2, pos[2], m_stagger[n][2]) : 0.0);
        m_data[n][idx] = f(x0, x1, x2);
      }
      if (m_memtype != MemType::host_only) m_data[n].copy_to_device();
    }
  }

  template <typename Func>
  void set_values(const Func& f) {
    for (int n = 0; n < Conf::dim; n++) {
      set_values(
          n, [&f, n](auto x0, auto x1, auto x2) { return f(n, x0, x1, x2); });
    }
  }

  typename Conf::multi_array_t& operator[](int n) { return at(n); }
  const typename Conf::multi_array_t& operator[](int n) const { return at(n); }

  inline typename Conf::multi_array_t& at(int n) {
    if (n < 0) n = 0;
    if (n >= N) n = N - 1;
    return m_data[n];
  }

  inline const typename Conf::multi_array_t& at(int n) const {
    if (n < 0) n = 0;
    if (n >= N) n = N - 1;
    return m_data[n];
  }

  stagger_t stagger(int n = 0) const { return m_stagger[n]; }
  vec_t<stagger_t, N> stagger_vec() const { return m_stagger; }

  void copy_from(const field_t<N, Conf>& other) {
    for (int i = 0; i < N; i++) {
      m_data[i].copy_from(other.m_data[i]);
    }
  }
  void add_by(const field_t<N, Conf>& other, typename Conf::value_t scale = 1.0);

  void copy_to_host() {
    for (int i = 0; i < N; i++) {
      m_data[i].copy_to_host();
    }
  }

  void copy_to_device() {
    for (int i = 0; i < N; i++) {
      m_data[i].copy_to_device();
    }
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_const_t, M> get_const_ptrs() const {
    vec_t<typename Conf::ndptr_const_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].dev_ndptr_const();
    }
    return result;
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_const_t, M> get_ptrs() const {
    return get_const_ptrs();
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_t, M> get_ptrs() {
    vec_t<typename Conf::ndptr_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].dev_ndptr();
    }
    return result;
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_t, M> host_ptrs() {
    vec_t<typename Conf::ndptr_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].host_ndptr();
    }
    return result;
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_const_t, M> host_ptrs() const {
    vec_t<typename Conf::ndptr_const_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].host_ndptr();
    }
    return result;
  }

  void set_memtype(MemType type);

  typename Conf::ndptr_const_t dev_ndptr(int n = 0) const {
    return m_data[n].dev_ndptr_const();
  }
  typename Conf::ndptr_t dev_ndptr(int n = 0) { return m_data[n].dev_ndptr(); }
  typename Conf::ndptr_const_t host_ndptr(int n = 0) const {
    return m_data[n].host_ndptr_const();
  }
  typename Conf::ndptr_t host_ndptr(int n = 0) { return m_data[n].host_ndptr(); }

  const Grid_t& grid() const { return *m_grid; }

 private:
  std::array<typename Conf::multi_array_t, N> m_data;
  vec_t<stagger_t, N> m_stagger;
  const typename Conf::grid_t* m_grid = nullptr;
  MemType m_memtype;
};

template <typename Conf>
using vector_field = field_t<3, Conf>;

template <typename Conf>
using scalar_field = field_t<1, Conf>;

template <typename Conf>
struct host_adapter<field_t<1, Conf>> {
  typedef typename Conf::ndptr_t type;
  typedef typename Conf::ndptr_const_t const_type;

  static inline const_type apply(const field_t<1, Conf>& f) {
    return f.host_ndptr();
  }
  static inline type apply(field_t<1, Conf>& f) {
    return f.host_ndptr();
  }
};

template <int N, typename Conf>
struct host_adapter<field_t<N, Conf>> {
  typedef vec_t<typename Conf::ndptr_t, N> type;
  typedef vec_t<typename Conf::ndptr_const_t, N> const_type;

  static inline const_type apply(const field_t<N, Conf>& f) {
    return f.host_ptrs();
  }
  static inline type apply(field_t<N, Conf>& f) {
    return f.host_ptrs();
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <typename Conf>
struct gpu_adapter<field_t<1, Conf>> {
  typedef typename Conf::ndptr_t type;
  typedef typename Conf::ndptr_const_t const_type;

  static inline const_type apply(const field_t<1, Conf>& f) {
    return f.dev_ndptr();
  }
  static inline type apply(field_t<1, Conf>& f) {
    return f.dev_ndptr();
  }
};

template <int N, typename Conf>
struct gpu_adapter<field_t<N, Conf>> {
  typedef vec_t<typename Conf::ndptr_t, N> type;
  typedef vec_t<typename Conf::ndptr_const_t, N> const_type;

  static inline const_type apply(const field_t<N, Conf>& f) {
    return f.get_const_ptrs();
  }
  static inline type apply(field_t<N, Conf>& f) {
    return f.get_ptrs();
  }
};

#endif

}  // namespace Aperture

#endif  // __FIELDS_H_
