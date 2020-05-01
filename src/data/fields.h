#ifndef __FIELDS_H_
#define __FIELDS_H_

#include "core/grid.hpp"
#include "core/ndptr.hpp"
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
////////////////////////////////////////////////////////////////////////////////
template <int N, typename Conf>
class field_t : public data_t {
 private:
  std::array<typename Conf::multi_array_t, N> m_data;
  std::array<stagger_t, N> m_stagger;
  const Grid<Conf::dim>* m_grid = nullptr;
  MemType m_memtype;

 public:
  field_t(MemType memtype = default_mem_type) : m_memtype(memtype) {}
  field_t(const Grid<Conf::dim>& grid, MemType memtype = default_mem_type);
  field_t(const Grid<Conf::dim>& grid, const std::array<stagger_t, N> st,
          MemType memtype = default_mem_type);
  field_t(const Grid<Conf::dim>& grid, field_type type,
          MemType memtype = default_mem_type);

  field_t(const field_t<N, Conf>& other) = delete;
  field_t(field_t<N, Conf>&& other) = default;

  field_t<N, Conf>& operator=(const field_t<N, Conf>& other) = delete;
  field_t<N, Conf>& operator=(field_t<N, Conf>&& other) = default;

  void init() override;

  void resize(const Grid<Conf::dim>& grid);
  void assign_dev(const typename Conf::value_t& value);
  void assign_host(const typename Conf::value_t& value);
  void assign(const typename Conf::value_t& value);

  template <typename Func>
  void set_values(int n, const Func& f) {
    if (n >= 0 && n < N) {
      Logger::print_debug("data[{}] has extent {}x{}", n, m_data[n].extent()[0],
                          m_data[n].extent()[1]);
      for (auto idx : m_data[n].indices()) {
        auto pos = idx.get_pos();
        double x0 = m_grid->template pos<0>(pos, m_stagger[n]);
        double x1 =
            (Conf::dim > 1 ? m_grid->template pos<1>(pos, m_stagger[n]) : 0.0);
        double x2 =
            (Conf::dim > 2 ? m_grid->template pos<2>(pos, m_stagger[n]) : 0.0);
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

  void copy_from(const field_t<N, Conf>& other) {
    for (int i = 0; i < N; i++) {
      m_data[i].copy_from(other.m_data[i]);
    }
  }
  void add_by(const field_t<N, Conf>& other, typename Conf::value_t scale = 1.0);

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_const_t, M> get_ptrs() const {
    vec_t<typename Conf::ndptr_const_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].get_const_ptr();
    }
    return result;
  }

  // Only provides this method for N > 1
  template <int M = N, greater_than_unity<M> = true>
  vec_t<typename Conf::ndptr_t, M> get_ptrs() {
    vec_t<typename Conf::ndptr_t, M> result;
    for (int i = 0; i < M; i++) {
      result[i] = m_data[i].get_ptr();
    }
    return result;
  }

  void set_memtype(MemType type);

  typename Conf::ndptr_const_t get_ptr(int n = 0) const {
    return m_data[n].get_const_ptr();
  }
  typename Conf::ndptr_t get_ptr(int n = 0) { return m_data[n].get_ptr(); }

  const Grid<Conf::dim>& grid() const { return *m_grid; }
};

template <typename Conf>
using vector_field = field_t<3, Conf>;

template <typename Conf>
using scalar_field = field_t<1, Conf>;

}  // namespace Aperture

#endif  // __FIELDS_H_
