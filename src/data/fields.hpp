#ifndef __FIELDS_H_
#define __FIELDS_H_

#include "core/grid.hpp"
#include "core/ndptr.hpp"
#include "framework/data.h"
#include "utils/stagger.h"
#include <array>

namespace Aperture {

enum field_type : char {
  face_centered,
  edge_centered,
  cell_centered,
  vert_centered
};

template <int N, typename Conf>
class field_t : public data_t {
 private:
  std::array<typename Conf::multi_array_t, N> m_data;
  std::array<stagger_t, N> m_stagger;
  const Conf& m_conf;

 public:
  field_t(const Conf& conf) : m_conf(conf) {}
  field_t(const Conf& conf, const std::array<stagger_t, N> st)
      : m_conf(conf), m_stagger(st) {}
  field_t(const Conf& conf, field_type type) : m_conf(conf) {
    if (type == field_type::face_centered) {
      m_stagger[0] = stagger_t(0b001);
      m_stagger[1] = stagger_t(0b010);
      m_stagger[2] = stagger_t(0b100);
    } else if (type == field_type::edge_centered) {
      m_stagger[0] = stagger_t(0b110);
      m_stagger[1] = stagger_t(0b101);
      m_stagger[2] = stagger_t(0b011);
    } else if (type == field_type::cell_centered) {
      m_stagger[0] = m_stagger[1] = m_stagger[2] = stagger_t(0b000);
    } else if (type == field_type::vert_centered) {
      m_stagger[0] = m_stagger[1] = m_stagger[2] = stagger_t(0b111);
    }
  }

  void init(const std::string& name, const sim_environment& env);
  void init(const extent_t<Conf::dim>& ext);

  template <typename Func>
  void set_values(int n, const Func& f, const Grid<Conf::dim>& grid) {
    if (n >= 0 && n < Conf::dim) {
      for (auto idx : m_data[n].indices()) {
        auto pos = idx.get_pos();
        double x0 = grid.template pos<0>(pos, m_stagger[n]);
        double x1 = grid.template pos<1>(pos, m_stagger[n]);
        double x2 = grid.template pos<2>(pos, m_stagger[n]);
        m_data[n][idx] = f(x0, x1, x2);
      }
#ifdef CUDA_ENABLED
      m_data[n].copy_to_device();
#endif
    }
  }

  template <typename Func>
  void set_values(const Func& f, const Grid<Conf::dim>& grid) {
    for (int n = 0; n < Conf::dim; n++) {
      set_values(n, [&f, n](auto x0, auto x1, auto x2) {
        return f(n, x0, x1, x2);
      });
    }
  }

  typename Conf::multi_array_t& operator[](int n) { return m_data[n]; }
  const typename Conf::multi_array_t& operator[](int n) const {
    return m_data[n];
  }
  stagger_t stagger(int n) const { return m_stagger[n]; }

  void copy_from(const field_t<N, Conf>& other) {
    for (int i = 0; i < N; i++) {
      m_data[i].copy_from(other.m_data[i]);
    }
  }

  vec_t<typename Conf::ndptr_const_t, N> get_ptrs() const;
  vec_t<typename Conf::ndptr_t, N> get_ptrs();
};

template <typename Conf>
using vector_field = field_t<3, Conf>;

template <typename Conf>
using scalar_field = field_t<1, Conf>;

}  // namespace Aperture

#endif  // __FIELDS_H_
