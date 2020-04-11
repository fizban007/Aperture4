#ifndef __FIELDS_H_
#define __FIELDS_H_

#include "core/grid.hpp"
#include "framework/data.h"
#include "utils/stagger.h"
#include <array>

namespace Aperture {

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

  void init(const std::string& name, const sim_environment& env);
  void init(const extent_t<Conf::dim>& ext);

  template <typename Func>
  void set_values(int n, const Func& f, const Grid<Conf::dim>& grid) {
    if (n >= 0 && n < Conf::dim) {
      for (auto idx : m_data[n].indices()) {
        auto pos = idx.get_pos();
        double x0 = grid.template pos<0>(pos[0], m_stagger[n][0]);
        double x1 = grid.template pos<1>(pos[1], m_stagger[n][1]);
        double x2 = grid.template pos<2>(pos[2], m_stagger[n][2]);
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
};

template <typename Conf>
using vector_field = field_t<3, Conf>;

template <typename Conf>
using scalar_field = field_t<1, Conf>;

}  // namespace Aperture

#endif  // __FIELDS_H_
