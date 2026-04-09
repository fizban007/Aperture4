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

#ifndef _GRID_KS_H_
#define _GRID_KS_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/multi_array.hpp"
#include "grid.h"
#include <array>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is the general grid class for GR Kerr-Schild coordinates. The class
///  implements a range of mathematical functions needed to compute the KS
///  coefficients, and provides a way to use these to compute area and length
///  elements.
////////////////////////////////////////////////////////////////////////////////
template <typename Conf>
class grid_ks_t : public grid_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  struct grid_ptrs {
    typename Conf::ndptr_const_t ag11dr_h;
    typename Conf::ndptr_const_t ag11dr_e;
    typename Conf::ndptr_const_t ag13dr_h;
    typename Conf::ndptr_const_t ag13dr_e;
    typename Conf::ndptr_const_t ag13dr_b;
    typename Conf::ndptr_const_t ag13dr_d;
    typename Conf::ndptr_const_t ag22dth_h;
    typename Conf::ndptr_const_t ag22dth_e;
    typename Conf::ndptr_const_t gbetadth_h;
    typename Conf::ndptr_const_t gbetadth_e;
    typename Conf::ndptr_const_t gbetadth_b;
    typename Conf::ndptr_const_t gbetadth_d;
    vec_t<typename Conf::ndptr_const_t, 3> Ad;
    vec_t<typename Conf::ndptr_const_t, 3> Ab;
  };

  value_t a = 0.99;
  grid_ptrs ptrs;

  grid_ks_t();
  template <template <class> class ExecPolicy>
  grid_ks_t(const domain_comm<Conf, ExecPolicy>& comm) : grid_t<Conf>(comm) {
    initialize();
  }
  grid_ks_t(const grid_ks_t<Conf>& grid) = default;
  grid_ks_t(grid_ks_t<Conf>&& grid) = default;
  virtual ~grid_ks_t() {}

  grid_ks_t<Conf>& operator=(const grid_ks_t<Conf>& grid) = default;
  grid_ks_t<Conf>& operator=(grid_ks_t<Conf>&& grid) = default;

  void initialize();

  static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  // static HD_INLINE value_t radius(value_t x1) { return x1; }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  static HD_INLINE value_t phi(value_t x3) { return x3; }
  static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  // static HD_INLINE value_t from_radius(value_t r) { return r; }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }
  static HD_INLINE value_t from_phi(value_t phi) { return phi; }

  inline vec_t<float, 2> cart_coord(const index_t<2>& pos) const {
    vec_t<float, 2> result;
    for (int i = 0; i < 2; i++) result[i] = this->coord(i, pos[i], false);
    float r = radius(result[0]);
    float th = theta(result[1]);
    result[0] = r * math::sin(th);
    result[1] = r * math::cos(th);
    return result;
  }

  inline vec_t<float, 3> cart_coord(const index_t<3>& pos) const {
    vec_t<float, 3> result;
    for (int i = 0; i < 3; i++) result[i] = this->coord(i, pos[i], false);
    float r = radius(result[0]);
    float th = theta(result[1]);
    float ph = phi(result[2]);
    result[0] = r * math::cos(ph) * math::sin(th);
    result[1] = r * math::sin(ph) * math::sin(th);
    result[2] = r * math::cos(th);
    return result;
  }

  void compute_coef();

  grid_ptrs get_grid_ptrs() const { return ptrs; }

  std::array<multi_array<value_t, Conf::dim>, 3> m_Ad;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ab;
  multi_array<value_t, Conf::dim> m_ag11dr_h;
  multi_array<value_t, Conf::dim> m_ag11dr_e;
  multi_array<value_t, Conf::dim> m_ag13dr_h;
  multi_array<value_t, Conf::dim> m_ag13dr_e;
  multi_array<value_t, Conf::dim> m_ag13dr_b;
  multi_array<value_t, Conf::dim> m_ag13dr_d;
  multi_array<value_t, Conf::dim> m_ag22dth_h;
  multi_array<value_t, Conf::dim> m_ag22dth_e;
  multi_array<value_t, Conf::dim> m_gbetadth_h;
  multi_array<value_t, Conf::dim> m_gbetadth_e;
  multi_array<value_t, Conf::dim> m_gbetadth_b;
  multi_array<value_t, Conf::dim> m_gbetadth_d;
};

}  // namespace Aperture

#endif  // _GRID_KS_H_
