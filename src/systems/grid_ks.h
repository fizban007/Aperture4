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

namespace Metric_KS {

HD_INLINE Scalar
rho2(Scalar a, Scalar r, Scalar th) {
  Scalar cth = math::cos(th);
  return r * r + a * a * cth * cth;
}

HD_INLINE Scalar
rho2(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return r * r + a * a * cth * cth;
}

HD_INLINE Scalar
Z(Scalar a, Scalar r, Scalar th) {
  return 2.0f * r / rho2(a, r, th);
}

HD_INLINE Scalar
Z(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 2.0f * r / rho2(a, r, sth, cth);
}

// HD_INLINE Scalar
// Delta(Scalar a, Scalar r) {
//   return r * r + a * a - 2.0f * r;
// }

HD_INLINE Scalar
Sigma(Scalar a, Scalar r, Scalar th) {
  Scalar r2a2 = r * r + a * a;
  Scalar sth = math::sin(th);
  return square(r2a2) - square(a * sth) * (r2a2 - 2.0f * r);
}

HD_INLINE Scalar
Sigma(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar r2a2 = r * r + a * a;
  return square(r2a2) - square(a * sth) * (r2a2 - 2.0f * r);
}

HD_INLINE Scalar
alpha(Scalar a, Scalar r, Scalar th) {
  return 1.0f / math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
alpha(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f / math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
beta1(Scalar a, Scalar r, Scalar th) {
  // Scalar z = Z(a, r, th);
  // return z / (1.0f + z);
  // return 2.0f / (r * (2.0f + r) + square(a * math::cos(th)));
  return 2.0f * r / (r * (2.0f + r) + square(a * math::cos(th)));
}

HD_INLINE Scalar
beta1(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  // Scalar z = Z(a, r, sth, cth);
  // return z / (1.0f + z);
  // return 2.0f / (r * (2.0f + r) + square(a * cth));
  return 2.0f * r / (r * (2.0f + r) + square(a * cth));
}

// HD_INLINE Scalar
// g_11(Scalar a, Scalar r, Scalar th) {
//   return 1.0f + Z(a, r, th);
// }

// HD_INLINE Scalar
// g_11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
//   return 1.0f + Z(a, r, sth, cth);
// }

// HD_INLINE Scalar
// g_22(Scalar a, Scalar r, Scalar th) {
//   return rho2(a, r, th);
// }

// HD_INLINE Scalar
// g_22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
//   return rho2(a, r, sth, cth);
// }

// HD_INLINE Scalar
// g_33(Scalar a, Scalar r, Scalar th) {
//   Scalar sth = math::sin(th);
//   return Sigma(a, r, th) * sth * sth / rho2(a, r, th);
// }

// HD_INLINE Scalar
// g_33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
//   return Sigma(a, r, sth, cth) * sth * sth / rho2(a, r, sth, cth);
// }

// HD_INLINE Scalar
// g_13(Scalar a, Scalar r, Scalar th) {
//   Scalar sth = math::sin(th);
//   return -a * sth * sth * (1.0f + Z(a, r, th));
// }

// HD_INLINE Scalar
// g_13(Scalar a, Scalar r, Scalar sth, Scalar cth) {
//   return -a * sth * sth * (1.0f + Z(a, r, sth, cth));
// }

HD_INLINE Scalar
ag_11(Scalar a, Scalar r, Scalar th) {
  return math::sqrt(1.0f + Z(a, r, th));
  // return r * r * math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return math::sqrt(1.0f + Z(a, r, sth, cth));
  // return r * r * math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
ag_22(Scalar a, Scalar r, Scalar th) {
  return rho2(a, r, th) / math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return rho2(a, r, sth, cth) / math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
ag_33(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return Sigma(a, r, th) * sth * sth /
         (rho2(a, r, th) * math::sqrt(1.0f + Z(a, r, th)));
}

HD_INLINE Scalar
ag_33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return Sigma(a, r, sth, cth) * sth * sth /
         (rho2(a, r, sth, cth) * math::sqrt(1.0f + Z(a, r, sth, cth)));
}

HD_INLINE Scalar
ag_13(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return -a * sth * sth * math::sqrt(1.0f + Z(a, r, th));
  // return -a * r * sth * sth * math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_13(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return -a * sth * sth * math::sqrt(1.0f + Z(a, r, sth, cth));
  // return -a * r * sth * sth * math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
sqrt_gamma(Scalar a, Scalar r, Scalar th) {
  // Scalar a2c2th = a * a * (1.0f + math::cos(2.0f * th));
  // return 0.5f * math::abs(math::sin(th)) *
  //        math::sqrt((a2c2th + 2.0f * r * r) * (a2c2th + 2.0f * r * (2.0f +
  //        r)));
  return rho2(a, r, th) * math::sin(th) * math::sqrt(1.0f + Z(a, r, th));
  // return r * rho2(a, r, th) * math::sin(th) * math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
sqrt_gamma(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  // Scalar a2c2th = a * a * (1.0f + cth * cth - sth * sth);
  // return 0.5f * math::abs(sth) *
  //        math::sqrt((a2c2th + 2.0f * r * r) * (a2c2th + 2.0f * r * (2.0f +
  //        r)));
  return rho2(a, r, sth, cth) * sth * math::sqrt(1.0f + Z(a, r, sth, cth));
  // return r * rho2(a, r, sth, cth) * sth * math::sqrt(1.0f + Z(a, r, sth,
  // cth));
}

// This returns the composite value of sqrt(gamma) * beta1
HD_INLINE Scalar
sq_gamma_beta(Scalar a, Scalar r, Scalar th) {
  // Scalar a2c2th = a * a * (1.0f + math::cos(2.0f * th));
  // return r * math::abs(math::sin(th)) *
  //        math::sqrt((a2c2th + 2.0f * r * r) *
  //                   (a2c2th + 2.0f * r * (2.0f + r))) /
  //        (r * (2.0f + r) + square(a * math::cos(th)));
  Scalar z = Z(a, r, th);
  return 2.0f * r * math::sin(th) / math::sqrt(1.0f + z);
}

HD_INLINE Scalar
sq_gamma_beta(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  // Scalar a2c2th = a * a * (1.0f + cth * cth - sth * sth);
  // return r * math::abs(sth) *
  //        math::sqrt((a2c2th + 2.0f * r * r) *
  //                   (a2c2th + 2.0f * r * (2.0f + r))) /
  //        (r * (2.0f + r) + square(a * cth));
  Scalar z = Z(a, r, sth, cth);
  return 2.0f * r * sth / math::sqrt(1.0f + z);
}

}  // namespace Metric_KS

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
    typename Conf::ndptr_const_t ag22dth_h;
    typename Conf::ndptr_const_t ag22dth_e;
    typename Conf::ndptr_const_t gbetadth_h;
    typename Conf::ndptr_const_t gbetadth_e;
    vec_t<typename Conf::ndptr_const_t, 3> Ad;
    vec_t<typename Conf::ndptr_const_t, 3> Ab;
  };

  value_t a = 0.99;
  grid_ptrs ptrs;

  grid_ks_t(sim_environment& env, const domain_comm<Conf>* comm);
  grid_ks_t(const grid_ks_t<Conf>& grid) = default;
  virtual ~grid_ks_t() {}

  grid_ks_t<Conf>& operator=(const grid_ks_t<Conf>& grid) = default;

  static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  // static HD_INLINE value_t radius(value_t x1) { return x1; }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  // static HD_INLINE value_t from_radius(value_t r) { return r; }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }

  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim>& pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = radius(result[0]);
    float th = theta(result[1]);
    result[0] = r * math::sin(th);
    result[1] = r * math::cos(th);
    return result;
  }

  void compute_coef();

  grid_ptrs get_grid_ptrs() const {
    return ptrs;
  }

  std::array<multi_array<value_t, Conf::dim>, 3> m_Ad;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ab;
  multi_array<value_t, Conf::dim> m_ag11dr_h;
  multi_array<value_t, Conf::dim> m_ag11dr_e;
  multi_array<value_t, Conf::dim> m_ag13dr_h;
  multi_array<value_t, Conf::dim> m_ag13dr_e;
  multi_array<value_t, Conf::dim> m_ag22dth_h;
  multi_array<value_t, Conf::dim> m_ag22dth_e;
  multi_array<value_t, Conf::dim> m_gbetadth_h;
  multi_array<value_t, Conf::dim> m_gbetadth_e;
};

}  // namespace Aperture

#endif  // _GRID_KS_H_
