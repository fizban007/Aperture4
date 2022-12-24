/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef GRID_POLAR_H_
#define GRID_POLAR_H_

#include "core/math.hpp"
#include "framework/environment.h"
#include "grid_curv.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is 2D polar coordinate grid class. For cylindrical 2D/3D, or spherical
///  2D/3D, consider using `grid_sph` or `grid_cyl`. This is only defined for 2D!
////////////////////////////////////////////////////////////////////////////////
template <typename Conf>
class grid_polar_t : public grid_curv_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  using grid_curv_t<Conf>::grid_curv_t;
  // ~grid_sph_t();

  static HD_INLINE value_t radius(value_t x1) { return x1; }
  // static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  static HD_INLINE value_t from_radius(value_t r) { return r; }
  // static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }

  // Coordinate for output position
  inline vec_t<float, 2> cart_coord(
      const index_t<2> &pos) const {
    vec_t<float, 2> result;
    for (int i = 0; i < 2; i++) {
      result[i] = this->coord(i, pos[i], false);
    }
    float r = radius(this->coord(0, pos[0], false));
    float th = theta(this->coord(1, pos[1], false));
    result[0] = r * math::cos(th);
    result[1] = r * math::sin(th);
    return result;
  }

  void compute_coef() override;
};

// Converting 3-vectors from Cartesian
template <typename FloatT>
HD_INLINE void
cart2polar(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2);
  v1 = v1n * c2 + v2n * s2;
  v2 = -v1n * s2 + v2n * c2;
  v3 = v3n;
}

template <typename FloatT>
HD_INLINE void
cart2polar(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
  return cart2polar(v[0], v[1], v[2], x[0], x[1], x[2]);
}

// Converting 3-vectors to Cartesian
template <typename FloatT>
HD_INLINE void
polar2cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2);
  v1 = v1n * c2 - v2n * s2;
  v2 = v1n * s2 + v2n * c2;
  v3 = v3n;
}

template <typename FloatT>
HD_INLINE void
polar2cart(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
  return polar2cart(v[0], v[1], v[2], x[0], x[1], x[2]);
}

template <typename Conf>
void
grid_polar_t<Conf>::compute_coef() {
  auto ext = this->extent();

  // Now this should work for both 2D and 3D
  for (auto idx : range(Conf::begin(ext), Conf::end(ext))) {
    auto pos = get_pos(idx, ext);

    double r = radius(this->template coord<0>(pos[0], false));
    double r_minus = radius(this->template coord<0>(pos[0] - 1, false));
    double rs = radius(this->template coord<0>(pos[0], true));
    double rs_plus = radius(this->template coord<0>(pos[0] + 1, true));

    double th = theta(this->template coord<1>(pos[1], false));
    double th_minus = theta(this->template coord<1>(pos[1] - 1, false));
    double ths = theta(this->template coord<1>(pos[1], true));
    double ths_plus = theta(this->template coord<1>(pos[1] + 1, true));

    // Note: Nothing depends on z, so staggering in z does not matter

    // Length elements for E field
    this->m_le[0][idx] = rs_plus - rs;
    this->m_le[1][idx] = rs * (ths_plus - ths);
    this->m_le[2][idx] = 1.0;

    // Length elements for B field
    this->m_lb[0][idx] = r - r_minus;
    this->m_lb[1][idx] = r * (th - th_minus);
    this->m_lb[2][idx] = 1.0;

    // Area elements for E field
    this->m_Ae[0][idx] = r * (th - th_minus);
    this->m_Ae[1][idx] = r - r_minus;
    this->m_Ae[2][idx] =
        0.5 * (square(r) - square(r_minus)) * (th - th_minus);

    // Area elements for B field
    this->m_Ab[0][idx] = rs * (ths_plus - ths);
    this->m_Ab[1][idx] = rs_plus - rs;
    this->m_Ab[2][idx] =
        0.5 * (square(rs_plus) - square(rs)) * (ths_plus - ths);

    // Volume element, defined at cell vertices
    this->m_dV[idx] = (th - th_minus) * (square(r) - square(r_minus)) * 0.5 /
                      (this->delta[0] * this->delta[1]);
  }

#ifdef GPU_ENABLED
  for (int i = 0; i < 3; i++) {
    this->m_le[i].copy_to_device();
    this->m_lb[i].copy_to_device();
    this->m_Ae[i].copy_to_device();
    this->m_Ab[i].copy_to_device();
  }
  this->m_dV.copy_to_device();
#endif
}

}

#endif // GRID_POLAR_H_
