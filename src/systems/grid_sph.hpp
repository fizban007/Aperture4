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

#ifndef _GRID_SPH_H_
#define _GRID_SPH_H_

#include "core/math.hpp"
#include "framework/environment.h"
#include "grid_curv.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is the general spherical grid class. The class implements two crucial
///  functions: radius and theta, and provides a way to use these to compute
///  area and length elements.
////////////////////////////////////////////////////////////////////////////////
template <typename Conf>
class grid_sph_t : public grid_curv_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;

  using grid_curv_t<Conf>::grid_curv_t;
  // ~grid_sph_t();

  // static HD_INLINE value_t radius(value_t x1) { return x1; }
  static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  // static HD_INLINE value_t from_radius(value_t r) { return r; }
  static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }

  // Coordinate for output position
  inline vec_t<float, Conf::dim> cart_coord(
      const index_t<Conf::dim> &pos) const override {
    vec_t<float, Conf::dim> result;
    for (int i = 0; i < Conf::dim; i++) result[i] = this->pos(i, pos[i], false);
    float r = radius(this->pos(0, pos[0], false));
    float th = theta(this->pos(1, pos[1], false));
    result[0] = r * math::sin(th);
    result[1] = r * math::cos(th);
    return result;
  }

  void compute_coef() override;
};

template <typename FloatT>
HD_INLINE void
cart2sph(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2), c3 = math::cos(x3),
         s3 = math::sin(x3);
  v1 = v1n * s2 * c3 + v2n * s2 * s3 + v3n * c2;
  v2 = v1n * c2 * c3 + v2n * c2 * s3 - v3n * s2;
  v3 = -v1n * s3 + v2n * c3;
}

template <typename FloatT>
HD_INLINE void
cart2sph(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
  return cart2sph(v[0], v[1], v[2], x[0], x[1], x[2]);
}

template <typename FloatT>
HD_INLINE void
sph2cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1, FloatT x2, FloatT x3) {
  FloatT v1n = v1, v2n = v2, v3n = v3;
  FloatT c2 = math::cos(x2), s2 = math::sin(x2), c3 = math::cos(x3),
         s3 = math::sin(x3);
  v1 = v1n * s2 * c3 + v2n * c2 * c3 - v3n * s3;
  v2 = v1n * s2 * s3 + v2n * c2 * s3 + v3n * c3;
  v3 = v1n * c2 - v2n * s2;
}

template <typename FloatT>
HD_INLINE void
sph2cart(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
  return sph2cart(v[0], v[1], v[2], x[0], x[1], x[2]);
}

template <typename FloatT>
HD_INLINE FloatT
beta_phi(FloatT r, FloatT theta, FloatT compactness, FloatT omega) {
  return -0.4f * compactness * omega * math::sin(theta) / (r * r);
}

template <typename FloatT>
HD_INLINE FloatT
alpha_gr(FloatT r, FloatT compactness) {
  return math::sqrt(1.0f - compactness / r);
}

template <typename Conf>
void
grid_sph_t<Conf>::compute_coef() {
  double r_g = 0.0;
  sim_env().params().get_value("compactness", r_g);

  auto ext = this->extent();

  // Now this should work for both 2D and 3D
  for (auto idx : range(Conf::begin(ext), Conf::end(ext))) {
    auto pos = idx.get_pos();

    double r = radius(this->template pos<0>(pos[0], false));
    double r_minus = radius(this->template pos<0>(pos[0] - 1, false));
    double rs = radius(this->template pos<0>(pos[0], true));
    double rs_plus = radius(this->template pos<0>(pos[0] + 1, true));

    double th = theta(this->template pos<1>(pos[1], false));
    double th_minus = theta(this->template pos<1>(pos[1] - 1, false));
    double ths = theta(this->template pos<1>(pos[1], true));
    double ths_plus = theta(this->template pos<1>(pos[1] + 1, true));

    // Length elements for E field
    this->m_le[0][idx] = rs_plus - rs;
    this->m_le[1][idx] = rs * this->delta[1];
    // if constexpr (Conf::dim == 2) {
    if (Conf::dim == 2) {
      this->m_le[2][idx] = rs * std::sin(ths);
    } else if (Conf::dim == 3) {
      this->m_le[2][idx] = rs * std::sin(ths) * this->delta[2];
    }

    // Length elements for B field
    this->m_lb[0][idx] = r - r_minus;
    this->m_lb[1][idx] = r * this->delta[1];
    // if constexpr (Conf::dim == 2) {
    if (Conf::dim == 2) {
      this->m_lb[2][idx] = r * std::sin(th);
    } else if (Conf::dim == 3) {
      this->m_lb[2][idx] = r * std::sin(th) * this->delta[2];
    }

    // Area elements for E field
    this->m_Ae[0][idx] = r * r * (std::cos(th_minus) - std::cos(th));
    if (std::abs(ths) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
    } else if (std::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
    }
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ae[0][idx] *= this->delta[2];
    }

    this->m_Ae[1][idx] = 0.5 * (square(r) - square(r_minus)) * std::sin(th);
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ae[1][idx] *= this->delta[2];
    }

    this->m_Ae[2][idx] =
        (cube(r) - cube(r_minus)) / 3.0 * (std::cos(th_minus) - std::cos(th));

    // Area elements for B field
    this->m_Ab[0][idx] = rs * rs * (std::cos(ths) - std::cos(ths_plus));
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ab[0][idx] *= this->delta[2];
    }

    if (std::abs(ths) > 0.1 * this->delta[1] &&
        std::abs(ths - M_PI) > 0.1 * this->delta[1])
      this->m_Ab[1][idx] = 0.5 * (square(rs_plus) - square(rs)) * std::sin(ths);
    else
      this->m_Ab[1][idx] = TINY;
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ab[1][idx] *= this->delta[2];
    }

    this->m_Ab[2][idx] =
        (cube(rs_plus) - cube(rs)) / 3.0 * (std::cos(ths) - std::cos(ths_plus));

    // Volume element, defined at cell vertices
    this->m_dV[idx] = (cube(r) - cube(r_minus)) / 3.0 *
                      (std::cos(th_minus) - std::cos(th)) /
                      (this->delta[0] * this->delta[1]);

    if (std::abs(ths) < 0.1 * this->delta[1] ||
        std::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_dV[idx] = (cube(r) - cube(r_minus)) * 2.0 / 3.0 *
                        (1.0 - std::cos(0.5 * this->delta[1])) /
                        (this->delta[0] * this->delta[1]);
    }
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_dV[idx] /= this->delta[2];
    }
  }

#ifdef CUDA_ENABLED
  for (int i = 0; i < 3; i++) {
    this->m_le[i].copy_to_device();
    this->m_lb[i].copy_to_device();
    this->m_Ae[i].copy_to_device();
    this->m_Ab[i].copy_to_device();
  }
  this->m_dV.copy_to_device();
#endif
}

}  // namespace Aperture

#endif  // _GRID_SPH_H_
