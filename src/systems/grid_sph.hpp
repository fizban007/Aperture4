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

  // using grid_curv_t<Conf>::grid_curv_t;
  grid_sph_t() : grid_curv_t<Conf>() {
    this->init();
  }
  template <template <class> class ExecPolicy>
  grid_sph_t(const domain_comm<Conf, ExecPolicy>& comm) : grid_curv_t<Conf>(comm) {
    this->init();
  }
  grid_sph_t(const grid_sph_t<Conf>& grid) = default;
  // ~grid_sph_t();

  // static HD_INLINE value_t radius(value_t x1) { return x1; }
  static HD_INLINE value_t radius(value_t x1) { return math::exp(x1); }
  static HD_INLINE value_t theta(value_t x2) { return x2; }
  // static HD_INLINE value_t from_radius(value_t r) { return r; }
  static HD_INLINE value_t from_radius(value_t r) { return math::log(r); }
  static HD_INLINE value_t from_theta(value_t theta) { return theta; }

  // Coordinate for output position
  vec_t<float, 2> cart_coord(
      const index_t<2> &pos) const {
    vec_t<float, 2> result;
    for (int i = 0; i < 2; i++) {
      result[i] = this->coord(i, pos[i], false);
    }
    float r = radius(this->coord(0, pos[0], false));
    float th = theta(this->coord(1, pos[1], false));
    result[0] = r * math::sin(th);
    result[1] = r * math::cos(th);
    return result;
  }

  vec_t<float, 3> cart_coord(
      const index_t<3> &pos) const {
    vec_t<float, 3> result;
    for (int i = 0; i < 3; i++) {
      result[i] = this->coord(i, pos[i], false);
    }
    float r = radius(this->coord(0, pos[0], false));
    float th = theta(this->coord(1, pos[1], false));
    float ph = this->coord(2, pos[2], false);
    result[0] = r * math::cos(ph) * math::sin(th);
    result[1] = r * math::sin(ph) * math::sin(th);
    result[2] = r * math::cos(th);
    return result;
  }

  template <typename FloatT>
  static HD_INLINE vec_t<FloatT, 3> coord_from_cart(const vec_t<FloatT, 3> &x_cart) {
    vec_t<FloatT, 3> result;
    result[0] = math::sqrt(x_cart.dot(x_cart));
    result[2] = math::atan2(x_cart[1], x_cart[0]);
    result[1] = math::acos(x_cart[2] / result[0]);
    result[0] = from_radius(result[0]);
    result[1] = from_theta(result[1]);
    return result;
  }

  template <typename FloatT>
  static HD_INLINE vec_t<FloatT, 3> coord_to_cart(const vec_t<FloatT, 3> &x_sph) {
    vec_t<FloatT, 3> result;
    FloatT r = radius(x_sph[0]);
    FloatT th = theta(x_sph[1]);
    FloatT sth = math::sin(th);
    result[0] = r * sth * math::cos(x_sph[2]);
    result[1] = r * sth * math::sin(x_sph[2]);
    result[2] = r * math::cos(th);
    return result;
  }

  template <typename FloatT>
  static HD_INLINE void vec_from_cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1,
                                      FloatT x2, FloatT x3) {
    FloatT v1n = v1, v2n = v2, v3n = v3;
    FloatT c2 = math::cos(theta(x2)), s2 = math::sin(theta(x2)),
        c3 = math::cos(x3), s3 = math::sin(x3);
    v1 = v1n * s2 * c3 + v2n * s2 * s3 + v3n * c2;
    v2 = v1n * c2 * c3 + v2n * c2 * s3 - v3n * s2;
    v3 = -v1n * s3 + v2n * c3;
  }

  template <typename FloatT>
  static HD_INLINE void vec_from_cart(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
    return vec_from_cart(v[0], v[1], v[2], x[0], x[1], x[2]);
  }

  template <typename FloatT>
  static HD_INLINE void vec_to_cart(FloatT &v1, FloatT &v2, FloatT &v3, FloatT x1,
                                    FloatT x2, FloatT x3) {
    FloatT v1n = v1, v2n = v2, v3n = v3;
    FloatT c2 = math::cos(theta(x2)), s2 = math::sin(theta(x2)),
        c3 = math::cos(x3), s3 = math::sin(x3);
    v1 = v1n * s2 * c3 + v2n * c2 * c3 - v3n * s3;
    v2 = v1n * s2 * s3 + v2n * c2 * s3 + v3n * c3;
    v3 = v1n * c2 - v2n * s2;
  }

  template <typename FloatT>
  static HD_INLINE void vec_to_cart(vec_t<FloatT, 3> &v, const vec_t<FloatT, 3> &x) {
    return vec_to_cart(v[0], v[1], v[2], x[0], x[1], x[2]);
  }
  // inline vec_t<float, Conf::dim> cart_coord(
  //     const index_t<Conf::dim> &pos) const override {
  //   vec_t<float, Conf::dim> result;
  //   for (int i = 0; i < Conf::dim; i++) result[i] = this->coord(i, pos[i], false);
  //   float r = radius(this->coord(0, pos[0], false));
  //   float th = theta(this->coord(1, pos[1], false));
  //   result[0] = r * math::sin(th);
  //   result[1] = r * math::cos(th);
  //   return result;
  // }

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
  // double r_g = 0.0;
  // sim_env().params().get_value("compactness", r_g);

  auto ext = this->extent();

  // Now this should work for both 2D and 3D
  for (auto idx : range(Conf::begin(ext), Conf::end(ext))) {
    auto pos = get_pos(idx, ext);

    double r = radius(this->coord(0, pos[0], false));
    double r_minus = radius(this->coord(0, pos[0] - 1, false));
    double rs = radius(this->coord(0, pos[0], true));
    double rs_plus = radius(this->coord(0, pos[0] + 1, true));

    double th = theta(this->coord(1, pos[1], false));
    double th_minus = theta(this->coord(1, pos[1] - 1, false));
    double ths = theta(this->coord(1, pos[1], true));
    double ths_plus = theta(this->coord(1, pos[1] + 1, true));

    // Note: Nothing depends on phi, so staggering in phi does not matter

    // Length elements for E field
    this->m_le[0][idx] = rs_plus - rs;
    this->m_le[1][idx] = rs * (ths_plus - ths);
    // if constexpr (Conf::dim == 2) {
    if (Conf::dim == 2) {
      this->m_le[2][idx] = rs * std::sin(ths);
    } else if (Conf::dim == 3) {
      this->m_le[2][idx] = rs * std::sin(ths) * this->delta[2];
    }

    // Length elements for B field
    this->m_lb[0][idx] = r - r_minus;
    this->m_lb[1][idx] = r * (th - th_minus);
    // if constexpr (Conf::dim == 2) {
    if (Conf::dim == 2) {
      this->m_lb[2][idx] = r * std::sin(th);
    } else if (Conf::dim == 3) {
      this->m_lb[2][idx] = r * std::sin(th) * this->delta[2];
    }

    // Area elements for E field
    this->m_Ae[0][idx] = r * r * (std::cos(th_minus) - std::cos(th));
    if (math::abs(ths) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * (1.0 - std::cos(theta(0.5 * this->delta[1])));
    } else if (math::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_Ae[0][idx] = r * r * (1.0 - std::cos(theta(0.5 * this->delta[1])));
    }
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ae[0][idx] *= this->delta[2];
    }

    this->m_Ae[1][idx] = 0.5 * (square(r) - square(r_minus)) * std::sin(th);
    // this->m_Ae[1][idx] = (cube(r) - cube(r_minus)) * std::sin(th) / 3.0;
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ae[1][idx] *= this->delta[2];
    }

    this->m_Ae[2][idx] =
        // (cube(r) - cube(r_minus)) * (std::cos(th_minus) - std::cos(th)) / 3.0;
        // 0.5 * (square(r) - square(r_minus)) * (std::cos(th_minus) - std::cos(th));
        // 0.5 * (square(r) - square(r_minus)) * this->delta[1];
        0.5 * (square(r) - square(r_minus)) * (th - th_minus);
    // if (math::abs(ths) < 0.1 * this->delta[1] ||
    //     math::abs(ths - M_PI) < 0.1 * this->delta[1]) {
    //   this->m_Ae[2][idx] =
    //       // (cube(r) - cube(r_minus)) * 2.0 * (1.0 - std::cos(0.5 * this->delta[1])) / 3.0;
    //       0.5 * (square(r) - square(r_minus)) * 2.0 * (1.0 - std::cos(0.5 * this->delta[1]));
    // }

    // Area elements for B field
    this->m_Ab[0][idx] = rs * rs * (std::cos(ths) - std::cos(ths_plus));
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ab[0][idx] *= this->delta[2];
    }

    // this->m_Ab[1][idx] = (cube(rs_plus) - cube(rs)) * std::sin(ths) / 3.0;
    this->m_Ab[1][idx] = 0.5 * (square(rs_plus) - square(rs)) * std::sin(ths);
    if (math::abs(ths) < 0.1 * this->delta[1] ||
        math::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      // this->m_Ab[1][idx] = 0.5 * (square(rs_plus) - square(rs)) * std::sin(ths);
      this->m_Ab[1][idx] = TINY;
    }
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_Ab[1][idx] *= this->delta[2];
    }

    this->m_Ab[2][idx] =
        // (cube(rs_plus) - cube(rs)) * (std::cos(ths) - std::cos(ths_plus)) / 3.0;
        // 0.5 * (square(rs_plus) - square(rs)) * (std::cos(ths) - std::cos(ths_plus));
        0.5 * (square(rs_plus) - square(rs)) * this->delta[1];

    // Volume element, defined at cell vertices
    this->m_dV[idx] = (cube(r) - cube(r_minus)) *
                      (std::cos(th_minus) - std::cos(th)) /
                      (this->delta[0] * this->delta[1] * 3.0);

    if (math::abs(ths) < 0.1 * this->delta[1] ||
        math::abs(ths - M_PI) < 0.1 * this->delta[1]) {
      this->m_dV[idx] = (cube(r) - cube(r_minus)) * 2.0 *
                        (1.0 - std::cos(0.5 * this->delta[1])) /
                        (this->delta[0] * this->delta[1] * 3.0);
    }
    // if constexpr (Conf::dim == 3) {
    if (Conf::dim == 3) {
      this->m_dV[idx] /= this->delta[2];
    }
  }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
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
