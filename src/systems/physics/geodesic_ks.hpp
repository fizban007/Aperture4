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

#ifndef __GEODESIC_KS_H_
#define __GEODESIC_KS_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "systems/grid_ks.h"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

namespace Aperture {

// This is declared as a template function so that we can write it in a header
// file rather than an individual cpp/cuda file
template <typename FloatT>
HOST_DEVICE vec_t<FloatT, 3>
geodesic_ks_u_rhs(FloatT a, const vec_t<FloatT, 3>& x,
                  const vec_t<FloatT, 3>& u, bool is_photon = false) {
  vec_t<FloatT, 3> result;
  result[2] = 0.0f;  // u_phi is conserved and does not need update

  FloatT r = x[0];
  FloatT th = x[1];
  // regularize theta from the axis
  // if (th < 1.0e-5f) th = 1.0e-5f;
  // if (th > M_PI - 1.0e-5f) th = M_PI - 1.0e-5f;

  FloatT sth = math::sin(th);
  FloatT cth = math::cos(th);
  FloatT alpha = Metric_KS::alpha(a, r, sth, cth);
  FloatT u0 = Metric_KS::u0(a, r, sth, cth, u, is_photon);
  FloatT rho2 = Metric_KS::rho2(a, r, sth, cth);

  // first term -\alpha u^0 \partial_i \alpha
  FloatT factor = math::sqrt(cube(1.0f + 2.0f * r / rho2)) * square(rho2);
  result[0] = -alpha * u0 * (r * r - square(a * cth)) / factor;
  result[1] = alpha * u0 * 2.0f * a * a * r * sth * cth / factor;

  // second term u_r \partial_i \beta^r
  factor = square(2.0f * r + rho2);
  result[0] += u[0] * 2.0f * (square(a * cth) - r * r) / factor;
  result[1] += u[0] * 4.0f * a * a * r * sth * cth / factor;

  // third term -u_ju_k \partial_i \gamma^jk
  FloatT rho2p2r = rho2 + 2.0f * r;
  FloatT a2r2 = a * a + r * r;
  result[0] -=
      (u[0] * u[0] *
           ((r - r * a2r2 / rho2) / rho2 +
            (2.0f * r * (1.0f + r) / rho2p2r - 1.0f) / rho2p2r) -
       r / square(rho2) *
           (u[1] * u[1] + 2.0f * a * u[0] * u[2] + u[2] * u[2] / square(sth))) /
      u0;
  result[1] -= (u[0] * u[0] * a * a * sth * cth *
                    (a2r2 / square(rho2) - 2.0f * r / square(rho2p2r)) +
                a * a * sth * cth / square(rho2) *
                    (u[1] * u[1] + 2.0f * a * u[0] * u[2]) -
                (rho2 - square(a * sth)) * cth / (square(rho2) * cube(sth)) *
                    u[2] * u[2]) /
               u0;

  return result;
}

template <typename FloatT>
HOST_DEVICE vec_t<FloatT, 3>
geodesic_ks_x_rhs(FloatT a, const vec_t<FloatT, 3>& x,
                  const vec_t<FloatT, 3>& u, bool is_photon = false) {
  vec_t<FloatT, 3> result;

  FloatT r = x[0];
  FloatT th = x[1];
  // regularize theta from the axis
  // if (th < 1.0e-5f) th = 1.0e-5f;
  // if (th > M_PI - 1.0e-5f) th = M_PI - 1.0e-5f;

  FloatT sth = math::sin(th);
  FloatT cth = math::cos(th);
  FloatT u0 = Metric_KS::u0(a, r, sth, cth, u, is_photon);
  FloatT rho2 = Metric_KS::rho2(a, r, sth, cth);

  result[0] = (Metric_KS::gu11(a, r, sth, cth) * u[0] + a * u[2] / rho2) / u0 -
              Metric_KS::beta1(a, r, sth, cth);
  result[1] = u[1] / rho2 / u0;
  result[2] = (u[2] / square(sth) + a * u[0]) / rho2 / u0;

  return result;
}

}  // namespace Aperture

#endif  // __GEODESIC_KS_H_
