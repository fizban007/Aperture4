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

#pragma once

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

namespace Aperture {

namespace Metric_BL {

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

HD_INLINE Scalar
Delta(Scalar a, Scalar r) {
  return r * r + a * a - 2.0f * r;
}

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
  return math::sqrt(rho2(a, r, th) * Delta(a, r) / Sigma(a, r, th));
}

HD_INLINE Scalar
beta3(Scalar a, Scalar r, Scalar th) {
  return -2.0f * a * r / Sigma(a, r, th);
}

HD_INLINE Scalar
gu00(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar a2r2 = a * a + r * r;
  return -1.0f - 2.0f * r * a2r2 / (a2r2 - 2.0f * r) / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
gu11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return (a * a + r * r - 2.0f * r) / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
gu22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
gu33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar rho2v = rho2(a, r, sth, cth);
  // return (rho2v - 2.0f * r) / (a * a + r * r - 2.0f * r) / rho2v / square(sth);
  return rho2v / Sigma(a, r, sth, cth) / square(sth);
}

HD_INLINE Scalar
gu03(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return -2.0f * a * r / (a * a + r * r - 2.0 * r) / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
dot_product_l(const vec_t<Scalar, 3>& v1,
              const vec_t<Scalar, 3>& v2,
              Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  Scalar cth = math::cos(th);
  return gu11(a, r, sth, cth) * v1[0] * v2[0] +
         gu22(a, r, sth, cth) * v1[1] * v2[1] +
         gu33(a, r, sth, cth) * v1[2] * v2[2];
}

HD_INLINE Scalar
u0(Scalar a, Scalar r, Scalar th, const vec_t<Scalar, 3> &u,
   bool is_photon = false) {
  return math::sqrt(dot_product_l(u, u, a, r, th) + (is_photon ? 0.0f : 1.0f)) /
         alpha(a, r, th);
}

HD_INLINE Scalar
u_0(Scalar a, Scalar r, Scalar th, const vec_t<Scalar, 3> &u,
    bool is_photon = false) {
  return u[2] * beta3(a, r, th) - math::sqrt(dot_product_l(u, u, a, r, th) + (is_photon ? 0.0f : 1.0f)) *
         alpha(a, r, th);
}

}  // namespace Metric_BL

}  // namespace Aperture
