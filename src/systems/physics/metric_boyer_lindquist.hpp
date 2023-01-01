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
  return (rho2v - 2.0f * r) / (a * a + r * r - 2.0f * r) / rho2v / square(sth);
}

HD_INLINE Scalar
gu03(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return -2.0f * a * r / (a * a + r * r - 2.0 * r) / rho2(a, r, sth, cth);
}

}  // namespace Metric_BL

}  // namespace Aperture
