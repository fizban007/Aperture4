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

#ifndef _LORENTZ_TRANSFORM_H_
#define _LORENTZ_TRANSFORM_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

namespace Aperture {

// template <typename Scalar>
// HD_INLINE vec_t<Scalar, 3>
// lorentz_transform_velocity(const vec_t<Scalar, 3>& u_orig,
//                            const vec_t<Scalar, 3>& v) {
//   Scalar gamma = 1.0f / math::sqrt(1.0f - v.dot(v));
//   Scalar udotv = u_orig.dot(v);

//   return (u_orig / gamma + v * (gamma * udotv / (gamma + 1.0f) - 1.0f)) /
//          (1.0f - udotv);
// }

template <typename Scalar>
HD_INLINE vec_t<Scalar, 4>
lorentz_transform_vector(Scalar u0, const vec_t<Scalar, 3>& u_orig,
                         const vec_t<Scalar, 3>& v) {
  Scalar v_sqr = v.dot(v);
  Scalar gamma = 1.0f / math::sqrt(1.0f - v_sqr);
  Scalar udotv = u_orig.dot(v);
  Scalar u0p = gamma * (u0 - udotv);
  vec_t<Scalar, 3> up = u_orig + v * ((gamma - 1.0f) * udotv / v_sqr - gamma * u0);
  return {u0p, up};
}

}  // namespace Aperture

#endif  // _LORENTZ_TRANSFORM_H_
