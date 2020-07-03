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

#ifndef __GRAVITY_H_
#define __GRAVITY_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Scalar>
HD_INLINE void
gravity(Scalar& p1, Scalar p2, Scalar p3, Scalar& gamma, Scalar r,
        Scalar dt, Scalar g0, Scalar q_over_m) {
  // Add an artificial gravity
  p1 -= dt * g0 / (r * r * math::abs(q_over_m));
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  // if (gamma != gamma) {
  //   printf(
  //       "NaN detected after gravity! p1 is %f, p2 is %f, p3 is "
  //       "%f, gamma is "
  //       "%f\n",
  //       p1, p2, p3, gamma);
  //   asm("trap;");
  // }
}


}

#endif // __GRAVITY_H_
