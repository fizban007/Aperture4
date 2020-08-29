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

#ifndef __SYNC_COOLING_H_
#define __SYNC_COOLING_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Scalar>
HD_INLINE void
sync_cooling(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
             Scalar B1, Scalar B2, Scalar B3, Scalar E1, Scalar E2,
             Scalar E3, Scalar q_over_m, Scalar cooling_coef, Scalar B0) {
  Scalar tmp1 = (E1 + (p2 * B3 - p3 * B2) / gamma) / q_over_m;
  Scalar tmp2 = (E2 + (p3 * B1 - p1 * B3) / gamma) / q_over_m;
  Scalar tmp3 = (E3 + (p1 * B2 - p2 * B1) / gamma) / q_over_m;
  Scalar tmp_sq = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
  Scalar bE = (p1 * E1 + p2 * E2 + p3 * E3) / (gamma * q_over_m);

  Scalar delta_p1 = cooling_coef *
                    (((tmp2 * B3 - tmp3 * B2) + bE * E1) / q_over_m -
                     gamma * p1 * (tmp_sq - bE * bE)) /
                    square(B0);
  Scalar delta_p2 = cooling_coef *
                    (((tmp3 * B1 - tmp1 * B3) + bE * E2) / q_over_m -
                     gamma * p2 * (tmp_sq - bE * bE)) /
                    square(B0);
  Scalar delta_p3 = cooling_coef *
                    (((tmp1 * B2 - tmp2 * B1) + bE * E3) / q_over_m -
                     gamma * p3 * (tmp_sq - bE * bE)) /
                    square(B0);

  p1 += delta_p1;
  p2 += delta_p2;
  p3 += delta_p3;
  // p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

template <typename Scalar>
HD_INLINE void
sync_kill_perp(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
               Scalar E1, Scalar E2, Scalar E3, Scalar B1,
               Scalar B2, Scalar B3, Scalar q_over_m, Scalar cooling_coef,
               Scalar B0) {
  // B1 /= q_over_m;
  // B2 /= q_over_m;
  // B3 /= q_over_m;
  Scalar p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  Scalar B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3);

  Scalar delta_p1 = -cooling_coef *
                    (p1 - B1 * pdotB / B_sqr);
  Scalar delta_p2 = -cooling_coef *
                    (p2 - B2 * pdotB / B_sqr);
  Scalar delta_p3 = -cooling_coef *
                    (p3 - B3 * pdotB / B_sqr);
  // Scalar dp = sqrt(delta_p1 * delta_p1 + delta_p2 * delta_p2 +
  //                  delta_p3 * delta_p3);
  // Scalar f = cube(math::sqrt(B_sqr) / B0);
  // Scalar f = B_sqr / square(B0);
  Scalar f = math::sqrt(B_sqr) / B0;
  // Scalar f = 1.0f;
  // if (sp == (int)ParticleType::ion) f *= 0.1f;
  p1 += delta_p1 * f;
  p2 += delta_p2 * f;
  p3 += delta_p3 * f;
  // p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

template <typename Scalar>
HD_INLINE void
sync_kill_gyration(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                   Scalar E1, Scalar E2, Scalar E3, Scalar B1,
                   Scalar B2, Scalar B3, Scalar q_over_m, Scalar cooling_coef,
                   Scalar B0) {
  // Step 1: recover velocities
  // p1 /= gamma;
  // p2 /= gamma;
  // p3 /= gamma;

  // Step 2: find drift velocity
  Scalar B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  Scalar EB = square(E1) + square(E2) + square(E3) + B_sqr;
  Scalar vd1 = (E2 * B3 - E3 * B2) / EB;
  Scalar vd2 = (E3 * B1 - E1 * B3) / EB;
  Scalar vd3 = (E1 * B2 - E2 * B1) / EB;
  Scalar vd_sqr = square(vd1) + square(vd2) + square(vd3);
  printf("vd1 is %f, vd2 is %f, vd3 is %f, EB is %f, vd_sqr is %f\n", vd1, vd2, vd3, EB, vd_sqr);

  if (vd_sqr > 1.0e-5) {
    vd1 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
    vd2 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
    vd3 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
  }


  // Step 3: find gyration velocity and use it to compute delta_v
  Scalar vdotB = (p1 * B1 + p2 * B2 + p3 * B3);

  Scalar delta_v1 = -cooling_coef *
                    (p1 - B1 * vdotB / B_sqr - gamma * vd1);
  Scalar delta_v2 = -cooling_coef *
                    (p2 - B2 * vdotB / B_sqr - gamma * vd2);
  Scalar delta_v3 = -cooling_coef *
                    (p3 - B3 * vdotB / B_sqr - gamma * vd3);

  Scalar f = math::sqrt(B_sqr) / B0;
 
  p1 += delta_v1 * f;
  p2 += delta_v2 * f;
  p3 += delta_v3 * f;

  // Step 4: recompute gamma and p
  // gamma = 1.0f / math::sqrt(1.0f - p1 * p1 - p2 * p2 - p3 * p3);
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  // p1 *= gamma;
  // p2 *= gamma;
  // p3 *= gamma;
}


}

#endif // __SYNC_COOLING_H_
