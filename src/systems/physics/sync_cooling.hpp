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

template <typename ptc_float_t, typename value_t>
HD_INLINE void
sync_cooling(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3, ptc_float_t& gamma,
             value_t B1, value_t B2, value_t B3, value_t E1, value_t E2,
             value_t E3, value_t q_over_m, value_t cooling_coef, value_t B0) {
  value_t tmp1 = (E1 + (p2 * B3 - p3 * B2) / gamma) / q_over_m;
  value_t tmp2 = (E2 + (p3 * B1 - p1 * B3) / gamma) / q_over_m;
  value_t tmp3 = (E3 + (p1 * B2 - p2 * B1) / gamma) / q_over_m;
  value_t tmp_sq = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
  value_t bE = (p1 * E1 + p2 * E2 + p3 * E3) / (gamma * q_over_m);

  value_t delta_p1 = cooling_coef *
                    (((tmp2 * B3 - tmp3 * B2) + bE * E1) / q_over_m -
                     gamma * p1 * (tmp_sq - bE * bE)) /
                    square(B0);
  value_t delta_p2 = cooling_coef *
                    (((tmp3 * B1 - tmp1 * B3) + bE * E2) / q_over_m -
                     gamma * p2 * (tmp_sq - bE * bE)) /
                    square(B0);
  value_t delta_p3 = cooling_coef *
                    (((tmp1 * B2 - tmp2 * B1) + bE * E3) / q_over_m -
                     gamma * p3 * (tmp_sq - bE * bE)) /
                    square(B0);

  p1 += delta_p1;
  p2 += delta_p2;
  p3 += delta_p3;
  // p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

template <typename ptc_float_t, typename value_t>
HD_INLINE void
sync_kill_perp(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3, ptc_float_t& gamma,
               value_t E1, value_t E2, value_t E3, value_t B1,
               value_t B2, value_t B3, value_t q_over_m, value_t cooling_coef,
               value_t B0) {
  // B1 /= q_over_m;
  // B2 /= q_over_m;
  // B3 /= q_over_m;
  value_t p = sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  value_t B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  value_t pdotB = (p1 * B1 + p2 * B2 + p3 * B3);

  value_t delta_p1 = -cooling_coef *
                    (p1 - B1 * pdotB / B_sqr);
  value_t delta_p2 = -cooling_coef *
                    (p2 - B2 * pdotB / B_sqr);
  value_t delta_p3 = -cooling_coef *
                    (p3 - B3 * pdotB / B_sqr);
  // value_t dp = sqrt(delta_p1 * delta_p1 + delta_p2 * delta_p2 +
  //                  delta_p3 * delta_p3);
  // value_t f = cube(math::sqrt(B_sqr) / B0);
  // value_t f = B_sqr / square(B0);
  value_t f = math::sqrt(B_sqr) / B0;
  // value_t f = 1.0f;
  // if (sp == (int)ParticleType::ion) f *= 0.1f;
  p1 += delta_p1 * f;
  p2 += delta_p2 * f;
  p3 += delta_p3 * f;
  // p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

template <typename ptc_float_t, typename value_t>
HD_INLINE void
sync_kill_gyration(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3, ptc_float_t& gamma,
                   value_t E1, value_t E2, value_t E3, value_t B1,
                   value_t B2, value_t B3, value_t q_over_m, value_t cooling_coef,
                   value_t B0) {
  // Step 1: recover velocities
  // p1 /= gamma;
  // p2 /= gamma;
  // p3 /= gamma;

  // Step 2: find drift velocity
  value_t B_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  value_t EB = square(E1) + square(E2) + square(E3) + B_sqr;
  value_t vd1 = (E2 * B3 - E3 * B2) / EB;
  value_t vd2 = (E3 * B1 - E1 * B3) / EB;
  value_t vd3 = (E1 * B2 - E2 * B1) / EB;
  value_t vd_sqr = square(vd1) + square(vd2) + square(vd3);
  // printf("vd1 is %f, vd2 is %f, vd3 is %f, EB is %f, vd_sqr is %f\n", vd1, vd2, vd3, EB, vd_sqr);

  if (vd_sqr > 1.0e-5) {
    vd1 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
    vd2 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
    vd3 *= (1.0f - math::sqrt(max(1.0f - 4.0f * vd_sqr, 0.0f))) / (2.0f * vd_sqr);
  }


  // Step 3: find gyration velocity and use it to compute delta_v
  value_t vdotB = (p1 * B1 + p2 * B2 + p3 * B3);

  value_t delta_v1 = -cooling_coef *
                    (p1 - B1 * vdotB / B_sqr - gamma * vd1);
  value_t delta_v2 = -cooling_coef *
                    (p2 - B2 * vdotB / B_sqr - gamma * vd2);
  value_t delta_v3 = -cooling_coef *
                    (p3 - B3 * vdotB / B_sqr - gamma * vd3);

  value_t f = math::sqrt(B_sqr) / B0;
 
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
