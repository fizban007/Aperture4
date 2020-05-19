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
               Scalar B1, Scalar B2, Scalar B3, Scalar E1,
               Scalar E2, Scalar E3, Scalar q_over_m, Scalar cooling_coef,
               Scalar B0) {
  B1 /= q_over_m;
  B2 /= q_over_m;
  B3 /= q_over_m;
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
  Scalar f = math::sqrt(B_sqr) / B0;
  // Scalar f = B_sqr / square(dev_params.B0);
  // Scalar f = 1.0f;
  // if (sp == (int)ParticleType::ion) f *= 0.1f;
  p1 += delta_p1 * f;
  p2 += delta_p2 * f;
  p3 += delta_p3 * f;
  p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  gamma = math::sqrt(1.0f + p * p);
}


}

#endif // __SYNC_COOLING_H_
