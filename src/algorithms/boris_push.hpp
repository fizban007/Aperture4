#ifndef __BORIS_PUSH_H_
#define __BORIS_PUSH_H_

#include "core/cuda_control.h"
#include <cmath>

namespace Aperture {

namespace Kernels {

template <typename Scalar>
HD_INLINE void
boris_push(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma, Scalar E1,
           Scalar E2, Scalar E3, Scalar B1, Scalar B2, Scalar B3,
           Scalar qdt_over_2m, Scalar dt) {
  Scalar pm1 = p1 + E1 * qdt_over_2m;
  Scalar pm2 = p2 + E2 * qdt_over_2m;
  Scalar pm3 = p3 + E3 * qdt_over_2m;
  Scalar gamma_m = std::sqrt(1.0f + pm1 * pm1 + pm2 * pm2 + pm3 * pm3);
  Scalar t1 = B1 * qdt_over_2m / gamma_m;
  Scalar t2 = B2 * qdt_over_2m / gamma_m;
  Scalar t3 = B3 * qdt_over_2m / gamma_m;
  Scalar t_sqr = t1 * t1 + t2 * t2 + t3 * t3;
  Scalar pt1 = pm2 * t3 - pm3 * t2 + pm1;
  Scalar pt2 = pm3 * t1 - pm1 * t3 + pm2;
  Scalar pt3 = pm1 * t2 - pm2 * t1 + pm3;

  p1 = pm1 + E1 * qdt_over_2m + (pt2 * t3 - pt3 * t2) * 2.0f / (1.0f + t_sqr);
  p2 = pm2 + E2 * qdt_over_2m + (pt3 * t1 - pt1 * t3) * 2.0f / (1.0f + t_sqr);
  p3 = pm3 + E3 * qdt_over_2m + (pt1 * t2 - pt2 * t1) * 2.0f / (1.0f + t_sqr);
  gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
}

}  // namespace Kernels

}  // namespace Aperture

#endif  // __BORIS_PUSH_H_
