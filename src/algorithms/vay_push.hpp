#ifndef __VAY_PUSH_H_
#define __VAY_PUSH_H_

#include "core/cuda_control.h"
#include <cmath>

namespace Aperture {

namespace Kernels {

template <typename Scalar>
HD_INLINE void
vay_push(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma, Scalar E1,
         Scalar E2, Scalar E3, Scalar B1, Scalar B2, Scalar B3,
         Scalar qdt_over_2m, Scalar dt) {
  E1 *= qdt_over_2m;
  E2 *= qdt_over_2m;
  E3 *= qdt_over_2m;
  B1 *= qdt_over_2m;
  B2 *= qdt_over_2m;
  B3 *= qdt_over_2m;

  Scalar up1 = p1 + 2.0f * E1 + (p2 * B3 - p3 * B2) / gamma;
  Scalar up2 = p2 + 2.0f * E2 + (p3 * B1 - p1 * B3) / gamma;
  Scalar up3 = p3 + 2.0f * E3 + (p1 * B2 - p2 * B1) / gamma;
  // printf("p prime is (%f, %f, %f), gamma is %f\n", up1, up2, up3,
  // gamma);
  Scalar tt = B1 * B1 + B2 * B2 + B3 * B3;
  Scalar ut = up1 * B1 + up2 * B2 + up3 * B3;

  Scalar sigma = 1.0f + up1 * up1 + up2 * up2 + up3 * up3 - tt;
  Scalar inv_gamma2 =
      2.0f / (sigma + std::sqrt(sigma * sigma + 4.0f * (tt + ut * ut)));
  Scalar s = 1.0f / (1.0f + inv_gamma2 * tt);
  gamma = 1.0f / std::sqrt(inv_gamma2);

  p1 = (up1 + B1 * ut * inv_gamma2 + (up2 * B3 - up3 * B2) / gamma) * s;
  p2 = (up2 + B2 * ut * inv_gamma2 + (up3 * B1 - up1 * B3) / gamma) * s;
  p3 = (up3 + B3 * ut * inv_gamma2 + (up1 * B2 - up2 * B1) / gamma) * s;
}

}  // namespace Kernels

}  // namespace Aperture

#endif  // __VAY_PUSH_H_
