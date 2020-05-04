#ifndef __PUSHERS_H_
#define __PUSHERS_H_

#include "core/cuda_control.h"
#include "utils/util_functions.h"
#include <cmath>

namespace Aperture {

enum class Pusher : char { boris, vay, higuera };

struct vay_pusher {
  template <typename Scalar>
  HD_INLINE void operator()(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                            Scalar E1, Scalar E2, Scalar E3, Scalar B1,
                            Scalar B2, Scalar B3, Scalar qdt_over_2m,
                            Scalar dt) {
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
};

struct boris_pusher {
  template <typename Scalar>
  HD_INLINE void operator()(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                            Scalar E1, Scalar E2, Scalar E3, Scalar B1,
                            Scalar B2, Scalar B3, Scalar qdt_over_2m,
                            Scalar dt) {
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
};

struct higuera_pusher {
  // template <typename Scalar>
  // HD_INLINE void operator()(Scalar& p1, Scalar& p2, Scalar& p3, Scalar&
  // gamma,
  //                           Scalar E1, Scalar E2, Scalar E3, Scalar B1,
  //                           Scalar B2, Scalar B3, Scalar qdt_over_2m,
  //                           Scalar dt) {
  //   E1 *= qdt_over_2m;
  //   E2 *= qdt_over_2m;
  //   E3 *= qdt_over_2m;
  //   B1 *= qdt_over_2m;
  //   B2 *= qdt_over_2m;
  //   B3 *= qdt_over_2m;

  //   Scalar pm1 = p1 + E1;
  //   Scalar pm2 = p2 + E2;
  //   Scalar pm3 = p3 + E3;
  //   Scalar gm2 = 1.0f + pm1 * pm1 + pm2 * pm2 + pm3 * pm3;
  //   Scalar b_dot_p = B1 * pm1 + B2 * pm2 + B3 * pm3;
  //   Scalar b_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  //   Scalar gamma_new = std::sqrt(0.5f * (gm2 - b_sqr +
  //                                        std::sqrt(square(gm2 - b_sqr) +
  //                                                  4.0f * (b_sqr + b_dot_p *
  //                                                  b_dot_p))));
  //   B1 /= gamma_new;
  //   B2 /= gamma_new;
  //   B3 /= gamma_new;
  //   Scalar t_sqr = B1 * B1 + B2 * B2 + B3 * B3;
  //   Scalar pt1 = pm2 * B3 - pm3 * B2 + pm1;
  //   Scalar pt2 = pm3 * B1 - pm1 * B3 + pm2;
  //   Scalar pt3 = pm1 * B2 - pm2 * B1 + pm3;

  //   p1 = pm1 + E1 + (pt2 * B3 - pt3 * B2) * 2.0f / (1.0f + t_sqr);
  //   p2 = pm2 + E2 + (pt3 * B1 - pt1 * B3) * 2.0f / (1.0f + t_sqr);
  //   p3 = pm3 + E3 + (pt1 * B2 - pt2 * B1) * 2.0f / (1.0f + t_sqr);
  //   gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  // }
  template <typename Scalar>
  HD_INLINE void operator()(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                            Scalar E1, Scalar E2, Scalar E3, Scalar B1,
                            Scalar B2, Scalar B3, Scalar qdt_over_2m,
                            Scalar dt) {
    Scalar pm1 = p1 + E1 * qdt_over_2m;
    Scalar pm2 = p2 + E2 * qdt_over_2m;
    Scalar pm3 = p3 + E3 * qdt_over_2m;
    Scalar gm2 = 1.0f + pm1 * pm1 + pm2 * pm2 + pm3 * pm3;
    Scalar b_dot_p = (B1 * pm1 + B2 * pm2 + B3 * pm3) * qdt_over_2m;
    Scalar b_sqr = (B1 * B1 + B2 * B2 + B3 * B3) * square(qdt_over_2m);
    Scalar gamma_new = std::sqrt(
        0.5f *
        (gm2 - b_sqr +
         std::sqrt(square(gm2 - b_sqr) + 4.0f * (b_sqr + b_dot_p * b_dot_p))));

    Scalar t_sqr = b_sqr / square(gamma_new);
    Scalar pt1 = (pm2 * B3 - pm3 * B2) / gamma_new + pm1;
    Scalar pt2 = (pm3 * B1 - pm1 * B3) / gamma_new + pm2;
    Scalar pt3 = (pm1 * B2 - pm2 * B1) / gamma_new + pm3;

    p1 =
        pm1 + E1 * qdt_over_2m +
        (pt2 * B3 - pt3 * B2) * 2.0f / (1.0f + t_sqr) * qdt_over_2m / gamma_new;
    p2 =
        pm2 + E2 * qdt_over_2m +
        (pt3 * B1 - pt1 * B3) * 2.0f / (1.0f + t_sqr) * qdt_over_2m / gamma_new;
    p3 =
        pm3 + E3 * qdt_over_2m +
        (pt1 * B2 - pt2 * B1) * 2.0f / (1.0f + t_sqr) * qdt_over_2m / gamma_new;
    gamma = std::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  }
};

using default_pusher = higuera_pusher;

}  // namespace Aperture

#endif  // __PUSHERS_H_
