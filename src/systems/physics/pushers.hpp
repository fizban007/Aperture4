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

#ifndef __PUSHERS_H_
#define __PUSHERS_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "utils/util_functions.h"

namespace Aperture {

enum class Pusher : char { boris, vay, higuera };

struct boris_pusher {
  template <typename ptc_float_t, typename field_t, typename value_t>
  HD_INLINE void operator()(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3,
                            ptc_float_t& gamma, field_t E1, field_t E2,
                            field_t E3, field_t B1, field_t B2, field_t B3,
                            value_t qdt_over_2m, value_t dt) {
    E1 *= qdt_over_2m;
    E2 *= qdt_over_2m;
    E3 *= qdt_over_2m;

    auto pm1 = p1 + E1;
    auto pm2 = p2 + E2;
    auto pm3 = p3 + E3;
    auto gamma_m = math::sqrt(pm1 * pm1 + pm2 * pm2 + pm3 * pm3 + 1.0F);
    auto t1 = B1 * qdt_over_2m / gamma_m;
    auto t2 = B2 * qdt_over_2m / gamma_m;
    auto t3 = B3 * qdt_over_2m / gamma_m;
    auto t_sqr = t1 * t1 + t2 * t2 + t3 * t3;
    auto pt1 = pm2 * t3 - pm3 * t2 + pm1;
    auto pt2 = pm3 * t1 - pm1 * t3 + pm2;
    auto pt3 = pm1 * t2 - pm2 * t1 + pm3;

    p1 = pm1 + E1 + (pt2 * t3 - pt3 * t2) * 2.0F / (t_sqr + 1.0F);
    p2 = pm2 + E2 + (pt3 * t1 - pt1 * t3) * 2.0F / (t_sqr + 1.0F);
    p3 = pm3 + E3 + (pt1 * t2 - pt2 * t1) * 2.0F / (t_sqr + 1.0F);
    gamma = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3 + 1.0F);
  }
};

struct vay_pusher {
  template <typename ptc_float_t, typename value_t>
  HD_INLINE void operator()(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3,
                            ptc_float_t& gamma, value_t E1, value_t E2,
                            value_t E3, value_t B1, value_t B2, value_t B3,
                            value_t qdt_over_2m, value_t dt) {
    E1 *= qdt_over_2m;
    E2 *= qdt_over_2m;
    E3 *= qdt_over_2m;
    B1 *= qdt_over_2m;
    B2 *= qdt_over_2m;
    B3 *= qdt_over_2m;

    value_t up1 = p1 + 2.0F * E1 + (p2 * B3 - p3 * B2) / gamma;
    value_t up2 = p2 + 2.0F * E2 + (p3 * B1 - p1 * B3) / gamma;
    value_t up3 = p3 + 2.0F * E3 + (p1 * B2 - p2 * B1) / gamma;
    // printf("p prime is (%f, %f, %f), gamma is %f\n", up1, up2, up3,
    // gamma);
    value_t tt = B1 * B1 + B2 * B2 + B3 * B3;
    value_t ut = up1 * B1 + up2 * B2 + up3 * B3;

    value_t sigma = 1.0F + up1 * up1 + up2 * up2 + up3 * up3 - tt;
    value_t inv_gamma2 =
        2.0F / (sigma + math::sqrt(sigma * sigma + 4.0F * (tt + ut * ut)));
    value_t s = 1.0F / (1.0F + inv_gamma2 * tt);
    gamma = 1.0F / math::sqrt(inv_gamma2);

    p1 = (up1 + B1 * ut * inv_gamma2 + (up2 * B3 - up3 * B2) / gamma) * s;
    p2 = (up2 + B2 * ut * inv_gamma2 + (up3 * B1 - up1 * B3) / gamma) * s;
    p3 = (up3 + B3 * ut * inv_gamma2 + (up1 * B2 - up2 * B1) / gamma) * s;
  }
};

struct higuera_pusher {
  template <typename ptc_float_t, typename value_t>
  HD_INLINE void operator()(ptc_float_t& p1, ptc_float_t& p2, ptc_float_t& p3,
                            ptc_float_t& gamma, value_t E1, value_t E2,
                            value_t E3, value_t B1, value_t B2, value_t B3,
                            value_t qdt_over_2m, value_t dt) {
    value_t pm1 = p1 + E1 * qdt_over_2m;
    value_t pm2 = p2 + E2 * qdt_over_2m;
    value_t pm3 = p3 + E3 * qdt_over_2m;
    value_t gm2 = 1.0F + pm1 * pm1 + pm2 * pm2 + pm3 * pm3;
    value_t b_dot_p = (B1 * pm1 + B2 * pm2 + B3 * pm3) * qdt_over_2m;
    value_t b_sqr = (B1 * B1 + B2 * B2 + B3 * B3) * square(qdt_over_2m);
    value_t gamma_new = math::sqrt(
        0.5F *
        (gm2 - b_sqr +
         math::sqrt(square(gm2 - b_sqr) + 4.0F * (b_sqr + b_dot_p * b_dot_p))));

    value_t t_sqr = b_sqr / square(gamma_new);
    value_t pt1 = (pm2 * B3 - pm3 * B2) * qdt_over_2m / gamma_new + pm1;
    value_t pt2 = (pm3 * B1 - pm1 * B3) * qdt_over_2m / gamma_new + pm2;
    value_t pt3 = (pm1 * B2 - pm2 * B1) * qdt_over_2m / gamma_new + pm3;

    p1 =
        pm1 + E1 * qdt_over_2m +
        (pt2 * B3 - pt3 * B2) * 2.0F / (1.0F + t_sqr) * qdt_over_2m / gamma_new;
    p2 =
        pm2 + E2 * qdt_over_2m +
        (pt3 * B1 - pt1 * B3) * 2.0F / (1.0F + t_sqr) * qdt_over_2m / gamma_new;
    p3 =
        pm3 + E3 * qdt_over_2m +
        (pt1 * B2 - pt2 * B1) * 2.0F / (1.0F + t_sqr) * qdt_over_2m / gamma_new;
    gamma = math::sqrt(1.0F + p1 * p1 + p2 * p2 + p3 * p3);
  }
};

// using default_pusher = higuera_pusher;
// using default_pusher = boris_pusher;
using default_pusher = vay_pusher;

}  // namespace Aperture

#endif  // __PUSHERS_H_
