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

#ifndef __WALD_SOLUTION_H_
#define __WALD_SOLUTION_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "systems/grid_ks.h"
#include "systems/physics/metric_kerr_schild.hpp"

namespace Aperture {

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_A0(value_t a, value_t r, value_t sth, value_t cth) {
  return a * r * (1.0f + cth * cth) / (r * r + a * a * cth * cth) - a;
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_Aphi(value_t a, value_t r, value_t sth, value_t cth) {
  return 0.5f * sth * sth *
         (r * r + a * a -
          2.0f * a * a * r * (1.0f + cth * cth) / (r * r + a * a * cth * cth));
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_dA0dr(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return a * (1.0f + cth * cth) * (square(a * cth) - r * r) / square(rho2);
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_dA0dth(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return 2.0f * a * r * sth * cth * (a * a - r * r) / square(rho2);
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_dAphdr(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return (r +
          a * a * (1.0f + cth * cth) * (2.0f * r * r / rho2 - 1.0f) / rho2) *
         sth * sth;
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_dAphdth(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  // return (2.0f * a * a * r * cth * sth *
  //         (-a * a * (1.0f + cth * cth) / rho2 + 1.0f) / rho2) *
  //            sth * sth +
  //        sth * cth *
  //            (a * a + r * r - 2.0f * a * a * r * (1.0f + cth * cth) / rho2);
  return (a * a + r * r - 2.0f * r) * sth * cth +
         2.0f * r * (r * r * r * r - a * a * a * a) * sth * cth / square(rho2);
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
wald_ks_dArdth(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return -a * sth * cth +
         2.0f * a * r * (a * a - r * r) * sth * cth / square(rho2);
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
gr_wald_solution_B(value_t a, value_t r, value_t th, value_t Bp,
                   int component) {
  value_t sth = math::sin(th);
  value_t cth = math::cos(th);
  value_t rho2 = Metric_KS::rho2(a, r, sth, cth);

  if (component == 2) {
    return -Bp * wald_ks_dArdth(a, r, sth, cth) /
           Metric_KS::sqrt_gamma(a, r, sth, cth);
  } else if (component == 1) {
    // Avoid axis singularity
    if (math::abs(th) < TINY || math::abs(th - M_PI) < TINY) {
      return 0.0;
    } else {
      return -Bp * wald_ks_dAphdr(a, r, sth, cth) /
             Metric_KS::sqrt_gamma(a, r, sth, cth);
    }
  } else if (component == 0) {
    return Bp * wald_ks_dAphdth(a, r, sth, cth) /
           Metric_KS::sqrt_gamma(a, r, sth, cth);
  }
  return 0.0;
}

template <typename value_t = Scalar>
HOST_DEVICE value_t
gr_wald_solution_D(value_t a, value_t r, value_t th, value_t Bp,
                   int component) {
  value_t sth = math::sin(th);
  value_t cth = math::cos(th);
  value_t rho2 = Metric_KS::rho2(a, r, sth, cth);

  if (component == 2) {
    // Avoid axis singularity
    if (math::abs(th) < TINY || math::abs(th - M_PI) < TINY) {
      return 0.0;
    } else {
      return Bp *
             (Metric_KS::gu33(a, r, sth, cth) *
                  Metric_KS::beta1(a, r, sth, cth) *
                  wald_ks_dAphdr(a, r, sth, cth) +
              Metric_KS::gu13(a, r, sth, cth) * wald_ks_dA0dr(a, r, sth, cth)) /
             Metric_KS::alpha(a, r, sth, cth);
    }
  } else if (component == 1) {
    return Bp * Metric_KS::gu22(a, r, sth, cth) *
           (wald_ks_dA0dth(a, r, sth, cth) -
            Metric_KS::beta1(a, r, sth, cth) * wald_ks_dArdth(a, r, sth, cth)) /
           Metric_KS::alpha(a, r, sth, cth);
  } else if (component == 0) {
    return Bp *
           (Metric_KS::gu11(a, r, sth, cth) * wald_ks_dA0dr(a, r, sth, cth) +
            Metric_KS::gu13(a, r, sth, cth) * Metric_KS::beta1(a, r, sth, cth) *
                wald_ks_dAphdr(a, r, sth, cth)) /
           Metric_KS::alpha(a, r, sth, cth);
  }
  return 0.0;
}

}  // namespace Aperture

#endif  // __WALD_SOLUTION_H_
