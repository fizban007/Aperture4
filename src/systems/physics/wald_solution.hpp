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

namespace Aperture {

template <typename Conf, typename value_t = typename Conf::value_t>
HOST_DEVICE value_t
gr_ks_dA0dr(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return a * (1.0f + cth * cth) * (-2.0f * r * r / rho2 + 1.0f) / rho2;
}

template <typename Conf, typename value_t = typename Conf::value_t>
HOST_DEVICE value_t
gr_ks_dA0dth(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return 2.0f * a * r * sth * cth *
         (2.0f * a * a * (1.0f + cth * cth) / rho2 - 1.0f) / rho2;
}

template <typename Conf, typename value_t = typename Conf::value_t>
HOST_DEVICE value_t
gr_ks_dAphdr(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return r + a * a * (1.0f + cth * cth) * (2.0f * r * r / rho2 - 1.0f) / rho2;
}

template <typename Conf, typename value_t = typename Conf::value_t>
HOST_DEVICE value_t
gr_ks_dAphdth(value_t a, value_t r, value_t sth, value_t cth) {
  value_t rho2 = r * r + a * a * cth * cth;
  return 2.0f * a * a * r * cth * sth *
         (-a * a * (1.0f + cth * cth) / rho2 + 1.0f) / rho2;
}

template <typename Conf, typename value_t = typename Conf::value_t>
HOST_DEVICE value_t
gr_wald_solution_B(value_t a, value_t r, value_t th, value_t Bp,
                   int component) {
  value_t sth = math::sin(th);
  value_t cth = math::cos(th);
  value_t rho2 = Metric_KS::rho2(a, r, sth, cth);

  if (component == 2) {
    return 0.0;
  } else if (component == 1) {
    // Avoid axis singularity
    if (math::abs(th) < TINY || math::abs(th - M_PI) < TINY) {
      return 0.0;
    } else {
      return -Bp * gr_ks_dAphdr<Conf>(a, r, sth, cth) /
             Metric_KS::sqrt_gamma(a, r, sth, cth);
    }
  } else if (component == 0) {
    return Bp * gr_ks_dAphdth<Conf>(a, r, sth, cth) /
           Metric_KS::sqrt_gamma(a, r, sth, cth);
  }
  return 0.0;
}

template <typename Conf, typename value_t = typename Conf::value_t>
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
             (-Metric_KS::gu33(a, r, sth, cth) *
                  Metric_KS::beta1(a, r, sth, cth) *
                  gr_ks_dAphdr<Conf>(a, r, sth, cth) +
              Metric_KS::gu13(a, r, sth, cth) *
                  gr_ks_dA0dr<Conf>(a, r, sth, cth)) /
             Metric_KS::alpha(a, r, sth, cth);
    }
  } else if (component == 1) {
    return Bp * Metric_KS::gu22(a, r, sth, cth) *
           gr_ks_dA0dth<Conf>(a, r, sth, cth) /
           Metric_KS::alpha(a, r, sth, cth);
  } else if (component == 0) {
    return Bp *
           (Metric_KS::gu11(a, r, sth, cth) *
                gr_ks_dA0dr<Conf>(a, r, sth, cth) -
            Metric_KS::gu13(a, r, sth, cth) * Metric_KS::beta1(a, r, sth, cth) *
                gr_ks_dAphdr<Conf>(a, r, sth, cth)) /
           Metric_KS::alpha(a, r, sth, cth);
  }
  return 0.0;
}

}  // namespace Aperture

#endif  // __WALD_SOLUTION_H_
