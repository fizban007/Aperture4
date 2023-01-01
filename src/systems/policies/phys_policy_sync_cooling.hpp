/*
 * Copyright (c) 2021 Alex Chen.
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

#pragma once

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "core/math.hpp"
#include "framework/environment.h"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
class phys_policy_sync_cooling {
 public:
  using value_t = typename Conf::value_t;

  void init() {
    sim_env().params().get_value("sync_cooling_coef", m_cooling_coef);
    sim_env().params().get_value("Bp", m_B0);
  }

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context,
                            const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {
    // value_t q_over_m = context.q / context.m;
    value_t q_over_m = math::abs(context.q) / context.m;
    value_t tmp1 = (context.E[0] + (context.p[1] * context.B[2] -
                                    context.p[2] * context.B[1]) /
                                       context.gamma) /
                   q_over_m;
    value_t tmp2 = (context.E[1] + (context.p[2] * context.B[0] -
                                    context.p[0] * context.B[2]) /
                                       context.gamma) /
                   q_over_m;
    value_t tmp3 = (context.E[2] + (context.p[0] * context.B[1] -
                                    context.p[3] * context.B[0]) /
                                       context.gamma) /
                   q_over_m;
    value_t tmp_sq = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
    value_t bE = (context.p.dot(context.E)) / (context.gamma * q_over_m);

    value_t delta_p1 =
        m_cooling_coef *
        (((tmp2 * context.B[2] - tmp3 * context.B[1]) + bE * context.E[0]) /
             q_over_m -
         context.gamma * context.p[0] * (tmp_sq - bE * bE)) /
        square(m_B0);
    value_t delta_p2 =
        m_cooling_coef *
        (((tmp3 * context.B[0] - tmp1 * context.B[2]) + bE * context.E[1]) /
             q_over_m -
         context.gamma * context.p[1] * (tmp_sq - bE * bE)) /
        square(m_B0);
    value_t delta_p3 =
        m_cooling_coef *
        (((tmp1 * context.B[1] - tmp2 * context.B[0]) + bE * context.E[2]) /
             q_over_m -
         context.gamma * context.p[2] * (tmp_sq - bE * bE)) /
        square(m_B0);

    context.p[0] += delta_p1 * dt / context.gamma;
    context.p[1] += delta_p2 * dt / context.gamma;
    context.p[2] += delta_p3 * dt / context.gamma;

    value_t p_sqr = context.p.dot(context.p);
    context.gamma = sqrt(1.0f + p_sqr);

    // pitch angle
    value_t mu = context.p.dot(context.B) / math::sqrt(p_sqr) /
                 math::sqrt(context.B.dot(context.B));
    printf("pitch angle is %f, gamma is %f, beta_para is %f\n", acos(mu),
           context.gamma, math::sqrt(p_sqr) / context.gamma * mu);
  }

 private:
  value_t m_cooling_coef = 1.0f;
  value_t m_B0 = 1000.0f;
};

}  // namespace Aperture
