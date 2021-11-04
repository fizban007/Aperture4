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

#ifndef _PHYS_POLICY_IC_COOLING_H_
#define _PHYS_POLICY_IC_COOLING_H_

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "core/math.hpp"
#include "framework/environment.h"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
class phys_policy_IC_cooling {
 public:
  using value_t = typename Conf::value_t;

  void init() {
    if (sim_env().params().has("gamma_IC") && sim_env().params().has("sigma")) {
      value_t gamma_IC = sim_env().params().get_as<double>("gamma_IC", 0.0);
      value_t sigma = sim_env().params().get_as<double>("sigma", 1.0);
      if (gamma_IC <= 0.0) {
        m_IC_coef = 0.0;
      } else {
        m_IC_coef = 0.3 * math::sqrt(sigma) / (4.0 * square(gamma_IC));
      }
    } else {
      sim_env().params().get_value("IC_compactness", m_IC_coef);
    }
  }

  template <typename ExecPolicy>
  void update(const ExecPolicy& policy) {}

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context,
                            const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {
    auto gamma = context.gamma;
    if (gamma - 1.0f < 1.0e-4) return;
    context.p[0] -= (4.0f / 3.0f) * m_IC_coef * gamma * context.p[0] * dt;
    context.p[1] -= (4.0f / 3.0f) * m_IC_coef * gamma * context.p[1] * dt;
    context.p[2] -= (4.0f / 3.0f) * m_IC_coef * gamma * context.p[2] * dt;

    value_t p_sqr = context.p.dot(context.p);
    context.gamma = sqrt(1.0f + p_sqr);
  }

 private:
  // IC coef is defined as sigma_T U_ph / m_e c omega, where omega is the inerse
  // time unit. f_IC = 0.3 * sqrt(sigma) / (4 * gamma_rad^2)
  value_t m_IC_coef = 0.1f;
};

}


#endif  // _PHYS_POLICY_IC_COOLING_H_
