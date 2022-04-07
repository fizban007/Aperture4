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

#ifndef _COORD_POLICY_CARTESIAN_SYNC_IC_COOLING_H_
#define _COORD_POLICY_CARTESIAN_SYNC_IC_COOLING_H_

#include "systems/policies/coord_policy_cartesian_impl_cooling.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian_sync_IC_cooling
    : public coord_policy_cartesian_impl_cooling<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using vec3 = vec_t<value_t, 3>;
  using coord_policy_cartesian<Conf>::coord_policy_cartesian;

  void init() {
    value_t gamma_sync = 100.0f, gamma_IC = 100.0f, sigma = 10.0f;
    sim_env().params().get_value("gamma_sync", gamma_sync);
    sim_env().params().get_value("gamma_IC", gamma_IC);
    sim_env().params().get_value("sigma", sigma);
  }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            vec_t<UIntT, Conf::dim>& pos, value_t dt) const {
    iterate(context.x, context.p, context.E, context.B, context.q / context.m,
            m_sync_coef, dt);

    // TODO: add IC cooling
    // TODO: add sync and IC radiation output
    context.gamma = math::sqrt(1.0f + context.p.dot(context.p));
    move_ptc(grid, context, pos, dt);
  }

 private:
  value_t m_sync_coef = 0.0f;
  value_t m_IC_coef = 0.0f;
};

}

#endif  // _COORD_POLICY_CARTESIAN_SYNC_IC_COOLING_H_
