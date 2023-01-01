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
#include "framework/environment.h"
#include "systems/grid_sph.hpp"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
class ptc_physics_policy_gravity_sph {
 public:
  using value_t = typename Conf::value_t;

  void init() { sim_env().params().get_value("gravity", m_g); }

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context,
                            const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {
    auto x1 = grid.coord(0, pos[0], context.x[0]);
    x1 = grid_sph_t<Conf>::radius(x1);
    context.p[0] -= m_g * dt / square(x1);
    context.gamma = sqrt(1.0f + context.p.dot(context.p));
  }

 private:
  value_t m_g = 1.0f;
};

}  // namespace Aperture
