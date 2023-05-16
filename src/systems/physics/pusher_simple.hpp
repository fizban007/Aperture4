/*
 * Copyright (c) 2023 Alex Chen.
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

#include "systems/grid.h"
#include "systems/physics/pushers.hpp"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
class pusher_simple {
 public:
  using value_t = typename Conf::value_t;
  using vec3 = vec_t<value_t, 3>;
  using grid_type = grid_t<Conf>;

  pusher_simple(const grid_t<Conf>& grid) : m_grid(grid) {}
  ~pusher_simple() = default;

  void init() {}

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HOST_DEVICE void push(const Grid<Conf::dim, value_t>& grid,
                        const extent_t<Conf::dim>& ext, PtcContext& context,
                        vec_t<UIntT, Conf::dim>& pos, value_t dt) const {
    m_pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
             decltype(context.q)(dt));
  }

 private:
  const grid_t<Conf>& m_grid;
  mutable typename Conf::pusher_t m_pusher;
};

}  // namespace Aperture
