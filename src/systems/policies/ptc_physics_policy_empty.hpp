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

#ifndef __PTC_PHYSICS_POLICY_EMPTY_H_
#define __PTC_PHYSICS_POLICY_EMPTY_H_

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "utils/vec.hpp"

namespace Aperture {

template <typename Conf>
class ptc_physics_policy_empty {
 public:
  using value_t = typename Conf::value_t;

  void init() {}

  void update() {}

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context, const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {}
};

}  // namespace Aperture

#endif  // __PTC_PHYSICS_POLICY_EMPTY_H_
