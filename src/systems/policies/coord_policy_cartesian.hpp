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

#ifndef __COORD_POLICY_CARTESIAN_H_
#define __COORD_POLICY_CARTESIAN_H_

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "core/math.hpp"
#include "core/particles.h"
#include "framework/environment.h"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian {
 public:
  typedef typename Conf::value_t value_t;

  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return 1.0f;
  }

  HD_INLINE static value_t x1(value_t x) { return x; }
  HD_INLINE static value_t x2(value_t x) { return x; }
  HD_INLINE static value_t x3(value_t x) { return x; }

  HD_INLINE static void update_ptc(const Grid<Conf::dim, value_t>& grid,
                                   ptc_context<Conf::dim, value_t>& context,
                                   index_t<Conf::dim>& pos, value_t q_over_m,
                                   value_t dt) {
    q_over_m *= 0.5f;

    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      boris_pusher pusher;

      pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], q_over_m, dt);
    }

    move_ptc(grid, context, pos, dt);
  }

  HD_INLINE static void move_ptc(const Grid<Conf::dim, value_t>& grid,
                                 ptc_context<Conf::dim, value_t>& context,
                                 index_t<Conf::dim>& pos, value_t dt) {
#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.new_x[i] = context.x[i] + (context.p[i] * dt / context.gamma) *
                                            grid.inv_delta[i];
      context.dc[i] = std::floor(context.new_x[i]);
      pos[i] += context.dc[i];
      context.new_x[i] -= (value_t)context.dc[i];
    }
#pragma unroll
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = context.x[i] + context.p[i] * dt / context.gamma;
    }
  }

  // template <typename ExecPolicy>
  // static void fill_multiplicity(particles_t& ptc, rng_states_t& states,
  //                               int mult, value_t weight) {
  //   auto num = ptc.number();

  //   // ExecPolicy::launch(
  //   //     [num, mult, weight] LAMBDA(auto ptc, auto states) {
  //   //       auto& grid = ExecPolicy::grid();
  //   //       auto ext = grid.extent();
  //   //       rng_t rng(states);
  //   //       ExecPolicy::loop(
  //   //           [&grid, num, ext, mult, weight] LAMBDA(auto idx, auto& ptc,
  //   //                                                  auto& rng) {
  //   //             // auto idx = Conf::idx(n, ext);
  //   //             auto pos = get_pos(idx, ext);
  //   //             if (grid.is_in_bound(pos)) {
  //   //               for (int i = 0; i < mult; i++) {
  //   //                 uint32_t offset = num + idx.linear * mult * 2 + i * 2;

  //   //                 ptc.x1[offset] = ptc.x1[offset + 1] =
  //   //                     rng.template uniform<value_t>();
  //   //                 ptc.x2[offset] = ptc.x2[offset + 1] =
  //   //                     rng.template uniform<value_t>();
  //   //                 ptc.x3[offset] = ptc.x3[offset + 1] =
  //   //                     rng.template uniform<value_t>();
  //   //                 ptc.p1[offset] = ptc.p1[offset + 1] = 0.0;
  //   //                 ptc.p2[offset] = ptc.p2[offset + 1] = 0.0;
  //   //                 ptc.p3[offset] = ptc.p3[offset + 1] = 0.0;
  //   //                 ptc.E[offset] = ptc.E[offset + 1] = 1.0;
  //   //                 ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
  //   //                 ptc.weight[offset] = ptc.weight[offset + 1] = weight;
  //   //                 ptc.flag[offset] = set_ptc_type_flag(
  //   //                     flag_or(PtcFlag::primary), PtcType::electron);
  //   //                 ptc.flag[offset + 1] = set_ptc_type_flag(
  //   //                     flag_or(PtcFlag::primary), PtcType::positron);
  //   //               }
  //   //             }
  //   //           },
  //   //           // 0u, ext.size(), ptc, rng);
  //   //           Conf::begin(ext), Conf::end(ext), ptc, rng);
  //   //     },
  //   //     ptc, states);
  //   // ExecPolicy::sync();
  // }
};

}  // namespace Aperture

#endif  // __COORD_POLICY_CARTESIAN_H_
