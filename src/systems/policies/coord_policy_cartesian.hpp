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
#include "data/data_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian {
 public:
  typedef typename Conf::value_t value_t;

  coord_policy_cartesian(const grid_t<Conf>& grid) : m_grid(grid) {}
  ~coord_policy_cartesian() = default;

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return 1.0f;
  }

  HD_INLINE static value_t x1(value_t x) { return x; }
  HD_INLINE static value_t x2(value_t x) { return x; }
  HD_INLINE static value_t x3(value_t x) { return x; }

  void init() {}

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            vec_t<UIntT, Conf::dim>& pos,
                            // FloatT q_over_m,
                            value_t dt) const {
#ifndef USE_SIMD
    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
#endif
      default_pusher pusher;

      pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
             decltype(context.q)(dt));
#ifndef USE_SIMD
    }
#endif

    move_ptc(grid, context, pos, dt);
  }

  // Abstracted moving routine that is shared by both ptc and ph
  template <typename PtcContext, typename UIntT>
  HD_INLINE void move_ptc(const Grid<Conf::dim, value_t>& grid,
                          PtcContext& context, vec_t<UIntT, Conf::dim>& pos,
                          value_t dt) const {
#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.new_x[i] = context.x[i] + (context.p[i] * dt / context.gamma) *
                                            grid.inv_delta[i];
#ifdef USE_SIMD
#ifdef USE_DOUBLE
      context.dc[i] = round_to_int(floor(context.new_x[i]));
#else
      context.dc[i] = roundi(floor(context.new_x[i]));
#endif
#else
      context.dc[i] = floor(context.new_x[i]);
#endif
      pos[i] += context.dc[i];
#ifdef USE_DOUBLE
      context.new_x[i] -= to_double(context.dc[i]);
#else
      context.new_x[i] -= to_float(context.dc[i]);
#endif
    }
#pragma unroll
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = context.x[i] + context.p[i] * dt / context.gamma;
    }
  }

  // Inline functions to be called in the photon update loop
  HD_INLINE void update_ph(const Grid<Conf::dim, value_t>& grid,
                           ph_context<Conf::dim, value_t>& context,
                           index_t<Conf::dim>& pos, value_t dt) const {
    move_ptc(grid, context, pos, dt);
  }

  // Extra processing routines
  template <typename ExecPolicy>
  void process_J_Rho(vector_field<Conf>& J, data_array<scalar_field<Conf>>& Rho,
                     value_t dt, bool process_rho) const {
    ExecPolicy::launch(
        [dt] LAMBDA(auto j) {
          auto& grid = ExecPolicy::grid();
          auto ext = grid.extent();
          // grid.cell_size() is simply the product of all deltas
          ExecPolicy::loop(
              Conf::begin(ext), Conf::end(ext),
              [&grid, &ext] LAMBDA(auto idx, auto& j) {
                auto pos = get_pos(idx, ext);
#pragma unroll
                for (int i = 0; i < Conf::dim; i++) {
                  j[i][idx] *= grid.delta[i];
                }
              },
              j);
        },
        J);
    ExecPolicy::sync();
  }

  template <typename ExecPolicy, int N>
  void filter_field(field_t<N, Conf>& field, typename Conf::multi_array_t& tmp,
                    const vec_t<bool, Conf::dim * 2>& is_boundary) const {
    for (int i = 0; i < N; i++) {
      filter_field_component<ExecPolicy>(field.at(i), tmp, is_boundary);
    }
  }

 private:
  const grid_t<Conf>& m_grid;
};

}  // namespace Aperture

#endif  // __COORD_POLICY_CARTESIAN_H_
