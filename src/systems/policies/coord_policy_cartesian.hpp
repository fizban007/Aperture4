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
#include "core/particles.h"
#include "data/data_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"
#include "systems/physics/pusher_simple.hpp"

namespace Aperture {

template <typename Conf, template <class> class Pusher = pusher_simple>
class coord_policy_cartesian_base {
 public:
  // typedef typename Conf::value_t value_t;
  using value_t = typename Conf::value_t;
  using grid_type = grid_t<Conf>;

  coord_policy_cartesian_base(const grid_t<Conf>& grid) :
      m_grid(grid), m_pusher(grid) {}
  ~coord_policy_cartesian_base() = default;

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return 1.0f;
  }

  HD_INLINE static value_t x1(value_t x) { return x; }
  HD_INLINE static value_t x2(value_t x) { return x; }
  HD_INLINE static value_t x3(value_t x) { return x; }

  void init() {
    m_pusher.init();
  }

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
      // default_pusher pusher;
      typename Conf::pusher_t pusher;

      // pusher(context.p[0], context.p[1], context.p[2], context.gamma,
      //        context.E[0], context.E[1], context.E[2], context.B[0],
      //        context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
      //        decltype(context.q)(dt));
      m_pusher.push(grid, ext, context, pos, dt);
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
                     scalar_field<Conf>& rho_total, value_t dt, bool process_rho) const {
    ExecPolicy::launch(
        [dt] LAMBDA(auto j) {
          auto& grid = ExecPolicy::grid();
          auto ext = grid.extent();
          // grid.cell_size() is simply the product of all deltas
          ExecPolicy::loop(Conf::begin(ext), Conf::end(ext),
                           [&] LAMBDA(auto idx) {
#pragma unroll
                             for (int i = 0; i < Conf::dim; i++) {
                               j[i][idx] *= grid.delta[i];
                             }
                           });
          // j);
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

 protected:
  const grid_t<Conf>& m_grid;
  Pusher<Conf> m_pusher;
};

template <typename Conf>
using coord_policy_cartesian = coord_policy_cartesian_base<Conf, pusher_simple>;

}  // namespace Aperture
