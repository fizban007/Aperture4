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

#ifndef __COORD_POLICY_SPHERICAL_H_
#define __COORD_POLICY_SPHERICAL_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
// #include "core/multi_array_exp.hpp"
#include "core/particles.h"
#include "data/data_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "systems/grid_sph.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_spherical {
 public:
  typedef typename Conf::value_t value_t;

  coord_policy_spherical(const grid_t<Conf>& grid)
      : m_grid(dynamic_cast<const grid_sph_t<Conf>&>(grid)) {}
  ~coord_policy_spherical() = default;

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return math::sin(grid_sph_t<Conf>::theta(x2));
  }

  HD_INLINE static value_t x1(value_t x) { return grid_sph_t<Conf>::radius(x); }
  HD_INLINE static value_t x2(value_t x) { return grid_sph_t<Conf>::theta(x); }
  HD_INLINE static value_t x3(value_t x) { return x; }

  // Inline functions to be called in the particle update loop
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            ptc_context<Conf::dim, value_t>& context,
                            index_t<Conf::dim>& pos, value_t q_over_m,
                            value_t dt) const {
    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      default_pusher pusher;

      pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], q_over_m * 0.5f, dt);
    }

    move_ptc(grid, context, pos, dt);
  }

  // Abstracted moving routine that is shared by both ptc and ph
  template <typename PtcContext>
  HD_INLINE void move_ptc(const Grid<Conf::dim, value_t>& grid,
                          PtcContext& context, index_t<Conf::dim>& pos,
                          value_t dt) const {
    // Global position in sph coord
    vec_t<value_t, 3> x_global_old(grid.template pos<0>(pos[0], context.x[0]),
                                   grid.template pos<1>(pos[1], context.x[1]),
                                   grid.template pos<2>(pos[2], context.x[2]));
    vec_t<value_t, 3> x_global_sph(x1(x_global_old[0]), x2(x_global_old[1]),
                                   x3(x_global_old[2]));

    // Global position in cartesian coord
    vec_t<value_t, 3> x_global_cart;
    value_t sth = math::sin(x_global_sph[1]);
    x_global_cart[0] = x_global_sph[0] * sth * math::cos(x_global_sph[2]);
    x_global_cart[1] = x_global_sph[0] * sth * math::sin(x_global_sph[2]);
    x_global_cart[2] = x_global_sph[0] * math::cos(x_global_sph[1]);

    // Transform momentum vector to cartesian
    sph2cart(context.p, x_global_sph);

    // Move in Cartesian coordinates
    x_global_cart += context.p * (dt / context.gamma);

    // Compute the new spherical location
    vec_t<value_t, 3> x_global_sph_new;
    x_global_sph_new[0] = math::sqrt(x_global_cart.dot(x_global_cart));
    x_global_sph_new[2] = math::atan2(x_global_cart[1], x_global_cart[0]);
    x_global_sph_new[1] = math::acos(x_global_cart[2] / x_global_sph_new[0]);

    // Transform the momentum vector to spherical at the new location
    cart2sph(context.p, x_global_sph_new);
    x_global_sph_new[0] = grid_sph_t<Conf>::from_radius(x_global_sph_new[0]);
    x_global_sph_new[1] = grid_sph_t<Conf>::from_theta(x_global_sph_new[1]);

#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.new_x[i] =
          context.x[i] +
          (x_global_sph_new[i] - x_global_old[i]) * grid.inv_delta[i];
      context.dc[i] = std::floor(context.new_x[i]);
      pos[i] += context.dc[i];
      context.new_x[i] -= (value_t)context.dc[i];
    }
#pragma unroll
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = context.x[i] + context.p[i] * (dt / context.gamma);
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
  void process_J_Rho(vector_field<Conf>& J,
                     data_array<scalar_field<Conf>>& Rho) const {}

  template <typename ExecPolicy>
  void filter_field(vector_field<Conf>& field,
                    typename Conf::multi_array_t& tmp,
                    const vec_t<bool, Conf::dim * 2>& is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ae[0],
                                       is_boundary);
    filter_field_component<ExecPolicy>(field.at(1), tmp, m_grid.m_Ae[1],
                                       is_boundary);
    filter_field_component<ExecPolicy>(field.at(2), tmp, m_grid.m_Ae[2],
                                       is_boundary);
  }

  template <typename ExecPolicy>
  void filter_field(scalar_field<Conf>& field,
                    typename Conf::multi_array_t& tmp,
                    const vec_t<bool, Conf::dim * 2>& is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_dV,
                                       is_boundary);
  }

 private:
  const grid_sph_t<Conf>& m_grid;
};

}  // namespace Aperture

#endif  // __COORD_POLICY_SPHERICAL_H_
