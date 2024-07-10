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
#include "core/math.hpp"
// #include "core/multi_array_exp.hpp"
#include "core/particles.h"
#include "data/data_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "systems/grid_polar.hpp"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"
#include "systems/physics/pusher_simple.hpp"

namespace Aperture {

template <typename Conf, template <class> class Pusher = pusher_simple>
class coord_policy_polar_base {
 public:
  using value_t = typename Conf::value_t;
  using grid_type = grid_polar_t<Conf>;

  coord_policy_polar_base(const grid_t<Conf>& grid)
      : m_grid(dynamic_cast<const grid_type&>(grid)), m_pusher(grid) {}
  ~coord_policy_polar_base() = default;

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return 1.0;
  }

  HD_INLINE static value_t x1(value_t x) { return grid_type::radius(x); }
  HD_INLINE static value_t x2(value_t x) { return grid_type::theta(x); }
  HD_INLINE static value_t x3(value_t x) { return x; }

  void init() {
    m_pusher.init();
  }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            index_t<Conf::dim>& pos, value_t dt) const {
    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      // default_pusher pusher;
      typename Conf::pusher_t pusher;

      pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], dt * context.q / context.m * 0.5f, dt);
    }

    move_ptc(grid, context, pos, dt);
  }

  // Abstracted moving routine that is shared by both ptc and ph
  template <typename PtcContext>
  HD_INLINE void move_ptc(const Grid<Conf::dim, value_t>& grid,
                          PtcContext& context, index_t<Conf::dim>& pos,
                          value_t dt) const {
    // Global position in polar coord
    vec_t<value_t, 3> x_global_old(
        grid.template coord<0>(pos[0], context.x[0]),
        grid.template coord<1>(pos[1], context.x[1]),
        grid.template coord<2>(pos[2], context.x[2]));
    vec_t<value_t, 3> x_global_pol(x1(x_global_old[0]), x2(x_global_old[1]),
                                   x3(x_global_old[2]));

    // Global position in cartesian coord
    vec_t<value_t, 3> x_global_cart;
    value_t sth = math::sin(x_global_pol[1]);
    value_t cth = math::cos(x_global_pol[1]);
    x_global_cart[0] = x_global_pol[0] * cth;
    x_global_cart[1] = x_global_pol[0] * sth;
    x_global_cart[2] = x_global_pol[2];

    // Transform momentum vector to cartesian
    polar2cart(context.p, x_global_pol);

    // Move in Cartesian coordinates
    x_global_cart += context.p * (dt / context.gamma);

    // Compute the new polar coord location
    vec_t<value_t, 3> x_global_pol_new;
    x_global_pol_new[2] = x_global_cart[2];
    x_global_pol_new[0] =
        math::sqrt(square(x_global_cart[0]) + square(x_global_cart[1]));
    x_global_pol_new[1] = math::atan2(x_global_cart[1], x_global_cart[0]);

    // Transform the momentum vector to polar coorpolar coord at the new
    // location
    cart2polar(context.p, x_global_pol_new);
    x_global_pol_new[0] = grid_type::from_radius(x_global_pol_new[0]);
    x_global_pol_new[1] = grid_type::from_theta(x_global_pol_new[1]);

#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.new_x[i] =
          context.x[i] +
          (x_global_pol_new[i] - x_global_old[i]) * grid.inv_delta[i];
      context.dc[i] = std::floor(context.new_x[i]);
      pos[i] += context.dc[i];
      context.new_x[i] -= (value_t)context.dc[i];
    }
#pragma unroll
    for (int i = Conf::dim; i < 3; i++) {
      // context.new_x[i] = context.x[i] + context.p[i] * (dt / context.gamma);
      context.new_x[i] = x_global_pol_new[i];
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
                     scalar_field<Conf>& rho_total,
                     value_t dt, bool process_rho) const {
    auto num_species = Rho.size();
    ExecPolicy::launch(
        [dt, num_species, process_rho] LAMBDA(auto j, auto rho, auto rho_total,
                                              auto grid_ptrs) {
          auto& grid = ExecPolicy::grid();
          auto ext = grid.extent();
          // grid.cell_size() is simply the product of all deltas
          auto w = grid.cell_size();
          ExecPolicy::loop(Conf::begin(ext), Conf::end(ext),
                           [&] LAMBDA(auto idx) {
                             auto pos = get_pos(idx, ext);

                             j[0][idx] *= w / grid_ptrs.Ae[0][idx];
                             j[1][idx] *= w / grid_ptrs.Ae[1][idx];

                             // Would be nice to turn this into constexpr, but
                             // this will do as well if (Conf::dim == 2)
                             //   j[2][idx] /= grid_ptrs.dV[idx];
                             // else if (Conf::dim == 3)
                             typename Conf::value_t theta =
                                 grid.template coord<1>(pos[1], true);
                             j[2][idx] *= w / grid_ptrs.Ae[2][idx];
                             rho_total[idx] /= grid_ptrs.dV[idx];
                             for (int n = 0; n < num_species; n++) {
                               rho[n][idx] /= grid_ptrs.dV[idx];
                             }
                           });
          // j, rho, grid_ptrs);
        },
        J, Rho, rho_total, m_grid.get_grid_ptrs());
    ExecPolicy::sync();
  }

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
  const grid_type& m_grid;
  Pusher<Conf> m_pusher;
};

template <typename Conf>
using coord_policy_polar = coord_policy_polar_base<Conf, pusher_simple>;

}  // namespace Aperture
