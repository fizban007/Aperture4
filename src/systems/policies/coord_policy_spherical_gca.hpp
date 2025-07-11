/*
 * Copyright (c) 2025 Alex Chen.
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

#include "coord_policy_spherical.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"

namespace Aperture {

// Currently we only support E x B drift in spherical coordinates. The rationale
// is that this is mainly used for magnetar simulations, where the gyro motion
// is not important, and the E x B drift is the main drifting motion of the particles.
template <typename Conf>
class coord_policy_spherical_gca : public coord_policy_spherical<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using coord_policy_spherical<Conf>::coord_policy_spherical;

  void init() {
    nonown_ptr<vector_field<Conf>> E;
    sim_env().get_data("E", E);
    nonown_ptr<vector_field<Conf>> B;
    sim_env().get_data("B", B);
#ifdef GPU_ENABLED
    m_E[0] = E->at(0).dev_ndptr_const();
    m_E[1] = E->at(1).dev_ndptr_const();
    m_E[2] = E->at(2).dev_ndptr_const();
    m_B[0] = B->at(0).dev_ndptr_const();
    m_B[1] = B->at(1).dev_ndptr_const();
    m_B[2] = B->at(2).dev_ndptr_const();
#else
    m_E[0] = E->at(0).host_ndptr_const();
    m_E[1] = E->at(1).host_ndptr_const();
    m_E[2] = E->at(2).host_ndptr_const();
    m_B[0] = B->at(0).host_ndptr_const();
    m_B[1] = B->at(1).host_ndptr_const();
    m_B[2] = B->at(2).host_ndptr_const();
#endif
  }

  HD_INLINE static vec_t<value_t, 3> f_v_E(const vec_t<value_t, 3>& E,
                                           const vec_t<value_t, 3>& B) {
    value_t EB_sqr = E.dot(E) + B.dot(B);
    // w_E[0] = (E[1] * B[2] - E[2] * B[1]) / EB_sqr;
    // w_E[1] = (E[2] * B[0] - E[0] * B[2]) / EB_sqr;
    // w_E[2] = (E[0] * B[1] - E[1] * B[0]) / EB_sqr;
    vec_t<value_t, 3> w_E = cross(E, B) / EB_sqr;
    value_t w2 = w_E.dot(w_E);
    if (w2 < TINY) {
      // if w2 is too small, essentially ExB is zero, we just return the zero vector
      return w_E;
    } else {
      w_E *= (1.0f - math::sqrt(max(1.0f - 4.0f * w2, 0.0f))) * 0.5f / w2;
      return w_E;
    }
  }

  HD_INLINE static value_t f_Gamma(value_t u_par, value_t mu, value_t B_mag,
                                   const vec_t<value_t, 3>& vE) {
    auto k = 1.0f / math::sqrt(max(1.0f - vE.dot(vE), TINY));
    // Note that here mu is defined as specific mu, or mu divided by mass of the
    // particle
    return k * math::sqrt(1.0f + u_par * u_par + 2.0f * mu * k * B_mag);
  }

  template <typename PtcContext>
  HOST_DEVICE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                              const extent_t<Conf::dim>& ext,
                              PtcContext& context, index_t<Conf::dim>& pos,
                              value_t dt) const {
    if (check_flag(context.flag, PtcFlag::ignore_EM)) {
      return;
    }
    // In this GCA pusher, p[0] is taken to be u_par, p[1] is taken to be the
    // magnetic moment mu.
    value_t u_par = context.p[0];
    value_t mu = context.p[1];

    // printf("u_par is %f, mu is %f\n", u_par, mu);

    // Compute the E x B drift velocity
    vec_t<value_t, 3> vE = f_v_E(context.E, context.B);
    // printf("vE is (%f, %f, %f)\n", vE[0], vE[1], vE[2]);

    // Compute B magnitude and the unit vector b
    value_t B_mag = math::sqrt(context.B.dot(context.B));
    vec_t<value_t, 3> b = context.B / B_mag;

    value_t E_par = context.E.dot(b);
    // printf("E_par is %f, B_mag is %f\n", E_par, B_mag);

    // Update the parallel momentum using parallel electric field
    value_t u_par_new = u_par + context.q * dt * E_par / context.m;
    // Compute the new Lorentz factor, since it depends on the new u_par
    value_t Gamma_new = f_Gamma(u_par_new, mu, B_mag, vE);
    // Need to update this because we are going to use this in the iteration
    context.gamma = Gamma_new;

    // printf("u_par_new is %f, Gamma_new is %f, E_par is %f\n", u_par_new,
    // Gamma_new, E_par);

    auto x_global = grid.coord_global(pos, context.x);
    auto x_iter = x_global;
    auto pos_iter = pos;
    auto interp = interp_t<1, Conf::dim>{};
    vec_t<value_t, 3> E_iter = context.E, B_iter = context.B;
    vec_t<value_t, 3> vE_iter = f_v_E(E_iter, B_iter);
    value_t B_mag_iter = math::sqrt(B_iter.dot(B_iter));
    B_iter /= B_mag_iter;
    context.new_x = context.x;

    // Iterate several times to get the updated position
    constexpr int n_iter = 5;
    for (int i = 0; i < n_iter; i++) {
      // TODO: dx is not correct. Need to transform to Cartesian, evaluate, then transform back?
      move_ptc_gca(grid, x_iter, b, B_iter, vE, vE_iter,
                 Gamma_new, context.gamma, u_par_new, dt);
      // printf("x_iter is (%f, %f, %f)\n", x_iter[0], x_iter[1], x_iter[2]);

      grid.from_global(x_iter, pos_iter, context.new_x);
      auto idx_iter = Conf::idx(pos_iter, ext);

      // Interpolate the E and B field at the new position
      E_iter[0] =
          interp(context.new_x, m_E[0], idx_iter, ext, stagger_t(0b110));
      E_iter[1] =
          interp(context.new_x, m_E[1], idx_iter, ext, stagger_t(0b101));
      E_iter[2] =
          interp(context.new_x, m_E[2], idx_iter, ext, stagger_t(0b011));
      B_iter[0] =
          interp(context.new_x, m_B[0], idx_iter, ext, stagger_t(0b001));
      B_iter[1] =
          interp(context.new_x, m_B[1], idx_iter, ext, stagger_t(0b010));
      B_iter[2] =
          interp(context.new_x, m_B[2], idx_iter, ext, stagger_t(0b100));
      vE_iter = f_v_E(E_iter, B_iter);
      B_mag_iter = math::sqrt(B_iter.dot(B_iter));
      B_iter /= B_mag_iter;
      context.gamma = f_Gamma(u_par_new, mu, B_mag_iter, vE_iter);
    }

#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.dc[i] = pos_iter[i] - pos[i];
    }
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = x_iter[i];
    }
    // printf("old_x (%f, %f, %f), new_x (%f, %f, %f), gamma is %f\n",
    //        context.x[0], context.x[1], context.x[2], context.new_x[0],
    //        context.new_x[1], context.new_x[2], context.gamma);
    pos = pos_iter;
    context.p[0] = u_par_new;
    context.p[1] = mu;
  }

  HD_INLINE void move_ptc_gca(const Grid<Conf::dim, value_t>& grid,
                              vec_t<value_t, 3>& x_global,
                              vec_t<value_t, 3> b,
                              vec_t<value_t, 3> b_iter,
                              vec_t<value_t, 3> vE,
                              vec_t<value_t, 3> vE_iter,
                              value_t gamma, value_t gamma_iter,
                              value_t u_par, value_t dt) const {
    using grid_type = grid_sph_t<Conf>;
    // Transform the momentum vector to cartesian at the current location
    vec_t<value_t, 3> x_global_cart = grid_type::coord_to_cart(x_global);

    // Transform the b vector to cartesian at the current location
    grid_type::vec_to_cart(b, x_global);
    grid_type::vec_to_cart(b_iter, x_global);
    grid_type::vec_to_cart(vE, x_global);
    grid_type::vec_to_cart(vE_iter, x_global);

    // Move in Cartesian coordinates
    x_global_cart += 0.5f * dt * u_par * (b / gamma + b_iter / gamma_iter) +
                     0.5f * dt * (vE + vE_iter);

    // Compute the new spherical location
    x_global = grid_type::coord_from_cart(x_global_cart);
  }

 private:
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
};

}  // namespace Aperture
