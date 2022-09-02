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

#ifndef _COORD_POLICY_CARTESIAN_GCA_LITE_H_
#define _COORD_POLICY_CARTESIAN_GCA_LITE_H_

#include "coord_policy_cartesian.hpp"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian_gca_lite : public coord_policy_cartesian<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using coord_policy_cartesian<Conf>::coord_policy_cartesian;

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
    vec_t<value_t, 3> w_E;
    value_t EB_sqr = E.dot(E) + B.dot(B);
    // w_E[0] = (E[1] * B[2] - E[2] * B[1]) / EB_sqr;
    // w_E[1] = (E[2] * B[0] - E[0] * B[2]) / EB_sqr;
    // w_E[2] = (E[0] * B[1] - E[1] * B[0]) / EB_sqr;
    w_E = cross(E, B) / EB_sqr;
    value_t w2 = w_E.dot(w_E);
    if (w2 < TINY) {
      return w_E;
    } else {
      w_E *= (1.0f - math::sqrt(1.0f - 4.0f * w2)) * 0.5f / w2;
      return w_E;
    }
  }

  HD_INLINE static value_t f_kappa(const vec_t<value_t, 3>& E,
                                   const vec_t<value_t, 3>& B) {
    auto vE = f_v_E(E, B);
    return 1.0f / math::sqrt(1.0f - vE.dot(vE));
  }

  HD_INLINE static value_t f_Gamma(value_t u_par, value_t mu,
                                   const vec_t<value_t, 3>& E,
                                   const vec_t<value_t, 3>& B) {
    auto k = f_kappa(E, B);
    auto B_mag = math::sqrt(B.dot(B));
    // Note that here mu is defined as specific mu, or mu divided by mass of the
    // particle
    return k * math::sqrt(1.0f + (u_par * u_par + 2.0f * mu * k * B_mag));
  }

  // This computes (\mathbf{v} \cdot\div)\mathbf{b}, the change of the
  // unit vector b along the direction of v
  template <typename FieldType>
  HOST_DEVICE static vec_t<value_t, 3> vec_div_b(
      const vec_t<value_t, 3>& v, const FieldType& B, vec_t<value_t, 3> rel_x,
      index_t<Conf::dim> pos, const Grid<Conf::dim, value_t>& grid,
      const vec_t<uint32_t, Conf::dim>& ext, value_t dt) {
    constexpr value_t h = 0.05f;
    vec_t<value_t, 3> vb, result;

    auto dv = v * (h * dt) / math::sqrt(v.dot(v));
    auto x_global = grid.pos_global(pos, rel_x);
    grid.from_global(x_global + dv, pos, rel_x);

    auto idx = Conf::idx(pos, ext);
    auto interp = interp_t<Conf::interp_order, Conf::dim>{};

    vb[0] = interp(rel_x, B[0], idx, ext, stagger_t(0b001));
    vb[1] = interp(rel_x, B[1], idx, ext, stagger_t(0b010));
    vb[2] = interp(rel_x, B[2], idx, ext, stagger_t(0b100));
    result = vb / math::sqrt(vb.dot(vb));

    grid.from_global(x_global - dv, pos, rel_x);
    auto idx2 = Conf::idx(pos, ext);

    vb[0] = interp(rel_x, B[0], idx2, ext, stagger_t(0b001));
    vb[1] = interp(rel_x, B[1], idx2, ext, stagger_t(0b010));
    vb[2] = interp(rel_x, B[2], idx2, ext, stagger_t(0b100));
    result -= vb / math::sqrt(vb.dot(vb));

    return result / (2.0f * h * dt);
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

    vec_t<value_t, 3> vE = f_v_E(context.E, context.B);
    // printf("vE is (%f, %f, %f)\n", vE[0], vE[1], vE[2]);

    value_t B_mag = math::sqrt(context.B.dot(context.B));
    vec_t<value_t, 3> b = context.B / B_mag;
    value_t E_par = context.E.dot(b);

    // printf("E_par is %f, B_mag is %f\n", E_par, B_mag);

    value_t u_par_new = u_par + context.q * dt * E_par / context.m;
    value_t Gamma_new = f_Gamma(u_par_new, mu, context.E, context.B);
    // Need to update this because we are going to use this in the iteration
    context.gamma = Gamma_new;

    // printf("u_par_new is %f, Gamma_new is %f\n", u_par_new, Gamma_new);

    auto x_global = grid.pos_global(pos, context.x);
    auto x_iter = x_global;
    auto pos_iter = pos;
    auto interp = interp_t<1, Conf::dim>{};
    vec_t<value_t, 3> E_iter = context.E, B_iter = context.B;
    context.new_x = context.x;

    // Iterate several times to get the updated position
    constexpr int n_iter = 4;
    for (int i = 0; i < n_iter; i++) {
      value_t B_mag_iter = math::sqrt(B_iter.dot(B_iter));

      vec_t<value_t, 3> dx = 0.5f * dt * u_par_new *
                                 (context.B / (Gamma_new * B_mag) +
                                  B_iter / (context.gamma * B_mag_iter)) +
                             0.5f * dt * (vE + f_v_E(E_iter, B_iter));
      x_iter = x_global + dx;
      // printf("x_iter is (%f, %f, %f)\n", x_iter[0], x_iter[1], x_iter[2]);

      grid.from_global(x_iter, pos_iter, context.new_x);

      auto idx_iter = Conf::idx(pos_iter, ext);

      // Interpolate the E and B field at the new position
      E_iter[0] = interp(context.new_x, m_E[0], idx_iter, ext, stagger_t(0b110));
      E_iter[1] = interp(context.new_x, m_E[1], idx_iter, ext, stagger_t(0b101));
      E_iter[2] = interp(context.new_x, m_E[2], idx_iter, ext, stagger_t(0b011));
      B_iter[0] = interp(context.new_x, m_B[0], idx_iter, ext, stagger_t(0b001));
      B_iter[1] = interp(context.new_x, m_B[1], idx_iter, ext, stagger_t(0b010));
      B_iter[2] = interp(context.new_x, m_B[2], idx_iter, ext, stagger_t(0b100));
      context.gamma = f_Gamma(u_par_new, mu, E_iter, B_iter);
    }

#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.dc[i] = pos_iter[i] - pos[i];
    }
    for (int i = Conf::dim; i < 3; i++) {
      context.new_x[i] = x_iter[i];
    }
    // printf("old_x (%f, %f, %f), new_x (%f, %f, %f), dc (%d, %d, %d)\n",
    //        context.x[0], context.x[1], context.x[2], context.new_x[0],
    //        context.new_x[1], context.new_x[2], context.dc[0], context.dc[1], context.dc[2]);
    pos = pos_iter;
    context.p[0] = u_par_new;
    context.p[1] = mu;
  }

 private:
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
};

}  // namespace Aperture

#endif  // _COORD_POLICY_CARTESIAN_GCA_LITE_H_
