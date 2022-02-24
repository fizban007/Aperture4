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

#ifndef _COORD_POLICY_CARTESIAN_IMPL_COOLING_H_
#define _COORD_POLICY_CARTESIAN_IMPL_COOLING_H_

#include "coord_policy_cartesian.hpp"
#include "data/multi_array_data.hpp"
#include "data/fields.h"
#include "data/scalar_data.hpp"
#include "framework/environment.h"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian_impl_cooling
    : public coord_policy_cartesian<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using vec3 = vec_t<value_t, 3>;
  using coord_policy_cartesian<Conf>::coord_policy_cartesian;

  void init() {
    value_t t_cool = 100.0f, sigma = 10.0f, Bg = 0.0f;
    sim_env().params().get_value("cooling", m_use_cooling);
    value_t sync_compactness = -1.0f;
    sim_env().params().get_value("sync_compactness", sync_compactness);
    sim_env().params().get_value("cooling_time", t_cool);
    if (sync_compactness < 0.0f) {
      sync_compactness = 1.0f / t_cool;
    }
    sim_env().params().get_value("sigma", sigma);
    sim_env().params().get_value("guide_field", Bg);
    if (Bg > 0.0f) {
      sigma = sigma + Bg*Bg*sigma;
    }
    // The cooling coefficient is effectively 2r_e\omega_p/3c in the dimensionless units
    if (!m_use_cooling) {
      m_cooling_coef = 0.0f;
    } else {
      m_cooling_coef = 2.0f * sync_compactness / sigma;
    }

    // If the config file specifies a synchrotron cooling coefficient, then we
    // use that instead. Sync cooling coefficient is roughly 2l_B/B^2
    if (sim_env().params().has("sync_cooling_coef")) {
      sim_env().params().get_value("sync_cooling_coef", m_cooling_coef);
    }

    auto sync_loss =
        sim_env().register_data<scalar_field<Conf>>(
            "sync_loss", this->m_grid, field_type::cell_centered, MemType::host_device);
    m_sync_loss = sync_loss->dev_ndptr();
    sync_loss->reset_after_output(true);
    auto sync_loss_total =
        sim_env().register_data<scalar_field<Conf>>(
            "sync_loss_total", this->m_grid, field_type::cell_centered, MemType::host_device);
    m_sync_loss_total = sync_loss_total->dev_ndptr();
    sync_loss_total->reset_after_output(true);
  }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            vec_t<UIntT, Conf::dim>& pos, value_t dt) const {
    value_t p1 = context.p[0];
    value_t p2 = context.p[1];
    value_t p3 = context.p[2];
    value_t gamma = context.gamma;
    auto flag = context.flag;

    default_pusher pusher;
    // Turn off synchrotron cooling for gamma < 1.001
    if (gamma <= 1.001f || check_flag(flag, PtcFlag::ignore_radiation) || m_cooling_coef == 0.0f) {
      pusher(context.p[0], context.p[1], context.p[2], context.gamma,
             context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
             decltype(context.q)(dt));
    } else {
      pusher(p1, p2, p3, gamma, context.E[0], context.E[1], context.E[2], context.B[0],
             context.B[1], context.B[2], dt * context.q / context.m * 0.5f,
             decltype(context.q)(dt));

      iterate(context.x, context.p, context.E, context.B, context.q / context.m,
              m_cooling_coef, dt);
      context.gamma = math::sqrt(1.0f + context.p.dot(context.p));

      auto idx = Conf::idx(pos, ext);
      value_t loss = context.weight * max(gamma - context.gamma, 0.0) / context.q;
      atomic_add(&m_sync_loss_total[idx], loss);
      if (!check_flag(context.flag, PtcFlag::exclude_from_spectrum)) {
        atomic_add(&m_sync_loss[idx], loss);
      }
    }

    move_ptc(grid, context, pos, dt);
  }

  HD_INLINE vec3 rhs_x(const vec3& u, value_t dt) const {
    value_t gamma = math::sqrt(1.0f + u.dot(u));
    return u * (dt / gamma);
  }

  HD_INLINE vec3 rhs_u(const vec3& E, const vec3& B, const vec3& u,
                       value_t e_over_m, value_t cooling_coef, value_t dt) const {
    vec3 result;
    value_t gamma = math::sqrt(1.0f + u.dot(u));
    vec3 Epbetaxb = E + cross(u, B) / gamma;

    result =
        e_over_m * Epbetaxb +
        cooling_coef *
            (cross(Epbetaxb, B) + E * u.dot(E) / gamma -
             u * (gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma))));
             // u * (-gamma * (Epbetaxb.dot(Epbetaxb) - square(u.dot(E) / gamma)));

    return result * dt;
  }

  HD_INLINE void iterate(vec3& x, vec3& u, const vec3& E, const vec3& B,
                         double e_over_m, double cooling_coef, double dt) const {
    // vec3 x0 = x, x1 = x;
    vec3 u0 = u, u1 = u;

    for (int i = 0; i < 4; i++) {
      // x1 = x0 + rhs_x((u0 + u) * 0.5, dt);
      u1 = u0 + rhs_u(E, B, (u0 + u) * 0.5, e_over_m, cooling_coef, dt);
      // x = x1;
      u = u1;
    }
  }

 private:
  bool m_use_cooling = false;
  value_t m_cooling_coef = 0.0f;
  mutable ndptr<value_t, Conf::dim> m_sync_loss;
  mutable ndptr<value_t, Conf::dim> m_sync_loss_total;
};

}  // namespace Aperture

#endif  // _COORD_POLICY_CARTESIAN_IMPL_COOLING_H_
