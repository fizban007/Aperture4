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
#include "framework/environment.h"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian_impl_cooling
    : public coord_policy_cartesian<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using vec3 = vec_t<value_t, 3>;
  using coord_policy_cartesian<Conf>::coord_policy_cartesian;

  void init() { sim_env().params().get_value("cooling_re", m_re); }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext, typename UIntT>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                            const extent_t<Conf::dim>& ext, PtcContext& context,
                            vec_t<UIntT, Conf::dim>& pos, value_t dt) const {
    iterate(context.x, context.p, context.E, context.B, context.q / context.m,
            2.0f * m_re / 3.0f, dt);
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

    return result * dt;
  }

  HOST_DEVICE void iterate(vec3& x, vec3& u, const vec3& E, const vec3& B,
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
  value_t m_re = 1.0f;
};

}  // namespace Aperture

#endif  // _COORD_POLICY_CARTESIAN_IMPL_COOLING_H_
