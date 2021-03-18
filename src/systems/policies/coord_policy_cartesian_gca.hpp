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

#ifndef _COORD_POLICY_CARTESIAN_GCA_H_
#define _COORD_POLICY_CARTESIAN_GCA_H_

#include "coord_policy_cartesian.hpp"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename Conf>
class coord_policy_cartesian_gca : public coord_policy_cartesian<Conf> {
 public:
  using value_t = typename Conf::value_t;
  using coord_policy_cartesian<Conf>::coord_policy_cartesian;

  void init() {
    nonown_ptr<vector_field<Conf>> E;
    sim_env().get_data("E", E);
    nonown_ptr<vector_field<Conf>> B;
    sim_env().get_data("B", B);
#ifdef CUDA_ENABLED
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

  HD_INLINE vec_t<value_t, 3> v_E(const vec_t<value_t, 3>& E,
                                  const vec_t<value_t, 3>& B) const {
    vec_t<value_t, 3> w_E;
    value_t EB_sqr = E.dot(E) + B.dot(B);
    // w_E[0] = (E[1] * B[2] - E[2] * B[1]) / EB_sqr;
    // w_E[1] = (E[2] * B[0] - E[0] * B[2]) / EB_sqr;
    // w_E[2] = (E[0] * B[1] - E[1] * B[0]) / EB_sqr;
    w_E = cross(E, B) / EB_sqr;
    value_t w2 = w_E.dot(w_E);
    w_E *= (1.0f - math::sqrt(1.0f - 4.0f * w2)) * 0.5f / w2;
    return w_E;
  }

  HD_INLINE value_t kappa(const vec_t<value_t, 3>& E,
                          const vec_t<value_t, 3>& B) const {
    auto vE = v_E(E, B);
    return 1.0f / math::sqrt(1.0f - vE.dot(vE));
  }

  HD_INLINE value_t Gamma(value_t u_par, value_t mu, const vec_t<value_t, 3>& E,
                          const vec_t<value_t, 3>& B) const {
    auto k = kappa(E, B);
    auto B_mag = math::sqrt(B.dot(B));
    // Note that here mu is defined as specific mu, or mu divided by mass of the
    // particle
    return k * math::sqrt(1.0f + (u_par * u_par + 2.0f * mu * k * B_mag));
  }

  // This computes (\mathbf{v} \cdot\div)\mathbf{b}, the change of the
  // unit vector b along the direction of v
  template <typename FieldType>
  HOST_DEVICE vec_t<value_t, 3> vec_div_b(const vec_t<value_t, 3>& v,
                                          const FieldType& B,
                                          const vec_t<value_t, 3>& x_global,
                                          const Grid<Conf::dim, value_t>& grid,
                                          const vec_t<uint32_t, 3>& ext,
                                          value_t dt) const {
    auto dv = v * (0.05f * dt) / math::sqrt(v.dot(v));
    index_t<Conf::dim> pos;
    vec_t<value_t, 3> rel_x, vb, result;
    grid.from_global(x_global + dv, pos, rel_x);
    auto idx = Conf::idx(pos, ext);

    auto interp = interp_t<Conf::interp_order, Conf::dim>{};

    vb[0] = interp(rel_x, B[0], idx, ext, stagger_t(0b001));
    vb[1] = interp(rel_x, B[1], idx, ext, stagger_t(0b010));
    vb[2] = interp(rel_x, B[2], idx, ext, stagger_t(0b100));
    result = vb / math::sqrt(vb.dot(vb));

    grid.from_global(x_global - dv, pos, rel_x);
    idx = Conf::idx(pos, ext);

    vb[0] = interp(rel_x, B[0], idx, ext, stagger_t(0b001));
    vb[1] = interp(rel_x, B[1], idx, ext, stagger_t(0b010));
    vb[2] = interp(rel_x, B[2], idx, ext, stagger_t(0b100));
    result -= vb / math::sqrt(vb.dot(vb));

    return result / (0.1f * dt);
  }

  HOST_DEVICE vec_t<value_t, 3> v_c(value_t u_par, value_t mu,
                                    const vec_t<value_t, 3>& E,
                                    const vec_t<value_t, 3>& B,
                                    const vec_t<value_t, 3>& x_global,
                                    const Grid<Conf::dim, value_t>& grid,
                                    const extent_t<Conf::dim>& ext, value_t dt,
                                    value_t q_over_m) const {
    vec_t<value_t, 3> result;
    auto B_mag = math::sqrt(B.dot(B));
    auto b = B / B_mag;
    auto vE = v_E(E, B);
    auto k = 1.0f / math::sqrt(1.0f - vE.dot(vE));
    auto g = k * math::sqrt(1.0f + (u_par * u_par + 2.0f * mu * k * B_mag));

    vec_t<value_t, 3> tmp =
        vec_div_b(b, m_B, x_global, grid, ext, dt) * (u_par * u_par / g);

    tmp += vec_div_b(vE, m_B, x_global, grid, ext, dt) * u_par;

    return cross(b, tmp) * (k * k / (B_mag * q_over_m));
  }

  HD_INLINE vec_t<value_t, 3> v_c(value_t u_par, value_t mu, value_t gamma,
                                  value_t k, value_t B_mag, value_t q_over_m,
                                  const vec_t<value_t, 3>& b,
                                  const vec_t<value_t, 3>& b_dot_div_b,
                                  const vec_t<value_t, 3>& vE_dot_div_b) const {
    return cross(b,
                 b_dot_div_b * (u_par * u_par / gamma) + vE_dot_div_b * u_par) *
           (k * k / (B_mag * q_over_m));
  }

  template <typename PtcContext>
  HOST_DEVICE void update_ptc(const Grid<Conf::dim, value_t>& grid,
                              const extent_t<Conf::dim>& ext,
                              PtcContext& context, index_t<Conf::dim>& pos,
                              value_t dt) const {

  }

 private:
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
};

}  // namespace Aperture

#endif  // _COORD_POLICY_CARTESIAN_GCA_H_
