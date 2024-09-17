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
#include "systems/grid_ks.h"
#include "systems/helpers/filter_field.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "systems/physics/geodesic_ks.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "utils/interpolation.hpp"

namespace Aperture {

namespace {

template <typename value_t>
HOST_DEVICE void
gr_ks_boris_update(value_t a, const vec_t<value_t, 3> &x, vec_t<value_t, 3> &u,
                   value_t &gamma, const vec_t<value_t, 3> &B,
                   const vec_t<value_t, 3> &D, value_t dt, value_t e_over_m) {
  value_t sth = math::sin(x[1]);
  value_t cth = math::cos(x[1]);

  value_t g_13 = Metric_KS::g_13(a, x[0], sth, cth);
  value_t g_11 = Metric_KS::g_11(a, x[0], sth, cth);
  value_t g_22 = Metric_KS::g_22(a, x[0], sth, cth);
  value_t g_33 = Metric_KS::g_33(a, x[0], sth, cth);
  value_t gu11 = Metric_KS::gu11(a, x[0], sth, cth);
  value_t gu22 = Metric_KS::gu22(a, x[0], sth, cth);
  value_t gu33 = Metric_KS::gu33(a, x[0], sth, cth);
  value_t gu13 = Metric_KS::gu13(a, x[0], sth, cth);
  value_t sqrtg = Metric_KS::sqrt_gamma(a, x[0], sth, cth);

  vec_t<value_t, 3> D_l(0.0f);
  D_l[0] = g_11 * D[0] + g_13 * D[2];
  D_l[1] = g_22 * D[1];
  D_l[2] = g_33 * D[2] + g_13 * D[0];
  D_l *= 0.5f * dt * e_over_m * Metric_KS::alpha(a, x[0], sth, cth);

  vec_t<value_t, 3> u_minus = u + D_l;

  vec_t<value_t, 3> t =
      B * 0.5f * dt * e_over_m / Metric_KS::u0(a, x[0], sth, cth, u_minus);
  value_t t2 = g_11 * t[0] * t[0] + g_22 * t[1] * t[1] + g_33 * t[2] * t[2] +
               2.0f * g_13 * t[0] * t[2];
  value_t s = 2.0f / (1.0f + t2);

  vec_t<value_t, 3> u_prime = u_minus;
  u_prime[0] += sqrtg * (gu22 * u_minus[1] * t[2] -
                         (gu33 * u_minus[2] + gu13 * u_minus[0]) * t[1]);
  u_prime[1] += sqrtg * ((gu33 * u_minus[2] + gu13 * u_minus[0]) * t[0] -
                         (gu11 * u_minus[0] + gu13 * u_minus[2]) * t[2]);
  u_prime[2] += sqrtg * ((gu11 * u_minus[0] + gu13 * u_minus[2]) * t[1] -
                         gu22 * u_minus[1] * t[0]);

  u = u_minus + D_l;
  u[0] += sqrtg *
          (gu22 * u_prime[1] * t[2] -
           (gu33 * u_prime[2] + gu13 * u_prime[0]) * t[1]) *
          s;
  u[1] += sqrtg *
          ((gu33 * u_prime[2] + gu13 * u_prime[0]) * t[0] -
           (gu11 * u_prime[0] + gu13 * u_prime[2]) * t[2]) *
          s;
  u[2] += sqrtg *
          ((gu11 * u_prime[0] + gu13 * u_prime[2]) * t[1] -
           gu22 * u_prime[1] * t[0]) *
          s;
  gamma = Metric_KS::u0(a, x[0], sth, cth, u, false);
}

template <typename value_t>
HOST_DEVICE void
gr_ks_vay_update(value_t a, const vec_t<value_t, 3> &x, vec_t<value_t, 3> &u,
                 value_t &gamma, const vec_t<value_t, 3> &B,
                 const vec_t<value_t, 3> &D, value_t dt, value_t e_over_m) {
  value_t sth = math::sin(x[1]);
  value_t cth = math::cos(x[1]);
  vec_t<value_t, 4> u_4;
  u_4[0] = Metric_KS::u_0(a, x[0], sth, cth, u, false);
  u_4[1] = u[0];
  u_4[2] = u[1];
  u_4[3] = u[2];

  auto u_fido = Metric_KS::convert_to_FIDO_lower(u_4, a, x[0], sth, cth);
  u_fido[0] = -u_fido[0]; // convert u_0 to gamma
  value_t g_11 = Metric_KS::g_11(a, x[0], sth, cth);
  value_t g_22 = Metric_KS::g_22(a, x[0], sth, cth);
  value_t g_33 = Metric_KS::g_33(a, x[0], sth, cth);
  value_t g_13 = Metric_KS::g_13(a, x[0], sth, cth);
  value_t sqrt_g11 = math::sqrt(g_11);
  value_t sqrt_g22 = math::sqrt(g_22);
  value_t sqrt_factor = math::sqrt(g_33 - g_13 * g_13 / g_11);

  value_t E1 = (D[0] * g_11 + D[2] * g_13) / sqrt_g11;
  value_t E2 = D[1] * sqrt_g22;
  value_t E3 = D[2] * sqrt_factor;
  value_t B1 = (B[0] * g_11 + B[2] * g_13) / sqrt_g11;
  value_t B2 = B[1] * sqrt_g22;
  value_t B3 = B[2] * sqrt_factor;

  vay_pusher pusher;
  pusher(u_fido[1], u_fido[2], u_fido[3], u_fido[0], E1, E2, E3, B1, B2, B3,
         dt * e_over_m * 0.5f, dt);
  u_fido[0] = -u_fido[0]; // convert gamma back to u_0

  u_4 = Metric_KS::convert_from_FIDO_lower(u_fido, a, x[0], sth, cth);
  u[0] = u_4[1];
  u[1] = u_4[2];
  u[2] = u_4[3];
  gamma = Metric_KS::u0(a, x[0], sth, cth, u, false);
}

template <typename value_t>
HOST_DEVICE void
gr_ks_momentum_advance(value_t a, value_t dt, vec_t<value_t, 3> &x,
                       vec_t<value_t, 3> &u, bool is_photon = false,
                       int n_iter = 3) {
  vec_t<value_t, 3> u0 = u, u1 = u;

  for (int i = 0; i < n_iter; i++) {
    // auto x_tmp = (x0 + x) * 0.5;
    auto u_tmp = (u0 + u1) * 0.5;
    u1 = u0 + geodesic_ks_u_rhs(a, x, u_tmp, is_photon) * dt;
  }
  u = u1;
}

template <typename value_t>
HOST_DEVICE void
gr_ks_position_advance(value_t a, value_t dt, vec_t<value_t, 3> &x,
                       vec_t<value_t, 3> &u, bool is_photon = false,
                       int n_iter = 3) {
  vec_t<value_t, 3> x0 = x, x1 = x;

  for (int i = 0; i < n_iter; i++) {
    auto x_tmp = (x0 + x1) * 0.5;
    // auto u_tmp = (u0 + u1) * 0.5;
    x1 = x0 + geodesic_ks_u_rhs(a, x_tmp, u, is_photon) * dt;
  }
  x = x1;
}

template <typename value_t>
HOST_DEVICE void
gr_ks_geodesic_advance(value_t a, value_t dt, vec_t<value_t, 3> &x,
                       vec_t<value_t, 3> &u, bool is_photon = false,
                       int n_iter = 5) {
  vec_t<value_t, 3> x0 = x, x1 = x;
  vec_t<value_t, 3> u0 = u, u1 = u;

  // if (is_photon) {
  //   // Use simple RK4 for photon
  //   auto kx1 = geodesic_ks_x_rhs(a, x, u, is_photon);
  //   auto ku1 = geodesic_ks_u_rhs(a, x, u, is_photon);
  //   auto kx2 = geodesic_ks_x_rhs(a, x + 0.5f * dt * kx1, u + 0.5f * dt * ku1,
  //                                 is_photon);
  //   auto ku2 = geodesic_ks_u_rhs(a, x + 0.5f * dt * kx1, u + 0.5f * dt * ku1,
  //                                 is_photon);
  //   auto kx3 = geodesic_ks_x_rhs(a, x + 0.5f * dt * kx2, u + 0.5f * dt * ku2,
  //                                 is_photon);
  //   auto ku3 = geodesic_ks_u_rhs(a, x + 0.5f * dt * kx2, u + 0.5f * dt * ku2,
  //                                 is_photon);
  //   auto kx4 = geodesic_ks_x_rhs(a, x + dt * kx3, u + dt * ku3, is_photon);
  //   auto ku4 = geodesic_ks_u_rhs(a, x + dt * kx3, u + dt * ku3, is_photon);

  //   x = x0 + (dt / 6.0f) * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4);
  //   u = u0 + (dt / 6.0f) * (ku1 + 2.0f * ku2 + 2.0f * ku3 + ku4);
  //   // printf("x is %f, %f, %f, u is %f, %f, %f\n", x[0], x[1], x[2], u[0], u[1], u[2]);
  //   // printf("u_0 is %f\n", Metric_KS::u_0(a, x[0], x[1], u));
  // } else {
    // Use predictor-corrector for particle
    for (int i = 0; i < n_iter; i++) {
      auto x_tmp = (x0 + x1) * 0.5;
      auto u_tmp = (u0 + u1) * 0.5;
      x1 = x0 + geodesic_ks_x_rhs(a, x_tmp, u_tmp, is_photon) * dt;
      u1 = u0 + geodesic_ks_u_rhs(a, x_tmp, u_tmp, is_photon) * dt;
    }
    // printf("x is %f, %f, %f, u is %f, %f, %f\n", x1[0], x1[1], x1[2], u1[0], u1[1], u1[2]);
    // printf("u_0 is %f\n", Metric_KS::u_0(a, x1[0], x1[1], u1));
    x = x1;
    u = u1;
  // }
}

}  // namespace

/// This is the Kerr Schild coordinate policy
template <typename Conf>
class coord_policy_gr_ks_sph {
 public:
  using value_t = typename Conf::value_t;
  using grid_type = grid_ks_t<Conf>;
  using this_type = coord_policy_gr_ks_sph<Conf>;

  coord_policy_gr_ks_sph(const grid_t<Conf> &grid)
      : m_grid(dynamic_cast<const grid_type &>(grid)) {
    m_a = m_grid.a;
  }
  ~coord_policy_gr_ks_sph() = default;

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

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return math::sin(grid_type::theta(x2));
  }

  HD_INLINE static value_t x1(value_t x) { return grid_type::radius(x); }
  HD_INLINE static value_t x2(value_t x) { return grid_type::theta(x); }
  HD_INLINE static value_t x3(value_t x) { return x; }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t> &grid,
                            const extent_t<Conf::dim> &ext, PtcContext &context,
                            index_t<Conf::dim> &pos, value_t dt) const {
    vec_t<value_t, 3> x_global = grid.coord_global(pos, context.x);
    x_global[0] = x1(x_global[0]);
    x_global[1] = x2(x_global[1]);

    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      gr_ks_vay_update(m_a, x_global, context.p, context.gamma, context.B,
                       // context.E, dt, context.q / context.m);
                         context.E, 0.5f * dt, context.q / context.m);
      // printf("%f, %f, %f\n", context.B[0], context.B[1], context.B[2]);
    }

    move_ptc(grid, context, x_global, pos, dt, false);
    // move_ptc(grid, context, x_global, pos, dt, true);

    // Second half push
    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      auto idx = typename Conf::idx_t(pos, ext);
      auto interp = interp_t<1, Conf::dim>{};
      context.E[0] = interp(context.new_x, m_E[0], idx, ext, stagger_t(0b110));
      context.E[1] = interp(context.new_x, m_E[1], idx, ext, stagger_t(0b101));
      context.E[2] = interp(context.new_x, m_E[2], idx, ext, stagger_t(0b011));
      context.B[0] = interp(context.new_x, m_B[0], idx, ext, stagger_t(0b001));
      context.B[1] = interp(context.new_x, m_B[1], idx, ext, stagger_t(0b010));
      context.B[2] = interp(context.new_x, m_B[2], idx, ext, stagger_t(0b100));

      // Note: context.p stores the lower components u_i, while gamma is upper
      // u^0.
      gr_ks_vay_update(m_a, x_global, context.p, context.gamma, context.B,
                         context.E, 0.5f * dt, context.q / context.m);
    }
  }

  // Abstracted moving routine that is shared by both ptc and ph
  template <typename PtcContext>
  HD_INLINE void move_ptc(const Grid<Conf::dim, value_t> &grid,
                          PtcContext &context, vec_t<value_t, 3> &new_x,
                          index_t<Conf::dim> &pos, value_t dt,
                          bool is_photon) const {
    vec_t<value_t, 3> x_global = new_x;
    // Both new_x and u are updated
    gr_ks_geodesic_advance(m_a, dt, new_x, context.p, is_photon);

    if (new_x[1] < 0.0) {
      new_x[1] = -new_x[1];
      new_x[2] += M_PI;
      context.p[1] = -context.p[1];
    } else if (new_x[1] >= M_PI) {
      new_x[1] = 2.0 * M_PI - new_x[1];
      new_x[2] -= M_PI;
      context.p[1] = -context.p[1];
    }
    auto r = new_x[0];
    auto th = new_x[1];

    context.new_x[0] = context.x[0] + (grid_type::from_radius(new_x[0]) -
                                       grid_type::from_radius(x_global[0])) *
                                          grid.inv_delta[0];
    context.new_x[1] = context.x[1] + (grid_type::from_theta(new_x[1]) -
                                       grid_type::from_theta(x_global[1])) *
                                          grid.inv_delta[1];
    
    // if constexpr (Conf::dim == 2) {
    if (Conf::dim == 2) {
      context.new_x[2] = new_x[2];
    } else {
      context.new_x[2] =
          context.x[2] + (new_x[2] - x_global[2]) * grid.inv_delta[2];
    }

#pragma unroll
    for (int i = 0; i < Conf::dim; i++) {
      context.dc[i] = std::floor(context.new_x[i]);
      pos[i] += context.dc[i];
      context.new_x[i] -= (value_t)context.dc[i];
    }

    // Update the energy of the particle for reference
    context.gamma = Metric_KS::u0(m_a, r, th, context.p, is_photon);
  }

  // Inline functions to be called in the photon update loop
  HD_INLINE void update_ph(const Grid<Conf::dim, value_t> &grid,
                           ph_context<Conf::dim, value_t> &context,
                           index_t<Conf::dim> &pos, value_t dt) const {
    vec_t<value_t, 3> x_global = grid.coord_global(pos, context.x);
    x_global[0] = x1(x_global[0]);
    x_global[1] = x2(x_global[1]);

    move_ptc(grid, context, x_global, pos, dt, true);
  }

  // Extra processing routines
  template <typename ExecPolicy>
  void process_J_Rho(vector_field<Conf> &J, data_array<scalar_field<Conf>> &Rho,
                     scalar_field<Conf> &rho_total, value_t dt, bool process_rho) const {
    auto num_species = Rho.size();
    auto a = m_a;
    ExecPolicy::launch(
        [dt, num_species, process_rho, a] LAMBDA(auto j, auto rho,
                                                 auto rho_total, auto grid_ptrs) {
          auto &grid = ExecPolicy::grid();
          auto ext = grid.extent();

          auto w = grid.cell_size();
          ExecPolicy::loop(
              Conf::begin(ext), Conf::end(ext),
              // [&] LAMBDA(auto idx, auto &j, auto &rho, const auto &grid_ptrs) {
              [&] LAMBDA(auto idx) {
                auto pos = get_pos(idx, ext);
                value_t r_s = this_type::x1(grid.coord(0, pos[0], true));
                value_t r = this_type::x1(grid.coord(0, pos[0], false));
                value_t th_s = this_type::x2(grid.coord(1, pos[1], true));
                value_t th = this_type::x2(grid.coord(1, pos[1], false));
                // if (grid.is_in_bound(pos)) {

                // if (pos[0] == 40 && pos[1] == grid.guard[1]) {
                //     // && th_s < 0.1 * grid.delta[1]) {
                //   printf("at %d, j1 is %f, j0 is %f, rho is %f\n",
                //          pos[0], j[1][idx], j[0][idx], rho_total[idx] / dt);
                // }
                // if (pos[0] == 39 && pos[1] == grid.guard[1]) {
                //   printf("at %d, j0 is %f\n",
                //          pos[0], j[0][idx]);
                // }
                j[0][idx] *= w / grid_ptrs.Ad[0][idx];
                j[1][idx] *= w / grid_ptrs.Ad[1][idx];
                j[2][idx] *= w / grid_ptrs.Ad[2][idx];
                rho_total[idx] *=
                    w / grid_ptrs.Ad[2][idx];  // A_phi is effectively dV
                for (int n = 0; n < num_species; n++) {
                  rho[n][idx] *=
                      w / grid_ptrs.Ad[2][idx];  // A_phi is effectively dV
                }
                // }
                if (th_s < 0.1 * grid.delta[1] ||
                    math::abs(th_s - M_PI) < 0.1 * grid.delta[1]) {
                  j[2][idx] = 0.0f;
                }
              });
              // j, rho, grid_ptrs);
        },
        J, Rho, rho_total, m_grid.get_grid_ptrs());
    ExecPolicy::sync();
  }

  template <typename ExecPolicy>
  void filter_field(vector_field<Conf> &field,
                    typename Conf::multi_array_t &tmp,
                    const vec_t<bool, Conf::dim * 2> &is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ad[0],
                                       is_boundary, field.stagger(0));
    filter_field_component<ExecPolicy>(field.at(1), tmp, m_grid.m_Ad[1],
                                       is_boundary, field.stagger(1));
    filter_field_component<ExecPolicy>(field.at(2), tmp, m_grid.m_Ad[2],
                                       is_boundary, field.stagger(2));
  }

  template <typename ExecPolicy>
  void filter_field(scalar_field<Conf> &field,
                    typename Conf::multi_array_t &tmp,
                    const vec_t<bool, Conf::dim * 2> &is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ad[2],
                                       is_boundary, field.stagger());
  }

 private:
  const grid_type &m_grid;
  value_t m_a;
  // vec_t<typename Conf::multi_array_t::cref_t, 3> m_E;
  // vec_t<typename Conf::multi_array_t::cref_t, 3> m_B;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
};

}  // namespace Aperture
