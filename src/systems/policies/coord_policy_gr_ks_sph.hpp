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

#ifndef __COORD_POLICY_GR_KS_SPH_H_
#define __COORD_POLICY_GR_KS_SPH_H_

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
                   const vec_t<value_t, 3> &B, const vec_t<value_t, 3> &D,
                   value_t dt, value_t e_over_m) {
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
}

template <typename value_t>
HOST_DEVICE void
gr_ks_geodesic_advance(value_t a, value_t dt, vec_t<value_t, 3> &x,
                       vec_t<value_t, 3> &u, bool is_photon = false,
                       int n_iter = 3) {
  vec_t<value_t, 3> x0 = x, x1 = x;
  vec_t<value_t, 3> u0 = u, u1 = u;

  for (int i = 0; i < n_iter; i++) {
    auto x_tmp = (x0 + x) * 0.5;
    auto u_tmp = (u0 + u) * 0.5;
    x1 = x0 + geodesic_ks_x_rhs(a, x_tmp, u_tmp, is_photon) * dt;
    u1 = u0 + geodesic_ks_u_rhs(a, x_tmp, u_tmp, is_photon) * dt;
    x = x1;
    u = u1;
  }
}

// template <typename Conf>
// void
// ptc_outflow(particle_data_t& ptc, const grid_ks_t<Conf>& grid,
//             int damping_length) {
//   auto ptc_num = ptc.number();
//   kernel_launch(
//       [ptc_num, damping_length] __device__(auto ptc, auto gp) {
//         auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
//         auto ext = grid.extent();
//         for (auto n : grid_stride_range(0, ptc_num)) {
//           auto c = ptc.cell[n];
//           if (c == empty_cell) continue;

//           auto idx = typename Conf::idx_t(c, grid.extent());
//           // auto pos = idx.get_pos();
//           auto pos = get_pos(idx, ext);
//           auto flag = ptc.flag[n];
//           if (check_flag(flag, PtcFlag::ignore_EM)) continue;
//           if (pos[0] > grid.dims[0] - damping_length + 2) {
//             flag |= flag_or(PtcFlag::ignore_EM);
//             ptc.flag[n] = flag;
//           }
//         }
//       },
//       ptc.get_dev_ptrs(), grid.get_grid_ptrs());
//   CudaSafeCall(cudaDeviceSynchronize());
//   CudaCheckError();
// }

} // namespace

template <typename Conf> class coord_policy_gr_ks_sph {
public:
  typedef typename Conf::value_t value_t;

  coord_policy_gr_ks_sph(const grid_t<Conf> &grid)
      : m_grid(dynamic_cast<const grid_ks_t<Conf> &>(grid)) {
    m_a = m_grid.a;
  }
  ~coord_policy_gr_ks_sph() = default;

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
    m_E[0] = E->at(0).ndptr_const();
    m_E[1] = E->at(1).ndptr_const();
    m_E[2] = E->at(2).ndptr_const();
    m_B[0] = B->at(0).ndptr_const();
    m_B[1] = B->at(1).ndptr_const();
    m_B[2] = B->at(2).ndptr_const();
#endif
  }

  // Static coordinate functions
  HD_INLINE static value_t weight_func(value_t x1, value_t x2,
                                       value_t x3 = 0.0f) {
    return math::sin(grid_ks_t<Conf>::theta(x2));
  }

  HD_INLINE static value_t x1(value_t x) { return grid_ks_t<Conf>::radius(x); }
  HD_INLINE static value_t x2(value_t x) { return grid_ks_t<Conf>::theta(x); }
  HD_INLINE static value_t x3(value_t x) { return x; }

  // Inline functions to be called in the particle update loop
  template <typename PtcContext>
  HD_INLINE void update_ptc(const Grid<Conf::dim, value_t> &grid,
                            const extent_t<Conf::dim> &ext, PtcContext &context,
                            index_t<Conf::dim> &pos, value_t dt) const {
    vec_t<value_t, 3> x_global = grid.pos_global(pos, context.x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);

    if (!check_flag(context.flag, PtcFlag::ignore_EM)) {
      gr_ks_boris_update(m_a, x_global, context.p, context.B, context.E,
                         0.5f * dt, context.q / context.m);
      // dt, context.q / context.m);
      // printf("%f, %f, %f\n", context.B[0], context.B[1], context.B[2]);
    }

    move_ptc(grid, context, x_global, pos, dt, false);

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

      gr_ks_boris_update(m_a, x_global, context.p, context.B, context.E,
                         0.5f * dt, context.q / context.m);
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

    context.new_x[0] =
        context.x[0] + (grid_ks_t<Conf>::from_radius(new_x[0]) -
                        grid_ks_t<Conf>::from_radius(x_global[0])) *
                           grid.inv_delta[0];
    context.new_x[1] =
        context.x[1] + (grid_ks_t<Conf>::from_theta(new_x[1]) -
                        grid_ks_t<Conf>::from_theta(x_global[1])) *
                           grid.inv_delta[1];
    if constexpr (Conf::dim == 2) {
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
  }

  // Inline functions to be called in the photon update loop
  HD_INLINE void update_ph(const Grid<Conf::dim, value_t> &grid,
                           ph_context<Conf::dim, value_t> &context,
                           index_t<Conf::dim> &pos, value_t dt) const {
    vec_t<value_t, 3> x_global = grid.pos_global(pos, context.x);
    x_global[0] = grid_ks_t<Conf>::radius(x_global[0]);
    x_global[1] = grid_ks_t<Conf>::theta(x_global[1]);

    move_ptc(grid, context, x_global, pos, dt, true);
  }

  // Extra processing routines
  template <typename ExecPolicy>
  void process_J_Rho(vector_field<Conf> &J, data_array<scalar_field<Conf>> &Rho,
                     value_t dt, bool process_rho) const {
    auto num_species = Rho.size();
    ExecPolicy::launch(
        [dt, num_species, process_rho] LAMBDA(auto j, auto rho,
                                              auto grid_ptrs) {
          auto &grid = ExecPolicy::grid();
          auto ext = grid.extent();

          auto w = grid.cell_size();
          ExecPolicy::loop(
              Conf::begin(ext), Conf::end(ext),
              [&grid, &ext, num_species, w, process_rho] LAMBDA(
                  auto idx, auto &j, auto &rho, const auto &grid_ptrs) {
                auto pos = get_pos(idx, ext);
                // if (grid.is_in_bound(pos)) {
                j[0][idx] *= w / grid_ptrs.Ad[0][idx];
                j[1][idx] *= w / grid_ptrs.Ad[1][idx];
                j[2][idx] *= w / grid_ptrs.Ad[2][idx];
                for (int n = 0; n < num_species; n++) {
                  rho[n][idx] *=
                      w / grid_ptrs.Ad[2][idx]; // A_phi is effectively dV
                }
                // }
                typename Conf::value_t theta =
                    grid.template pos<1>(pos[1], true);
                if (theta < 0.1 * grid.delta[1] ||
                    math::abs(theta - M_PI) < 0.1 * grid.delta[1]) {
                  j[2][idx] = 0.0f;
                }
              },
              j, rho, grid_ptrs);
        },
        J, Rho, m_grid.get_grid_ptrs());
    ExecPolicy::sync();
  }

  template <typename ExecPolicy>
  void filter_field(vector_field<Conf> &field,
                    typename Conf::multi_array_t &tmp,
                    const vec_t<bool, Conf::dim * 2> &is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ad[0],
                                       is_boundary);
    filter_field_component<ExecPolicy>(field.at(1), tmp, m_grid.m_Ad[1],
                                       is_boundary);
    filter_field_component<ExecPolicy>(field.at(2), tmp, m_grid.m_Ad[2],
                                       is_boundary);
  }

  template <typename ExecPolicy>
  void filter_field(scalar_field<Conf> &field,
                    typename Conf::multi_array_t &tmp,
                    const vec_t<bool, Conf::dim * 2> &is_boundary) const {
    filter_field_component<ExecPolicy>(field.at(0), tmp, m_grid.m_Ab[2],
                                       is_boundary);
  }

private:
  const grid_ks_t<Conf> &m_grid;
  value_t m_a;
  // vec_t<typename Conf::multi_array_t::cref_t, 3> m_E;
  // vec_t<typename Conf::multi_array_t::cref_t, 3> m_B;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_E;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
};

} // namespace Aperture

#endif // __COORD_POLICY_GR_KS_SPH_H_
