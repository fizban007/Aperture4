/*
 * Copyright (c) 2020 Alex Chen.
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

#include "core/cuda_control.h"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_gr_ks.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

namespace {

template <typename Conf>
void
axis_boundary_e(vector_field<Conf> &D, const grid_ks_t<Conf> &grid) {
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [] __device__(auto D) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto n1_0 = grid.guard[1];
          if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_0, true))) <
              0.1f * grid.delta[1]) {
            // At the theta = 0 axis
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            D[2][idx] = 0.0f;               // Ephi to zero
            D[1][idx.dec_y()] = D[1][idx];  // Mirror Eth
          }
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_pi, true)) -
                  M_PI) < 0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            D[2][idx] = 0.0f;               // Ephi to zero
            D[1][idx] = D[1][idx.dec_y()];  // Mirror Eth
          }
        }
      },
      D.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
axis_boundary_b(vector_field<Conf> &B, const grid_ks_t<Conf> &grid) {
  typedef typename Conf::idx_t idx_t;
  kernel_launch(
      [] __device__(auto B) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          // for (int n1_0 = grid.guard[1]; n1_0 >= 0; n1_0--) {
          int n1_0 = grid.guard[1];
          if (grid_ks_t<Conf>::theta(grid.template pos<1>(n1_0, true)) <
              0.1f * grid.delta[1]) {
            // At the theta = 0 axis
            auto idx = idx_t(index_t<2>(n0, n1_0), ext);
            B[1][idx] = 0.0f;               // Bth to zero
            B[2][idx.dec_y()] = B[2][idx];  // Mirror Bphi
            B[0][idx.dec_y()] = B[0][idx];  // Mirror Br
          }
          int n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid_ks_t<Conf>::theta(grid.template pos<1>(n1_pi, true)) -
                  M_PI) < 0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = idx_t(index_t<2>(n0, n1_pi), ext);
            B[1][idx] = 0.0f;               // Bth to zero
            B[2][idx] = B[2][idx.dec_y()];  // Mirror Bphi
            B[0][idx] = B[0][idx.dec_y()];  // Mirror Br
          }
        }
      },
      B.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
inner_boundary(vector_field<Conf> &D, vector_field<Conf> &B,
               const grid_ks_t<Conf> &grid) {
  using value_t = typename Conf::value_t;
  using namespace Metric_KS;

  int boundary_cell = grid.guard[0];

  kernel_launch(
      [boundary_cell] __device__(auto D, auto B, auto grid_ptrs, auto a) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          auto pos = index_t<2>(boundary_cell, n1);
          auto idx = Conf::idx(pos, ext);

          B[1][idx.dec_x()] = B[1][idx];
          B[2][idx.dec_x()] = B[2][idx];
          D[0][idx.dec_x()] = D[0][idx];

          D[1][idx.dec_x()] = D[1][idx];
          D[2][idx.dec_x()] = D[2][idx];
          B[0][idx.dec_x()] = B[0][idx];
        }
      },
      D.get_ptrs(), B.get_ptrs(), grid.get_grid_ptrs(), grid.a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_flux(scalar_field<Conf> &flux, const vector_field<Conf> &b,
             const grid_ks_t<Conf> &grid) {
  flux.init();
  auto ext = grid.extent();
  kernel_launch(
      [ext] __device__(auto flux, auto b, auto a, auto grid_ptrs) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(n0, true));

          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            auto pos = index_t<Conf::dim>(n0, n1);
            auto idx = typename Conf::idx_t(pos, ext);

            flux[idx] = flux[idx.dec_y()] + b[0][idx] * grid_ptrs.Ab[0][idx];
          }
        }
      },
      flux.dev_ndptr(), b.get_ptrs(), grid.a, grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_divs(scalar_field<Conf> &divD, scalar_field<Conf> &divB,
             const vector_field<Conf> &D, const vector_field<Conf> &B,
             const grid_ks_t<Conf> &grid) {
  kernel_launch(
      [] __device__(auto div_e, auto e, auto div_b, auto b, auto grid_ptrs) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            div_b[idx] = (b[0][idx.inc_x()] * grid_ptrs.Ab[0][idx.inc_x()] -
                          b[0][idx] * grid_ptrs.Ab[0][idx] +
                          b[1][idx.inc_y()] * grid_ptrs.Ab[1][idx.inc_y()] -
                          b[1][idx] * grid_ptrs.Ab[1][idx]) /
                         grid_ptrs.Ab[2][idx];

            div_e[idx] = (e[0][idx] * grid_ptrs.Ad[0][idx] -
                          e[0][idx.dec_x()] * grid_ptrs.Ad[0][idx.dec_x()] +
                          e[1][idx] * grid_ptrs.Ad[1][idx] -
                          e[1][idx.dec_y()] * grid_ptrs.Ad[1][idx.dec_y()]) /
                         grid_ptrs.Ad[2][idx];

            // if (pos[0] == 9 && pos[1] == 254) {
            //   printf(
            //       "divB is %f, bA0_p is %f, bA0_m is %f, bA1_p is %f, bA1_m
            //       is "
            //       "%f\n",
            //       div_b[idx], b[0][idx.inc_x()] *
            //       grid_ptrs.Ab[0][idx.inc_x()], b[0][idx] *
            //       grid_ptrs.Ab[0][idx], b[1][idx.inc_y()] *
            //       grid_ptrs.Ab[1][idx.inc_y()], b[1][idx] *
            //       grid_ptrs.Ab[1][idx]);
            // }
          }
        }
      },
      divD[0].dev_ndptr(), D.get_const_ptrs(), divB[0].dev_ndptr(),
      B.get_const_ptrs(), grid.get_grid_ptrs());
}

template <typename Conf>
void
damping_boundary(vector_field<Conf> &E, vector_field<Conf> &B,
                 const vector_field<Conf> &E0, const vector_field<Conf> &B0,
                 int damping_length, typename Conf::value_t damping_coef) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  kernel_launch(
      [damping_length, damping_coef] __device__(auto e, auto b, auto e0,
                                                auto b0) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n1 :
             grid_stride_range(grid.guard[1], grid.dims[1] - grid.guard[1])) {
          // for (int i = 0; i < damping_length - grid.skirt[0] - 1; i++) {
          for (int i = 0; i < damping_length - 1; i++) {
            int n0 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            e[0][idx] = e0[0][idx] + lambda * (e[0][idx] - e0[0][idx]);
            e[1][idx] = e0[1][idx] + lambda * (e[1][idx] - e0[1][idx]);
            e[2][idx] = e0[2][idx] + lambda * (e[2][idx] - e0[2][idx]);
            b[0][idx] = b0[0][idx] + lambda * (b[0][idx] - b0[0][idx]);
            b[1][idx] = b0[1][idx] + lambda * (b[1][idx] - b0[1][idx]);
            b[2][idx] = b0[2][idx] + lambda * (b[2][idx] - b0[2][idx]);
          }
        }
      },
      E.get_ptrs(), B.get_ptrs(), E0.get_ptrs(), B0.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace

template <typename Conf>
field_solver_gr_ks_cu<Conf>::~field_solver_gr_ks_cu() {
  // sp_buffer.resize(0);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::init() {
  field_solver<Conf>::init();

  sim_env().params().get_value("bh_spin", m_a);
  Logger::print_info("bh_spin in field solver is {}", m_a);
  sim_env().params().get_value("implicit_beta", this->m_beta);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);

  m_prev_D.resize(this->m_grid);
  m_prev_B.resize(this->m_grid);
  m_new_D.resize(this->m_grid);
  m_new_B.resize(this->m_grid);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::register_data_components() {
  field_solver_cu<Conf>::register_data_components();

  flux = sim_env().template register_data<scalar_field<Conf>>(
      "flux", this->m_grid, field_type::vert_centered);
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::iterate_predictor(double dt) {
  auto a = m_a;
  auto beta = this->m_beta;
  vec_t<bool, Conf::dim * 2> is_boundary;
  is_boundary = true;
  if (this->m_comm != nullptr)
    is_boundary = this->m_comm->domain_info().is_boundary;

  // Iterate all fields at once
  auto update_B_kernel = [dt, a, beta, is_boundary] __device__(
                             auto B, auto prevD, auto prevB, auto nextD,
                             auto nextB, auto grid_ptrs) {
    using namespace Metric_KS;

    auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
    auto ext = grid.extent();
    auto alpha = 1.0f - beta;
    for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
      auto pos = get_pos(idx, ext);
      if (grid.is_in_bound(pos)) {
        // First construct the auxiliary fields E and H
        // value_t r =
        //     grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
        value_t r_sp =
            grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0] + 1, true));
        value_t r_sm =
            grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));

        // value_t th =
        //     grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
        value_t th_sp =
            grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1] + 1, true));
        value_t th_sm =
            grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));

        value_t sth = math::sin(th_sm);
        value_t cth = math::cos(th_sm);
        auto Eph00 =
            ag_33(a, r_sm, sth, cth) *
                (alpha * prevD[2][idx] + beta * nextD[2][idx]) +
            ag_13(a, r_sm, sth, cth) * 0.5f *
                (alpha * prevD[0][idx] + beta * nextD[0][idx] +
                 alpha * prevD[0][idx.dec_x()] + beta * nextD[0][idx.dec_x()]) +
            0.5f * sq_gamma_beta(a, r_sm, sth, cth) *
                ((alpha * prevB[1][idx] + beta * nextB[1][idx]) +
                 (alpha * prevB[1][idx.dec_x()] +
                  beta * nextB[1][idx.dec_x()]));

        auto Eph10 =
            ag_33(a, r_sp, sth, cth) *
                (alpha * prevD[2][idx.inc_x()] + beta * nextD[2][idx.inc_x()]) +
            ag_13(a, r_sp, sth, cth) * 0.5f *
                (alpha * prevD[0][idx.inc_x()] + beta * nextD[0][idx.inc_x()] +
                 alpha * prevD[0][idx] + beta * nextD[0][idx]) +
            0.5f * sq_gamma_beta(a, r_sp, sth, cth) *
                ((alpha * prevB[1][idx.inc_x()] +
                  beta * nextB[1][idx.inc_x()]) +
                 (alpha * prevB[1][idx] + beta * nextB[1][idx]));

        B[1][idx] = prevB[1][idx] - dt * (Eph00 - Eph10) / grid_ptrs.Ab[1][idx];

        sth = math::sin(th_sp);
        cth = math::cos(th_sp);
        auto Eph01 =
            ag_33(a, r_sm, sth, cth) *
                (alpha * prevD[2][idx.inc_y()] + beta * nextD[2][idx.inc_y()]) +
            ag_13(a, r_sm, sth, cth) * 0.5f *
                (alpha * prevD[0][idx.inc_y()] + beta * nextD[0][idx.inc_y()] +
                 alpha * prevD[0][idx.dec_x().inc_y()] +
                 beta * nextD[0][idx.dec_x().inc_y()]) +
            0.5f * sq_gamma_beta(a, r_sm, sth, cth) *
                ((alpha * prevB[1][idx.inc_y()] +
                  beta * nextB[1][idx.inc_y()]) +
                 (alpha * prevB[1][idx.dec_x().inc_y()] +
                  beta * nextB[1][idx.dec_x().inc_y()]));

        B[0][idx] = prevB[0][idx] - dt * (Eph01 - Eph00) / grid_ptrs.Ab[0][idx];

        // Updating Bph
        auto Er1 =
            grid_ptrs.ag11dr_e[idx.inc_y()] *
                (alpha * prevD[0][idx.inc_y()] + beta * nextD[0][idx.inc_y()]) +
            grid_ptrs.ag13dr_e[idx.inc_y()] * 0.5f *
                (alpha * prevD[2][idx.inc_y()] + beta * nextD[2][idx.inc_y()] +
                 alpha * prevD[2][idx.inc_y().inc_x()] +
                 beta * nextD[2][idx.inc_y().inc_x()]);

        auto Er0 =
            grid_ptrs.ag11dr_e[idx] *
                (alpha * prevD[0][idx] + beta * nextD[0][idx]) +
            grid_ptrs.ag13dr_e[idx] * 0.5f *
                (alpha * prevD[2][idx] + beta * nextD[2][idx] +
                 alpha * prevD[2][idx.inc_x()] + beta * nextD[2][idx.inc_x()]);

        auto Eth1 =
            grid_ptrs.ag22dth_e[idx.inc_x()] *
                (alpha * prevD[1][idx.inc_x()] + beta * nextD[1][idx.inc_x()]) -
            grid_ptrs.gbetadth_e[idx.inc_x()] * 0.5f *
                (alpha * (prevB[2][idx.inc_x()] + prevB[2][idx]) +
                 beta * (nextB[2][idx.inc_x()] + nextB[2][idx]));

        auto Eth0 = grid_ptrs.ag22dth_e[idx] *
                        (alpha * prevD[1][idx] + beta * nextD[1][idx]) -
                    grid_ptrs.gbetadth_e[idx] * 0.5f *
                        (alpha * (prevB[2][idx] + prevB[2][idx.dec_x()]) +
                         beta * (nextB[2][idx] + nextB[2][idx.dec_x()]));

        B[2][idx] = prevB[2][idx] -
                    dt * ((Er0 - Er1) + (Eth1 - Eth0)) / grid_ptrs.Ab[2][idx];

        // Boundary conditions
        if (pos[1] == grid.guard[1] && is_boundary[2]) {
          // theta = 0 axis
          B[1][idx] = 0.0f;
        }

        if (pos[1] == grid.dims[1] - grid.guard[1] - 1 && is_boundary[3]) {
          // theta = pi axis
          B[1][idx.inc_y()] = 0.0f;
        }

        if (pos[0] == grid.guard[0] && is_boundary[0]) {
          // Inner boundary
          B[0][idx.dec_x()] = B[0][idx];
          B[1][idx.dec_x()] = B[1][idx];
          B[2][idx.dec_x()] = B[2][idx];
        }

        if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && is_boundary[1]) {
          // Outer boundary
          B[0][idx.inc_x()] = B[0][idx];
          B[1][idx.inc_x()] = B[1][idx];
          B[2][idx.inc_x()] = B[2][idx];
        }
      }
    }
  };

  auto update_D_kernel = [dt, a, beta, is_boundary] __device__(
                             auto D, auto prevD, auto prevB, auto nextD,
                             auto nextB, auto J, auto grid_ptrs) {
    using namespace Metric_KS;

    auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
    auto ext = grid.extent();
    auto alpha = 1.0f - beta;
    for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
      auto pos = get_pos(idx, ext);
      if (grid.is_in_bound(pos)) {
        // First construct the auxiliary fields E and H
        // value_t r =
        //     grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
        value_t r_p =
            grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
        value_t r_m =
            grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0] - 1, false));

        // value_t th =
        //     grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
        value_t th_p =
            grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
        value_t th_m =
            grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1] - 1, false));

        value_t sth = math::sin(th_p);
        value_t cth = math::cos(th_p);
        auto Hph11 =
            ag_33(a, r_p, sth, cth) *
                (alpha * prevB[2][idx] + beta * nextB[2][idx]) +
            ag_13(a, r_p, sth, cth) * 0.5f *
                (alpha * prevB[0][idx] + beta * nextB[0][idx] +
                 alpha * prevB[0][idx.inc_x()] + beta * nextB[0][idx.inc_x()]) -
            0.5f * sq_gamma_beta(a, r_p, sth, cth) *
                ((alpha * prevD[1][idx] + beta * nextD[1][idx]) +
                 (alpha * prevD[1][idx.inc_x()] +
                  beta * nextD[1][idx.inc_x()]));

        auto Hph01 =
            ag_33(a, r_m, sth, cth) *
                (alpha * prevB[2][idx.dec_x()] + beta * nextB[2][idx.dec_x()]) +
            ag_13(a, r_m, sth, cth) * 0.5f *
                (alpha * prevB[0][idx.dec_x()] + beta * nextB[0][idx.dec_x()] +
                 alpha * prevB[0][idx] + beta * nextB[0][idx]) -
            0.5f * sq_gamma_beta(a, r_m, sth, cth) *
                ((alpha * prevD[1][idx.dec_x()] +
                  beta * nextD[1][idx.dec_x()]) +
                 (alpha * prevD[1][idx] + beta * nextD[1][idx]));

        D[1][idx] = prevD[1][idx] +
                    dt * (Hph01 - Hph11) / grid_ptrs.Ad[1][idx] -
                    dt * J[1][idx];

        sth = math::sin(th_m);
        cth = math::cos(th_m);
        auto Hph10 =
            ag_33(a, r_p, sth, cth) *
                (alpha * prevB[2][idx.dec_y()] + beta * nextB[2][idx.dec_y()]) +
            ag_13(a, r_p, sth, cth) * 0.5f *
                (alpha * prevB[0][idx.dec_y()] + beta * nextB[0][idx.dec_y()] +
                 alpha * prevB[0][idx.inc_x().dec_y()] +
                 beta * nextB[0][idx.inc_x().dec_y()]) -
            0.5f * sq_gamma_beta(a, r_p, sth, cth) *
                ((alpha * prevD[1][idx.dec_y()] +
                  beta * nextD[1][idx.dec_y()]) +
                 (alpha * prevD[1][idx.inc_x().dec_y()] +
                  beta * nextD[1][idx.inc_x().dec_y()]));

        if (pos[1] == grid.guard[1] && is_boundary[2]) {
          Hph10 = -Hph11;
        }

        D[0][idx] = prevD[0][idx] +
                    dt * (Hph11 - Hph10) / grid_ptrs.Ad[0][idx] -
                    dt * J[0][idx];

        // Do an extra cell at the theta = PI axis
        if (pos[1] == grid.dims[1] - grid.guard[1] - 1 && is_boundary[3]) {
          D[0][idx.inc_y()] = prevD[0][idx.inc_y()] +
              dt * (-2.0f * Hph11) / grid_ptrs.Ad[0][idx.inc_y()] -
              dt * J[0][idx.inc_y()];

          if (pos[0] == grid.guard[0] && is_boundary[0]) {
            D[0][idx.inc_y().dec_x()] = D[0][idx.inc_y()];
          }
          if (pos[0] == grid.dims[0] - grid.guard[0] - 1 &&
              is_boundary[1]) {
            D[0][idx.inc_y().inc_x()] = D[0][idx.inc_y()];
          }
        }

        // Updating Dph
        auto Hr0 =
            grid_ptrs.ag11dr_h[idx.dec_y()] *
                (alpha * prevB[0][idx.dec_y()] + beta * nextB[0][idx.dec_y()]) +
            grid_ptrs.ag13dr_h[idx.dec_y()] * 0.5f *
                (alpha * prevB[2][idx.dec_y()] + beta * nextB[2][idx.dec_y()] +
                 alpha * prevB[2][idx.dec_y().dec_x()] +
                 beta * nextB[2][idx.dec_y().dec_x()]);

        auto Hr1 =
            grid_ptrs.ag11dr_h[idx] *
                (alpha * prevB[0][idx] + beta * nextB[0][idx]) +
            grid_ptrs.ag13dr_h[idx] * 0.5f *
                (alpha * prevB[2][idx] + beta * nextB[2][idx] +
                 alpha * prevB[2][idx.dec_x()] + beta * nextB[2][idx.dec_x()]);

        auto Hth0 =
            grid_ptrs.ag22dth_h[idx.dec_x()] *
                (alpha * prevB[1][idx.dec_x()] + beta * nextB[1][idx.dec_x()]) +
            grid_ptrs.gbetadth_h[idx.dec_x()] * 0.5f *
                (alpha * (prevD[2][idx.dec_x()] + prevD[2][idx]) +
                 beta * (nextD[2][idx.dec_x()] + nextD[2][idx]));

        auto Hth1 = grid_ptrs.ag22dth_h[idx] *
                        (alpha * prevB[1][idx] + beta * nextB[1][idx]) +
                    grid_ptrs.gbetadth_h[idx] * 0.5f *
                        (alpha * (prevD[2][idx] + prevD[2][idx.inc_x()]) +
                         beta * (nextD[2][idx] + nextD[2][idx.inc_x()]));

        D[2][idx] = prevD[2][idx] +
                    dt * ((Hr0 - Hr1) + (Hth1 - Hth0)) / grid_ptrs.Ad[2][idx] -
                    dt * J[2][idx];

        // Boundary conditions
        if (pos[1] == grid.guard[1] && is_boundary[2]) {
          // theta = 0 axis
          D[2][idx] = 0.0f;
          if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && is_boundary[1]) {
            D[2][idx.inc_x()] = 0.0f;
          }
          D[1][idx.dec_y()] = D[1][idx];
        }

        if (pos[1] == grid.dims[1] - grid.guard[1] - 1 && is_boundary[3]) {
          // theta = pi axis
          D[2][idx.inc_y()] = 0.0f;
          if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && is_boundary[1]) {
            D[2][idx.inc_x().inc_y()] = 0.0f;
          }
          D[1][idx.inc_y()] = D[1][idx];
        }

        if (pos[0] == grid.guard[0] && is_boundary[0]) {
          // inner boundary
          D[0][idx.dec_x()] = D[0][idx];
          D[1][idx.dec_x()] = D[1][idx];
          D[2][idx.dec_x()] = D[2][idx];
        }

        if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && is_boundary[1]) {
          // outer boundary
          D[0][idx.inc_x()] = D[0][idx];
          D[1][idx.inc_x()] = D[1][idx];
          D[2][idx.inc_x()] = D[2][idx];
        }
      }
    }
  };

  m_prev_B.copy_from(*(this->B));
  m_prev_D.copy_from(*(this->E));
  // m_new_B.copy_from(*(this->B));
  // m_new_D.copy_from(*(this->E));

  // First pass, predictor values in m_new_B and m_new_D
  kernel_launch(update_B_kernel, m_new_B.get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_ks_grid.get_grid_ptrs());
  kernel_launch(update_D_kernel, m_new_D.get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), this->J->get_const_ptrs(),
                m_ks_grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(m_new_B);
    this->m_comm->send_guard_cells(m_new_D);
  }

  // Second pass, use predictor values and new values in E and B
  kernel_launch(update_B_kernel, this->B->get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_new_D.get_const_ptrs(),
                m_new_B.get_const_ptrs(), m_ks_grid.get_grid_ptrs());
  kernel_launch(update_D_kernel, this->E->get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_new_D.get_const_ptrs(),
                m_new_B.get_const_ptrs(), this->J->get_const_ptrs(),
                m_ks_grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->B));
    this->m_comm->send_guard_cells(*(this->E));
  }

  // Third pass, use E and B and store predictor values in m_new_B and m_new_D
  kernel_launch(update_B_kernel, m_new_B.get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), this->E->get_const_ptrs(),
                this->B->get_const_ptrs(), m_ks_grid.get_grid_ptrs());
  kernel_launch(update_D_kernel, m_new_D.get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), this->E->get_const_ptrs(),
                this->B->get_const_ptrs(), this->J->get_const_ptrs(),
                m_ks_grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(m_new_B);
    this->m_comm->send_guard_cells(m_new_D);
  }

  // Final pass, use m_new_B and m_new_D to generate next timestep
  kernel_launch(update_B_kernel, this->B->get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_new_D.get_const_ptrs(),
                m_new_B.get_const_ptrs(), m_ks_grid.get_grid_ptrs());
  kernel_launch(update_D_kernel, this->E->get_ptrs(), m_prev_D.get_const_ptrs(),
                m_prev_B.get_const_ptrs(), m_new_D.get_const_ptrs(),
                m_new_B.get_const_ptrs(), this->J->get_const_ptrs(),
                m_ks_grid.get_grid_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->B));
    this->m_comm->send_guard_cells(*(this->E));
  }

  CudaCheckError();
}

template <typename Conf>
void
field_solver_gr_ks_cu<Conf>::update(double dt, uint32_t step) {
  iterate_predictor(dt);

  // apply damping boundary condition at outer boundary
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary(*(this->E), *(this->B), *(this->E0), *(this->B0),
                     m_damping_length, m_damping_coef);
  }

  this->Etotal->copy_from(*(this->E));
  this->Btotal->copy_from(*(this->B));

  if (step % this->m_data_interval == 0) {
    compute_divs(*(this->divE), *(this->divB), *(this->E), *(this->B), m_ks_grid);

    compute_flux(*flux, *(this->Btotal), m_ks_grid);
  }

  CudaSafeCall(cudaDeviceSynchronize());
}

template class field_solver_gr_ks_cu<Config<2, float>>;
template class field_solver_gr_ks_cu<Config<2, double>>;

}  // namespace Aperture
