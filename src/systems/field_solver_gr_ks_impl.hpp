/*
 * Copyright (c) 2023 Alex Chen.
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

#include "core/gpu_translation_layer.h"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver_gr_ks.h"
#include "field_solver_base_impl.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "systems/physics/metric_kerr_schild.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

namespace {

template <int Dim>
using fd = finite_diff<Dim, 2>;

template <typename Conf, template <class> class ExecPolicy>
void
damping_boundary(vector_field<Conf>& e, vector_field<Conf>& b,
                 int damping_length, typename Conf::value_t damping_coef) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  // kernel_launch(
  ExecPolicy<Conf>::launch(
      [damping_length, damping_coef] LAMBDA(auto e, auto b) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n1 : grid_stride_range(0, grid.dims[1])) {
        ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
          for (int i = 0; i < damping_length - 1; i++) {
            int n0 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            e[0][idx] *= lambda;
            e[1][idx] *= lambda;
            e[2][idx] *= lambda;
            b[1][idx] *= lambda;
            b[2][idx] *= lambda;
          }
        });
      },
      e, b);
  ExecPolicy<Conf>::sync();
}

}  // namespace

template <typename Conf, template <class> class ExecPolicy>
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::field_solver(
    const grid_ks_t<Conf>& grid, const domain_comm<Conf, ExecPolicy>* comm)
    : field_solver_base<Conf>(grid), m_ks_grid(grid), m_comm(comm) {
  ExecPolicy<Conf>::set_grid(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::~field_solver() {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::init() {
  field_solver_base<Conf>::init();

  sim_env().params().get_value("bh_spin", m_a);
  Logger::print_info("bh_spin in field solver is {}", m_a);
  sim_env().params().get_value("implicit_beta", this->m_beta);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);

  auto type = ExecPolicy<Conf>::data_mem_type();
  m_prev_D = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);
  m_new_D = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);
  m_prev_B = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);
  m_new_B = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);
  // m_prev_D.resize(this->m_grid);
  // m_prev_B.resize(this->m_grid);
  // m_new_D.resize(this->m_grid);
  // m_new_B.resize(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_gr_ks_sph>::register_data_components() {
  auto type = ExecPolicy<Conf>::data_mem_type();
  field_solver_base<Conf>::register_data_components_impl(type);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::iterate_predictor(
    double dt) {
  auto a = m_a;
  auto beta = this->m_beta;
  vec_t<bool, Conf::dim* 2> is_boundary = true;
  if (this->m_comm != nullptr) {
    is_boundary = this->m_comm->domain_info().is_boundary;
  }

  // Iterate all fields at once
  auto update_B_kernel = [dt, a, beta, is_boundary] LAMBDA(
                             auto B, auto prevD, auto prevB, auto nextD,
                             auto nextB, auto grid_ptrs) {
    using namespace Metric_KS;

    auto& grid = ExecPolicy<Conf>::grid();
    auto ext = grid.extent();
    auto alpha = 1.0f - beta;
    ExecPolicy<
        Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
      auto pos = get_pos(idx, ext);
      if (grid.is_in_bound(pos)) {
        // First construct the auxiliary fields E and H
        value_t r_sp =
            grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0] + 1, true));
        value_t r_sm =
            grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], true));

        value_t th_sp =
            grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1] + 1, true));
        value_t th_sm =
            grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], true));

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
    });
  };

  auto update_D_kernel = [dt, a, beta, is_boundary] LAMBDA(
                             auto D, auto prevD, auto prevB, auto nextD,
                             auto nextB, auto J, auto grid_ptrs) {
    using namespace Metric_KS;

    auto& grid = ExecPolicy<Conf>::grid();
    auto ext = grid.extent();
    auto alpha = 1.0f - beta;
    ExecPolicy<
        Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
      // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
      auto pos = get_pos(idx, ext);
      if (grid.is_in_bound(pos)) {
        // First construct the auxiliary fields E and H
        // value_t r =
        //     grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], false));
        value_t r_p =
            grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], false));
        value_t r_m =
            grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0] - 1, false));

        // value_t th =
        //     grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], false));
        value_t th_p =
            grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], false));
        value_t th_m =
            grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1] - 1, false));

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
          D[0][idx.inc_y()] =
              prevD[0][idx.inc_y()] +
              dt * (-2.0f * Hph11) / grid_ptrs.Ad[0][idx.inc_y()] -
              dt * J[0][idx.inc_y()];

          if (pos[0] == grid.guard[0] && is_boundary[0]) {
            D[0][idx.inc_y().dec_x()] = D[0][idx.inc_y()];
          }
          if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && is_boundary[1]) {
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
    });
  };

  m_prev_B->copy_from(*(this->B));
  m_prev_D->copy_from(*(this->E));
  // m_new_B.copy_from(*(this->B));
  // m_new_D.copy_from(*(this->E));

  // First pass, predictor values in m_new_B and m_new_D
  ExecPolicy<Conf>::launch(update_B_kernel, *m_new_B, *m_prev_D, *m_prev_B,
                           *m_prev_D, *m_prev_B, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::launch(update_D_kernel, *m_new_D, *m_prev_D, *m_prev_B,
                           *m_prev_D, *m_prev_B, *(this->J),
                           m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*m_new_B);
    this->m_comm->send_guard_cells(*m_new_D);
  }

  // Second pass, use predictor values and new values in E and B
  ExecPolicy<Conf>::launch(update_B_kernel, *(this->B), *m_prev_D, *m_prev_B,
                           *m_new_D, *m_new_B, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::launch(update_D_kernel, *(this->E), *m_prev_D, *m_prev_B,
                           *m_new_D, *m_new_B, *(this->J),
                           m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->B));
    this->m_comm->send_guard_cells(*(this->E));
  }

  // Third pass, use E and B and store predictor values in m_new_B and m_new_D
  ExecPolicy<Conf>::launch(update_B_kernel, *m_new_B, *m_prev_D, *m_prev_B,
                           *(this->E), *(this->B), m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::launch(update_D_kernel, *m_new_D, *m_prev_D, *m_prev_B,
                           *(this->E), *(this->B), *(this->J),
                           m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*m_new_B);
    this->m_comm->send_guard_cells(*m_new_D);
  }

  // Final pass, use m_new_B and m_new_D to generate next timestep
  ExecPolicy<Conf>::launch(update_B_kernel, *(this->B), *m_prev_D, *m_prev_B,
                           *m_new_D, *m_new_B, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::launch(update_D_kernel, *(this->E), *m_prev_D, *m_prev_B,
                           *m_new_D, *m_new_B, *(this->J),
                           m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->B));
    this->m_comm->send_guard_cells(*(this->E));
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::update_semi_implicit(
    double dt, double alpha, double beta, double time) {
  iterate_predictor(dt);

  // apply damping boundary condition at outer boundary
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary<Conf, ExecPolicy>(*(this->E), *(this->B), m_damping_length,
                                       m_damping_coef);
  }

  // this->Etotal->copy_from(*(this->E));
  // this->Btotal->copy_from(*(this->B));
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::update_explicit(
    double dt, double time) {
  update_semi_implicit(dt, 1.0 - this->m_beta, this->m_beta, time);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_divs_e_b() {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_flux() {
  this->flux->init();
  auto a = m_a;
  ExecPolicy<Conf>::launch(
      [a] LAMBDA (auto flux, auto b, auto grid_ptrs) {
        auto &grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
          auto r = grid_ks_t<Conf>::radius(grid.template coord<0>(n0, true));

          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            Scalar th = grid_ks_t<Conf>::theta(grid.template coord<1>(n1, false));
            Scalar th_p =
                grid_ks_t<Conf>::theta(grid.template coord<1>(n1 + 1, true));
            Scalar th_m =
                grid_ks_t<Conf>::theta(grid.template coord<1>(n1, true));
            auto dth = th_p - th_m;

            auto pos = index_t<Conf::dim>(n0, n1);
            auto idx = typename Conf::idx_t(pos, ext);

            flux[idx] = flux[idx.dec_y()] +
                        // b[0][idx] * Metric_KS::sqrt_gamma(a, r, th) * dth;
                        b[0][idx] * grid_ptrs.Ab[0][idx];
          }
        });

      }, this->flux, this->B, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_EB_sqr() {}

}  // namespace Aperture
