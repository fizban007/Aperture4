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
#include "field_solver_base_impl.hpp"
#include "field_solver_gr_ks.h"
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
                 vector_field<Conf>& e0, vector_field<Conf>& b0,
                 int damping_length, typename Conf::value_t damping_coef,
                 bool damp_to_background = true) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  // kernel_launch(
  ExecPolicy<Conf>::launch(
      [damping_length, damping_coef, damp_to_background] 
        LAMBDA(auto e, auto e0, auto b, auto b0) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n1 : grid_stride_range(0, grid.dims[1])) {
        ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
          for (int i = 0; i < damping_length - 1; i++) {
            int n0 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            if (damp_to_background) {
              e[0][idx] *= lambda;
              e[1][idx] *= lambda;
              e[2][idx] *= lambda;
              b[0][idx] *= lambda;
              b[1][idx] *= lambda;
              b[2][idx] *= lambda;
            } else {
              e[0][idx] = lambda * (e[0][idx] + e0[0][idx]) - e0[0][idx];
              e[1][idx] = lambda * (e[1][idx] + e0[1][idx]) - e0[1][idx];
              e[2][idx] = lambda * (e[2][idx] + e0[2][idx]) - e0[2][idx];
              b[0][idx] = lambda * (b[0][idx] + b0[0][idx]) - b0[0][idx];
              b[1][idx] = lambda * (b[1][idx] + b0[1][idx]) - b0[1][idx];
              b[2][idx] = lambda * (b[2][idx] + b0[2][idx]) - b0[2][idx];
            }
          }
        });
      },
      e, e0, b, b0);
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
  sim_env().params().get_value("damp_to_background", m_damp_to_background);

  auto type = ExecPolicy<Conf>::data_mem_type();
  m_dD_dt = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);
  m_dB_dt = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);

  m_tmpD = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);
  m_tmpB = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);

  m_tmpdD_dt = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);
  m_dD_dt_prev3 = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::edge_centered, type);

  m_tmpdB_dt = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);
  m_dB_dt_prev3 = std::make_unique<vector_field<Conf>>(
      this->m_grid, field_type::face_centered, type);

  // m_auxE = std::make_unique<vector_field<Conf>>(
  //     this->m_grid, field_type::edge_centered, type);
  // m_auxH = std::make_unique<vector_field<Conf>>(
  //     this->m_grid, field_type::face_centered, type);
  m_auxE = sim_env().register_data<vector_field<Conf>>(
      "auxE", this->m_grid, field_type::edge_centered, type);
  m_auxH = sim_env().register_data<vector_field<Conf>>(
      "auxH", this->m_grid, field_type::face_centered, type);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_gr_ks_sph>::register_data_components() {
  auto type = ExecPolicy<Conf>::data_mem_type();
  field_solver_base<Conf>::register_data_components_impl(type);

  DdotB = sim_env().register_data<scalar_field<Conf>>(
      "DdotB", this->m_grid, field_type::cell_centered, type);
  Bmag = sim_env().register_data<scalar_field<Conf>>(
      "Bmag", this->m_grid, field_type::cell_centered, type);
  Jmag = sim_env().register_data<scalar_field<Conf>>(
      "Jmag", this->m_grid, field_type::vert_centered, type);
  // Sigma = sim_env().register_data<scalar_field<Conf>>(
  //     "sigma", this->m_grid, field_type::cell_centered, type);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_aux_E(
    const vector_field<Conf>& B, const vector_field<Conf>& D) {
  auto a = m_a;
  m_auxE->init();
  using value_t = typename Conf::value_t;

  ExecPolicy<Conf>::launch(
      [a] LAMBDA(auto B, auto D, auto auxE, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();

        ExecPolicy<
            Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
          auto pos = get_pos(idx, ext);
          // if (grid.is_in_bound(pos)) {
          if (pos[0] > 0 && pos[0] < grid.dims[0] - 1) {
            value_t r = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], false));
            value_t r_plus = grid_ks_t<Conf>::radius(
                grid.coord(0, pos[0], false) + 0.5f * grid.delta[0]);
            value_t r_minus = grid_ks_t<Conf>::radius(
                grid.coord(0, pos[0], false) - 0.5f * grid.delta[0]);
            value_t th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));

            // Handle coordinate axis
            if (math::abs(th) < 0.1f * grid.delta[1] ||
                math::abs(th - M_PI) < 0.1f * grid.delta[1]) {
              auxE[0][idx] = grid_ptrs.ag11dr_e[idx] * D[0][idx];
              // auxE[0][idx] =
                  // Metric_KS::ag_11(a, r, th) * D[0][idx] * r * grid.delta[0];
              // if (pos[0] == 100 && pos[1] == grid.N[1] + grid.guard[1]) {
              //   printf("auxE0 is %f, D0 is %f\n", auxE[0][idx], D[0][idx]);
              // }
            } else {
              auxE[0][idx] =
                  // Metric_KS::ag_11(a, r, th) * D[0][idx] * r * grid.delta[0] +
                  grid_ptrs.ag11dr_e[idx] * D[0][idx] +
                  // 0.5 *
                  //     (Metric_KS::ag_13(a, r_minus, th) * D[2][idx] * r_minus +
                  //      Metric_KS::ag_13(a, r_plus, th) * D[2][idx.inc_x()] *
                  //          r_plus) *
                  //     grid.delta[0];
                  0.5 * (grid_ptrs.ag13dr_d[idx] * D[2][idx] +
                         grid_ptrs.ag13dr_d[idx.inc_x()] * D[2][idx.inc_x()]);
              // 0.5 * grid_ptrs.ag13dr_e[idx] * (D[2][idx] +
              // D[2][idx.inc_x()]); 0.5 * Metric_KS::ag_13(a, r, th) *
              // (D[2][idx] + D[2][idx.inc_x()]);
            }

            r_plus = r;
            r = r_minus;
            r_minus = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], true) -
                                              0.5f * grid.delta[0]);
            th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], false));

            auxE[1][idx] =
                // Metric_KS::ag_22(a, r, th) * D[1][idx] * grid.delta[1] -
                    grid_ptrs.ag22dth_e[idx] * D[1][idx] -
                // 0.5 *
                //     (Metric_KS::sq_gamma_beta(a, r_plus, th) * B[2][idx] +
                //      Metric_KS::sq_gamma_beta(a, r_minus, th) *
                //          B[2][idx.dec_x()]) *
                //     grid.delta[1];
            0.5 * (grid_ptrs.gbetadth_b[idx] * B[2][idx] +
                   grid_ptrs.gbetadth_b[idx.dec_x()] *
                   B[2][idx.dec_x()]);
            // 0.5 * grid_ptrs.gbetadth_e[idx] * (B[2][idx] +
            // B[2][idx.dec_x()]); 0.5 * Metric_KS::sq_gamma_beta(a, r, th)
            // * (B[2][idx] + B[2][idx.dec_x()]);

            th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));

            // Handle coordinate axis
            if (math::abs(th) < 0.1f * grid.delta[1] ||
                math::abs(th - M_PI) < 0.1f * grid.delta[1]) {
              auxE[2][idx] =
                  0.5 * (Metric_KS::ag_13(a, r_plus, th) * D[0][idx] +
                         Metric_KS::ag_13(a, r_minus, th) * D[0][idx.dec_x()]);
              // 0.5 * Metric_KS::ag_13(a, r, th) * (D[0][idx] +
              // D[0][idx.dec_x()]);
            } else {
              auxE[2][idx] =
                  Metric_KS::ag_33(a, r, th) * D[2][idx] +
                  0.5 * (Metric_KS::ag_13(a, r_plus, th) * D[0][idx] +
                         Metric_KS::ag_13(a, r_minus, th) * D[0][idx.dec_x()]) +
                  0.5 * (Metric_KS::sq_gamma_beta(a, r_plus, th) * B[1][idx] +
                         Metric_KS::sq_gamma_beta(a, r_minus, th) *
                             B[1][idx.dec_x()]);
              // 0.5 * Metric_KS::ag_13(a, r, th) * (D[0][idx] +
              // D[0][idx.dec_x()]) + 0.5 * Metric_KS::sq_gamma_beta(a, r,
              // th) * (B[1][idx] + B[1][idx.dec_x()]);
            }
          }
        });
      },
      B, D, *(m_auxE), m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_aux_H(
    const vector_field<Conf>& B, const vector_field<Conf>& D) {
  auto a = m_a;
  m_auxH->init();

  ExecPolicy<Conf>::launch(
      [a] LAMBDA(auto B, auto D, auto auxH, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();

        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              // if (grid.is_in_bound(pos)) {
              if (pos[0] > 0 && pos[0] < grid.dims[0] - 1) {
                value_t r =
                    grid_ks_t<Conf>::radius(grid.coord(0, pos[0], true));
                value_t r_plus = grid_ks_t<Conf>::radius(
                    grid.coord(0, pos[0], true) + 0.5f * grid.delta[0]);
                value_t r_minus = grid_ks_t<Conf>::radius(
                    grid.coord(0, pos[0], true) - 0.5f * grid.delta[0]);
                value_t th =
                    grid_ks_t<Conf>::theta(grid.coord(1, pos[1], false));

                auxH[0][idx] =
                    grid_ptrs.ag11dr_h[idx] * B[0][idx] +
                    // Metric_KS::ag_11(a, r, th) * B[0][idx] * r * grid.delta[0] +
                    // 0.5 *
                    //     (Metric_KS::ag_13(a, r_minus, th) * B[2][idx.dec_x()] *
                    //          r_minus +
                    //      Metric_KS::ag_13(a, r_plus, th) * B[2][idx] * r_plus) *
                    //     grid.delta[0];
                0.5 * (grid_ptrs.ag13dr_b[idx.dec_x()] * B[2][idx.dec_x()] +
                       grid_ptrs.ag13dr_b[idx] * B[2][idx]);
                // 0.5 * grid_ptrs.ag13dr_h[idx] * (B[2][idx.dec_x()] +
                // B[2][idx]); 0.5 * Metric_KS::ag_13(a, r, th) *
                // (B[2][idx.dec_x()] + B[2][idx]);

                r_minus = r;
                r = r_plus;
                r_plus = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], false) +
                                                 0.5f * grid.delta[0]);
                th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));

                // Handle coordinate axis
                if (math::abs(th) < 0.1f * grid.delta[1] ||
                    math::abs(th - M_PI) < 0.1f * grid.delta[1]) {
                  auxH[1][idx] = 0.0f;
                } else {
                  auxH[1][idx] =
                      grid_ptrs.ag22dth_h[idx] * B[1][idx] +
                      // Metric_KS::ag_22(a, r, th) * B[1][idx] * grid.delta[1] +
                      // 0.5 *
                      //     (Metric_KS::sq_gamma_beta(a, r_plus, th) *
                      //          D[2][idx.inc_x()] +
                      //      Metric_KS::sq_gamma_beta(a, r_minus, th) *
                      //          D[2][idx]) *
                      //     grid.delta[1];
                  0.5 * (grid_ptrs.gbetadth_d[idx.inc_x()] *
                             D[2][idx.inc_x()] +
                         grid_ptrs.gbetadth_d[idx] *
                             D[2][idx]);
                  // 0.5 * grid_ptrs.gbetadth_h[idx] * (D[2][idx.inc_x()] +
                  // D[2][idx]); 0.5 * Metric_KS::sq_gamma_beta(a, r, th) *
                  // (D[2][idx.inc_x()] + D[2][idx]);
                }

                th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], false));

                auxH[2][idx] =
                    Metric_KS::ag_33(a, r, th) * B[2][idx] +
                    0.5 * (Metric_KS::ag_13(a, r_plus, th) * B[0][idx.inc_x()] +
                           Metric_KS::ag_13(a, r_minus, th) * B[0][idx]) -
                    0.5 *
                        (Metric_KS::sq_gamma_beta(a, r_plus, th) *
                             D[1][idx.inc_x()] +
                         Metric_KS::sq_gamma_beta(a, r_minus, th) * D[1][idx]);
                // 0.5 * Metric_KS::ag_13(a, r, th) * (B[0][idx] +
                // B[0][idx.inc_x()]) + 0.5 * Metric_KS::sq_gamma_beta(a, r, th)
                // * (D[1][idx] + D[1][idx.inc_x()]);
              }
            });
      },
      B, D, *(m_auxH), m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_dB_dt(
    vector_field<Conf>& dB_dt, const vector_field<Conf>& B,
    const vector_field<Conf>& D) {
  vec_t<bool, Conf::dim * 2> is_boundary = true;
  if (this->m_comm != nullptr) {
    is_boundary = this->m_comm->domain_info().is_boundary;
  }
  dB_dt.init();
  compute_aux_E(B, D);

  using namespace Metric_KS;

  ExecPolicy<Conf>::launch(
      [is_boundary] LAMBDA(auto auxE, auto dB_dt, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();

        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                // dB_r / dt. dphi is taken to be 1
                dB_dt[0][idx] = -(auxE[2][idx.inc_y()] - auxE[2][idx]) /
                                grid_ptrs.Ab[0][idx];

                // dB_th / dt. dphi is taken to be 1
                dB_dt[1][idx] = -(auxE[2][idx] - auxE[2][idx.inc_x()]) /
                                grid_ptrs.Ab[1][idx];

                // dB_phi / dt
                dB_dt[2][idx] = -((auxE[1][idx.inc_x()] - auxE[1][idx]) +
                                  (auxE[0][idx] - auxE[0][idx.inc_y()])) /
                                grid_ptrs.Ab[2][idx];

                // if (pos[0] == 100 && pos[1] == grid.N[1] + grid.guard[1] - 1) {
                //   printf("dBphi/dt is %f, auxE0 are %f and %f\n", dB_dt[2][idx],
                //          auxE[0][idx], auxE[0][idx.inc_y()]);
                // }

                // Boundary conditions
                if (pos[1] == grid.guard[1] && is_boundary[2]) {
                  // theta = 0 axis
                  dB_dt[1][idx] = 0.0f;
                }

                if (pos[1] == grid.dims[1] - grid.guard[1] - 1 &&
                    is_boundary[3]) {
                  // theta = pi axis
                  dB_dt[1][idx.inc_y()] = 0.0f;
                }
              }
            });
      },
      *(m_auxE), dB_dt, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_dD_dt(
    vector_field<Conf>& dD_dt, const vector_field<Conf>& B,
    const vector_field<Conf>& D, const vector_field<Conf>& J) {
  vec_t<bool, Conf::dim* 2> is_boundary = true;
  if (this->m_comm != nullptr) {
    is_boundary = this->m_comm->domain_info().is_boundary;
  }
  // for (int i = 0; i < Conf::dim * 2; i++) {
  //   printf("is_boundary[%d] is %d, expected %d\n", i, is_boundary[i],
  //          this->m_comm->domain_info().is_boundary[i]);
  // }
  dD_dt.init();
  compute_aux_H(B, D);

  using namespace Metric_KS;
  ExecPolicy<Conf>::launch(
      [is_boundary] LAMBDA(auto auxH, auto J, auto dD_dt, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                value_t th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));
                if (pos[1] == grid.guard[1] && 
                    math::abs(th) < 0.1 * grid.delta[1]) {
                  dD_dt[0][idx] =
                      (auxH[2][idx]) / grid_ptrs.Ad[0][idx]
                      -J[0][idx];
                } else {
                  dD_dt[0][idx] = (auxH[2][idx] - auxH[2][idx.dec_y()]) /
                                      grid_ptrs.Ad[0][idx] -
                                  J[0][idx];
                  // dD_dt[0][idx] = -J[0][idx];
                }
                // At theta = pi boundary, do an additional update
                if (pos[1] == grid.N[1] + grid.guard[1] - 1 &&
                    math::abs(th + grid.delta[1] - M_PI) < 0.1 * grid.delta[1]) {
                  dD_dt[0][idx.inc_y()] =
                      (-auxH[2][idx]) / grid_ptrs.Ad[0][idx.inc_y()]
                      - J[0][idx.inc_y()];
                  // if (pos[0] == 100) {
                  //   printf("pi boundary for D0!\n");
                  // }
                  // if (pos[0] == grid.guard[0] && is_boundary[0]) {
                  //   D[0][idx.inc_y().dec_x()] = D[0][idx.inc_y()];
                  // }
                  // if (pos[0] == grid.dims[0] - grid.guard[0] - 1 &&
                  // is_boundary[1]) {
                  //   D[0][idx.inc_y().inc_x()] = D[0][idx.inc_y()];
                  // }
                }

                dD_dt[1][idx] = (auxH[2][idx.dec_x()] - auxH[2][idx]) /
                                    grid_ptrs.Ad[1][idx] -
                                J[1][idx];
                // dD_dt[1][idx] = -J[1][idx];
                // if (pos[1] == grid.guard[1] && 
                //     math::abs(th) < 0.1 * grid.delta[1] &&
                //     (pos[0] == 40 || pos[0] == 39)) {
                //   printf("At %d, %d, D0 is %f, D1 is %f\n", pos[0], pos[1], dD_dt[0][idx], dD_dt[1][idx]);
                // }

                if (pos[1] == grid.guard[1] && is_boundary[2]) {
                  dD_dt[2][idx] = 0.0f;
                } else {
                  dD_dt[2][idx] = ((auxH[1][idx] - auxH[1][idx.dec_x()]) +
                                   (auxH[0][idx.dec_y()] - auxH[0][idx])) /
                                      grid_ptrs.Ad[2][idx] -
                                  J[2][idx];
                }
                // At theta = pi boundary, do an additional update
                if (pos[1] == grid.dims[1] - grid.guard[1] - 1 &&
                    is_boundary[3]) {
                  dD_dt[2][idx.inc_y()] = 0.0f;
                }
                // Boundary conditions
                // if (pos[1] == grid.guard[1] && is_boundary[2]) {
                //   // theta = 0 axis
                //   dD_dt[2][idx] = 0.0f;
                //   if (pos[0] == grid.dims[0] - grid.guard[0] - 1 &&
                //       is_boundary[1]) {
                //     dD_dt[2][idx.inc_x()] = 0.0f;
                //   }
                //   // D[1][idx.dec_y()] = D[1][idx];
                // }

                // if (pos[1] == grid.dims[1] - grid.guard[1] - 1 &&
                // is_boundary[3]) {
                //   // theta = pi axis
                //   dD_dt[2][idx.inc_y()] = 0.0f;
                //   if (pos[0] == grid.dims[0] - grid.guard[0] - 1 &&
                //       is_boundary[1]) {
                //     dD_dt[2][idx.inc_x().inc_y()] = 0.0f;
                //   }
                //   // D[1][idx.inc_y()] = D[1][idx];
                // }

                // if (pos[0] == grid.guard[0] && is_boundary[0]) {
                //   // inner boundary
                //   D[0][idx.dec_x()] = D[0][idx];
                //   D[1][idx.dec_x()] = D[1][idx];
                //   D[2][idx.dec_x()] = D[2][idx];
                // }

                // if (pos[0] == grid.dims[0] - grid.guard[0] - 1 &&
                // is_boundary[1])
                // {
                //   // outer boundary
                //   D[0][idx.inc_x()] = D[0][idx];
                //   D[1][idx.inc_x()] = D[1][idx];
                //   D[2][idx.inc_x()] = D[2][idx];
                // }
              }
            });
      },
      *(m_auxH), J, dD_dt, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::iterate_predictor(
    double dt) {
  // The following implements the 2nd order predictor-corrector scheme, aka Heun
  // method

  // Compute the RHS at the current time step n
  compute_dB_dt(*m_dB_dt, *(this->B), *(this->E));
  compute_dD_dt(*m_dD_dt, *(this->B), *(this->E), *(this->J));

  // Construct the Euler estimate for the next time step n+1
  m_tmpB->copy_from(*(this->B));
  m_tmpD->copy_from(*(this->E));
  m_tmpB->add_by(*m_dB_dt, dt);
  m_tmpD->add_by(*m_dD_dt, dt);

  // Communicate if necessary
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*m_tmpB);
    this->m_comm->send_guard_cells(*m_tmpD);
  }

  boundary_conditions(*m_tmpD, *m_tmpB);

  // Iterate EC several times
  for (int i = 0; i < 7; i++) {
    // Compute the RHS using new values at n+1
    compute_dB_dt(*m_tmpdB_dt, *m_tmpB, *m_tmpD);
    compute_dD_dt(*m_tmpdD_dt, *m_tmpB, *m_tmpD, *(this->J));

    // Set new B and D at n+1
    m_tmpB->copy_from(*(this->B));
    m_tmpD->copy_from(*(this->E));
    // m_tmpB->add_by(*m_tmpdB_dt, dt * 0.5f);
    // m_tmpB->add_by(*m_dB_dt, dt * 0.5f);
    // m_tmpD->add_by(*m_tmpdD_dt, dt * 0.5f);
    // m_tmpD->add_by(*m_dD_dt, dt * 0.5f);
    m_tmpB->add_by(*m_tmpdB_dt, dt * this->m_beta);
    m_tmpB->add_by(*m_dB_dt, dt * this->m_alpha);
    m_tmpD->add_by(*m_tmpdD_dt, dt * this->m_beta);
    m_tmpD->add_by(*m_dD_dt, dt * this->m_alpha);

    // Communicate the result
    if (this->m_comm != nullptr) {
      this->m_comm->send_guard_cells(*m_tmpB);
      this->m_comm->send_guard_cells(*m_tmpD);
    }

    boundary_conditions(*m_tmpD, *m_tmpB);
  }

  this->E->copy_from(*m_tmpD);
  this->B->copy_from(*m_tmpB);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::boundary_conditions(
    vector_field<Conf>& D, vector_field<Conf>& B) {
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[0]) {
  // if (false) {
    // Inner boundary inside horizon
    ExecPolicy<Conf>::launch(
        [] LAMBDA(auto D, auto B, auto D0, auto B0) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();

          ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
            auto pos = index_t<Conf::dim>(grid.guard[0], n1);
            auto idx = typename Conf::idx_t(pos, ext);
            D[0][idx.dec_x()] = D[0][idx] + D0[0][idx] - D0[0][idx.dec_x()];
            D[1][idx.dec_x()] = D[1][idx] + D0[1][idx] - D0[1][idx.dec_x()];
            D[2][idx.dec_x()] = D[2][idx] + D0[2][idx] - D0[2][idx.dec_x()];
            B[0][idx.dec_x()] = B[0][idx] + B0[0][idx] - B0[0][idx.dec_x()];
            B[1][idx.dec_x()] = B[1][idx] + B0[1][idx] - B0[1][idx.dec_x()];
            B[2][idx.dec_x()] = B[2][idx] + B0[2][idx] - B0[2][idx.dec_x()];
            // B[0][idx.dec_x()] = -B0[0][idx.dec_x()];
            // B[1][idx.dec_x()] = -B0[1][idx.dec_x()];
            // B[2][idx.dec_x()] = -B0[2][idx.dec_x()];
            // D[0][idx.dec_x()] = -D0[0][idx.dec_x()];
            // D[1][idx.dec_x()] = -D0[1][idx.dec_x()];
            // D[2][idx.dec_x()] = -D0[2][idx.dec_x()];
            // B[1][idx] = -B0[1][idx];
            // B[2][idx] = -B0[2][idx];
          });
        },
        D, B, this->E0, this->B0);
    ExecPolicy<Conf>::sync();
  }
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[2]) {
    // Axis at theta = 0
    ExecPolicy<Conf>::launch(
        [] LAMBDA(auto D, auto B) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();

          ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
            auto pos = index_t<Conf::dim>(n0, grid.guard[1]);
            auto idx = typename Conf::idx_t(pos, ext);
            // D[0][idx.dec_y()] = D[0][idx.inc_y()];
            // D[1][idx.dec_x()] = D[1][idx];
            D[2][idx] = 0.0f;
            B[2][idx.dec_y()] = -B[2][idx];
            // B[2][idx] = 0.0f;
            B[1][idx] = 0.0f;
            // B[2][idx.dec_x()] = B[2][idx];
          });
        },
        D, B);
    ExecPolicy<Conf>::sync();
  }
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[3]) {
    // Axis at theta = pi
    ExecPolicy<Conf>::launch(
        [] LAMBDA(auto D, auto B) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();

          ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
            auto pos = index_t<Conf::dim>(n0, grid.dims[1] - grid.guard[1] - 1);
            auto idx = typename Conf::idx_t(pos, ext);
            // D[0][idx.inc_y(2)] = D[0][idx];
            // D[1][idx.dec_x()] = D[1][idx];
            D[2][idx.inc_y()] = 0.0f;
            B[2][idx.inc_y()] = -B[2][idx];
            // B[2][idx] = 0.0f;
            B[1][idx.inc_y()] = 0.0f;
            // B[2][idx.dec_x()] = B[2][idx];
          });
        },
        D, B);
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::update_semi_implicit(
    double dt, double alpha, double beta, double time) {
  iterate_predictor(dt);

  // apply damping boundary condition at outer boundary

  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1])
  {
    damping_boundary<Conf, ExecPolicy>(*(this->E), *(this->B),
                                       *(this->E0), *(this->B0),
                                       m_damping_length, m_damping_coef,
                                       m_damp_to_background);
  }
  // this->Etotal->copy_from(*(this->E));
  // this->Btotal->copy_from(*(this->B));
  compute_DdotB_J_B();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::update_explicit(
    double dt, double time) {
  update_semi_implicit(dt, 1.0 - this->m_beta, this->m_beta, time);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_divs_e_b() {
  this->divE->init();
  this->divB->init();
  auto a = m_a;
  // if (this->m_comm != nullptr) {
  //   this->m_comm->send_guard_cells(*(this->B));
  //   this->m_comm->send_guard_cells(*(this->E));
  // }

  ExecPolicy<Conf>::launch(
      [a] LAMBDA(auto d, auto b, auto divD, auto divB, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            auto r = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], false));
            auto r_s = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], true));
            auto th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], false));
            auto th_s = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));

            divB[idx] = (b[0][idx.inc_x()] * grid_ptrs.Ab[0][idx.inc_x()] -
                         b[0][idx] * grid_ptrs.Ab[0][idx] +
                         b[1][idx.inc_y()] * grid_ptrs.Ab[1][idx.inc_y()] -
                         b[1][idx] * grid_ptrs.Ab[1][idx]) /
                grid_ptrs.Ab[2][idx];
            divD[idx] = (d[0][idx] * grid_ptrs.Ad[0][idx] -
                         d[0][idx.dec_x()] * grid_ptrs.Ad[0][idx.dec_x()] +
                         d[1][idx] * grid_ptrs.Ad[1][idx] -
                         d[1][idx.dec_y()] * grid_ptrs.Ad[1][idx.dec_y()]) /
                grid_ptrs.Ad[2][idx];
            if (pos[1] == grid.guard[1] &&
                math::abs(th_s) < 0.1 * grid.delta[1]) {
              divD[idx] = (d[0][idx] * grid_ptrs.Ad[0][idx] -
                           d[0][idx.dec_x()] * grid_ptrs.Ad[0][idx.dec_x()] +
                           d[1][idx] * grid_ptrs.Ad[1][idx]) /
                  grid_ptrs.Ad[2][idx];
            }
            if (pos[1] == grid.N[1] + grid.guard[1] - 1 &&
                math::abs(th_s + grid.delta[1] - M_PI) < 0.1 * grid.delta[1]) {
              divD[idx.inc_y()] = (d[0][idx.inc_y()] * grid_ptrs.Ad[0][idx.inc_y()] -
                          d[0][idx.inc_y().dec_x()] * grid_ptrs.Ad[0][idx.inc_y().dec_x()] -
                          d[1][idx] * grid_ptrs.Ad[1][idx]) /
                  grid_ptrs.Ad[2][idx.inc_y()];
            }
          }
        });
      }, this->E, this->B, this->divE, this->divB, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->divB));
    this->m_comm->send_guard_cells(*(this->divE));
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_flux() {
  this->flux->init();
  auto a = m_a;
  ExecPolicy<Conf>::launch(
      [a] LAMBDA(auto flux, auto b, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
          auto r = grid_ks_t<Conf>::radius(grid.template coord<0>(n0, true));

          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            Scalar th =
                grid_ks_t<Conf>::theta(grid.template coord<1>(n1, false));
            Scalar th_p =
                grid_ks_t<Conf>::theta(grid.template coord<1>(n1 + 1, true));
            Scalar th_m =
                grid_ks_t<Conf>::theta(grid.template coord<1>(n1, true));
            // auto dth = th_p - th_m;

            auto pos = index_t<Conf::dim>(n0, n1);
            auto idx = typename Conf::idx_t(pos, ext);

            flux[idx] = flux[idx.dec_y()] +
                        // b[0][idx] * Metric_KS::sqrt_gamma(a, r, th) * dth;
                        b[0][idx] * grid_ptrs.Ab[0][idx];
          }
        });

      },
      this->flux, this->Btotal, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_EB_sqr() {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>::compute_DdotB_J_B() {
  DdotB->init();
  Bmag->init();
  Jmag->init();
  auto a = m_a;
  ExecPolicy<Conf>::launch(
      [a] LAMBDA(auto DdotB, auto Bmag, auto Jmag, auto e, auto b, auto j, auto grid_ptrs) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
          auto pos = get_pos(idx, ext);
          auto r = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], false));
          auto r_s = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], true));
          auto th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], false));
          auto th_s = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], true));

          auto sth = std::sin(th_s);
          auto cth = std::cos(th_s);

          vec_t<value_t, 3> e_vec{e[0][idx], e[1][idx], e[2][idx]};
          vec_t<value_t, 3> b_vec{b[0][idx], b[1][idx], b[2][idx]};
          vec_t<value_t, 3> j_vec{j[0][idx], j[1][idx], j[2][idx]};

          DdotB[idx] = Metric_KS::dot_product_u(e_vec, b_vec, a, r, sth, cth);
          Bmag[idx] = math::sqrt(Metric_KS::dot_product_u(b_vec, b_vec, a, r, sth, cth));
          Jmag[idx] = math::sqrt(Metric_KS::dot_product_u(j_vec, j_vec, a, r, sth, cth));
        });
      },
      DdotB, Bmag, Jmag, this->Etotal, this->Btotal, this->J, m_ks_grid.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

}  // namespace Aperture
