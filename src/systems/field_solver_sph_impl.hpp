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

#include "field_solver_sph.h"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset.hpp"
#include "core/ndsubset_dev.hpp"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/double_buffer.h"

namespace Aperture {

namespace {

template <int Dim>
using fd = finite_diff<Dim, 2>;

template <typename Conf, template <class> class ExecPolicy>
void
compute_double_circ(vector_field<Conf>& result, const vector_field<Conf>& b,
                    const grid_sph_t<Conf>& grid, typename Conf::value_t coef) {
  if constexpr (Conf::dim == 2) {
    auto ext = grid.extent();
    // kernel_launch(
    ExecPolicy<Conf>::launch(
        [coef, ext] LAMBDA(auto result, auto b, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          // for (auto n : grid_stride_range(0, ext.size())) {
          ExecPolicy<Conf>::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                // auto idx = typename Conf::idx_t(n, ext);
                // auto pos = idx.get_pos();
                auto pos = get_pos(idx, ext);
                if (grid.is_in_bound(pos)) {
                  auto idx_mx = idx.dec_x();
                  auto idx_my = idx.dec_y();
                  auto idx_px = idx.inc_x();
                  auto idx_py = idx.inc_y();
                  auto idx_pymx = idx.inc_y().dec_x();
                  auto idx_pxmy = idx.inc_x().dec_y();
                  result[0][idx] =
                      coef *
                      (gp.le[2][idx_py] *
                           fd<Conf::dim>::circ2(b, gp.lb, idx_pymx, idx, idx_py,
                                                idx_py) /
                           gp.Ae[2][idx_py] -
                       gp.le[2][idx] *
                           fd<Conf::dim>::circ2(b, gp.lb, idx_mx, idx_my, idx,
                                                idx) /
                           gp.Ae[2][idx]) /
                      gp.Ab[0][idx];

                  result[1][idx] =
                      coef *
                      (gp.le[2][idx] *
                           fd<Conf::dim>::circ2(b, gp.lb, idx_mx, idx_my, idx,
                                                idx) /
                           gp.Ae[2][idx] -
                       gp.le[2][idx_px] *
                           fd<Conf::dim>::circ2(b, gp.lb, idx, idx_pxmy, idx_px,
                                                idx_px) /
                           gp.Ae[2][idx_px]) /
                      gp.Ab[1][idx];
                  // Take care of axis boundary
                  Scalar theta = grid.template coord<1>(pos[1], true);
                  if (abs(theta) < 0.1 * grid.delta[1]) {
                    result[1][idx] = 0.0f;
                  }

                  result[2][idx] =
                      coef *
                      (gp.le[0][idx] *
                           fd<Conf::dim>::circ0(b, gp.lb, idx_my, idx) /
                           gp.Ae[0][idx_py] -
                       gp.le[0][idx_py] *
                           fd<Conf::dim>::circ0(b, gp.lb, idx, idx_py) /
                           gp.Ae[0][idx] +
                       gp.le[1][idx_px] *
                           fd<Conf::dim>::circ1(b, gp.lb, idx, idx_px) /
                           gp.Ae[1][idx_px] -
                       gp.le[1][idx] *
                           fd<Conf::dim>::circ1(b, gp.lb, idx_mx, idx) /
                           gp.Ae[1][idx]) /
                      gp.Ab[2][idx];
                }
              });
        },
        result, b, grid.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_implicit_rhs(vector_field<Conf>& result, const vector_field<Conf>& e,
                     const vector_field<Conf>& j, const grid_sph_t<Conf>& grid,
                     typename Conf::value_t alpha, typename Conf::value_t beta,
                     typename Conf::value_t dt) {
  if constexpr (Conf::dim == 2) {
    auto ext = grid.extent();
    // kernel_launch(
    ExecPolicy<Conf>::launch(
        [alpha, beta, dt, ext] LAMBDA(auto result, auto e, auto j, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          // gp is short for grid_ptrs
          // for (auto n : grid_stride_range(0, ext.size())) {
          ExecPolicy<Conf>::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                // auto idx = result[0].idx_at(n, ext);
                auto pos = get_pos(idx, ext);
                if (grid.is_in_bound(pos)) {
                  auto idx_py = idx.inc_y();
                  result[0][idx] +=
                      -dt *
                      (fd<Conf::dim>::circ0(e, gp.le, idx, idx_py) -
                       dt * beta *
                           fd<Conf::dim>::circ0(j, gp.le, idx, idx_py)) /
                      gp.Ab[0][idx];

                  auto idx_px = idx.inc_x();
                  result[1][idx] +=
                      -dt *
                      (fd<Conf::dim>::circ1(e, gp.le, idx, idx_px) -
                       dt * beta *
                           fd<Conf::dim>::circ1(j, gp.le, idx, idx_px)) /
                      gp.Ab[1][idx];

                  // Take care of axis boundary
                  Scalar theta = grid.template coord<1>(pos[1], true);
                  if (abs(theta) < 0.1 * grid.delta[1]) {
                    result[1][idx] = 0.0f;
                  }

                  result[2][idx] +=
                      -dt *
                      (fd<Conf::dim>::circ2(e, gp.le, idx, idx, idx_px,
                                            idx_py) -
                       dt * beta *
                           fd<Conf::dim>::circ2(j, gp.le, idx, idx, idx_px,
                                                idx_py)) /
                      gp.Ab[2][idx];
                }
              });
        },
        result, e, j, grid.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
axis_boundary_e(vector_field<Conf>& e, const grid_sph_t<Conf>& grid) {
  auto ext = grid.extent();
  // kernel_launch(
  ExecPolicy<Conf>::launch(
      [ext] LAMBDA(auto e) {
        auto& grid = ExecPolicy<Conf>::grid();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid.template coord<1>(n1_0, true)) < 0.1f * grid.delta[1]) {
            // At the theta = 0 axis
            // Set E_phi and B_theta to zero
            auto idx = Conf::idx(index_t<Conf::dim>(n0, n1_0), ext);
            e[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            e[1][idx.dec_y()] = e[1][idx];
            // e[0][idx] = 0.0f;
          }
          // printf("boundary pi at %f\n", grid.template coord<1>(n1_pi, true));
          if (abs(grid.template coord<1>(n1_pi, true) - M_PI) <
              0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = Conf::idx(index_t<Conf::dim>(n0, n1_pi), ext);
            e[2][idx] = 0.0f;
            // e[1][idx] = 0.0;
            e[1][idx] = e[1][idx.dec_y()];
            // e[0][idx] = 0.0f;
          }
        });
      },
      e);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
axis_boundary_b(vector_field<Conf>& b, const grid_sph_t<Conf>& grid) {
  auto ext = grid.extent();
  ExecPolicy<Conf>::launch(
      // kernel_launch(
      [ext] LAMBDA(auto b) {
        auto& grid = ExecPolicy<Conf>::grid();
        // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
        ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
          auto n1_0 = grid.guard[1];
          auto n1_pi = grid.dims[1] - grid.guard[1];
          if (abs(grid.template coord<1>(n1_0, true)) < 0.1f * grid.delta[1]) {
            // At the theta = 0 axis

            // Set E_phi and B_theta to zero
            auto idx = Conf::idx(index_t<Conf::dim>(n0, n1_0), ext);
            b[1][idx] = 0.0;
            b[2][idx.dec_y()] = b[2][idx];
            // b[2][idx.dec_y()] = 0.0;
            b[0][idx.dec_y()] = b[0][idx];
          }
          // printf("boundary pi at %f\n", grid.template coord<1>(n1_pi, true));
          if (abs(grid.template coord<1>(n1_pi, true) - M_PI) <
              0.1f * grid.delta[1]) {
            // At the theta = pi axis
            auto idx = Conf::idx(index_t<Conf::dim>(n0, n1_pi), ext);
            b[1][idx] = 0.0;
            b[2][idx] = b[2][idx.dec_y()];
            // b[2][idx] = 0.0;
            b[0][idx] = b[0][idx.dec_y()];
          }
        });
      },
      b);
  ExecPolicy<Conf>::sync();
}

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
field_solver<Conf, ExecPolicy, coord_policy_spherical>::field_solver(const grid_sph_t<Conf>& grid,
               const domain_comm<Conf, ExecPolicy>* comm)
      : field_solver_base<Conf>(grid), m_grid_sph(grid), m_comm(comm) {
  ExecPolicy<Conf>::set_grid(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::init() {
  field_solver_base<Conf>::init();

  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_spherical>::register_data_components() {
  auto type = ExecPolicy<Conf>::data_mem_type();

  field_solver_base<Conf>::register_data_components_impl(type);

  if (this->m_use_implicit) {
    m_tmp_b1 = sim_env().register_data<vector_field<Conf>>(
        "tmp_b1", this->m_grid, field_type::face_centered, type);
    m_tmp_b2 = sim_env().register_data<vector_field<Conf>>(
        "tmp_b2", this->m_grid, field_type::face_centered, type);
    m_bnew = sim_env().register_data<vector_field<Conf>>(
        "tmp_bnew", this->m_grid, field_type::face_centered, type);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::update_explicit(
    double dt, double time) {
  if (time < TINY)
    compute_b_update_explicit(0.5 * dt);
  else
    compute_b_update_explicit(dt);

  // Communicate B guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));
  axis_boundary_b<Conf, ExecPolicy>(*(this->B), m_grid_sph);

  compute_e_update_explicit(dt);
  axis_boundary_e<Conf, ExecPolicy>(*(this->E), m_grid_sph);

  // Communicate E guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // apply coordinate boundary condition
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary<Conf, ExecPolicy>(*(this->E), *(this->B), m_damping_length,
                                       m_damping_coef);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::update_semi_implicit(
    double dt, double alpha, double beta, double time) {
  Logger::print_detail("updating sph fields implicitly");
  // set m_tmp_b1 to B
  m_tmp_b1->copy_from(*(this->B));

  // Assemble the RHS
  compute_double_circ<Conf, ExecPolicy>(*m_tmp_b2, *m_tmp_b1, m_grid_sph,
                      -alpha * beta * dt * dt);
  m_tmp_b1->add_by(*m_tmp_b2);

  // Send guard cells for m_tmp_b1
  if (this->m_comm != nullptr)
    this->m_comm->send_guard_cells(*m_tmp_b1);

  compute_implicit_rhs<Conf, ExecPolicy>(*m_tmp_b1, *(this->E), *(this->J), m_grid_sph, alpha,
                       beta, dt);
  axis_boundary_b<Conf, ExecPolicy>(*m_tmp_b1, m_grid_sph);
  // m_tmp_b1->add_by(*(this->B0), -1.0);

  // Since we need to iterate, define a double buffer to switch quickly between
  // operand and result.
  m_bnew->copy_from(*m_tmp_b1);
  // m_tmp_b1->add_by(*(this->B0), -1.0);
  auto buffer = make_double_buffer(*m_tmp_b1, *m_tmp_b2);
  for (int i = 0; i < 6; i++) {
    compute_double_circ<Conf, ExecPolicy>(buffer.alt(), buffer.main(), m_grid_sph,
                        -beta * beta * dt * dt);

    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(buffer.alt());
    axis_boundary_b<Conf, ExecPolicy>(buffer.alt(), m_grid_sph);
    this->m_bnew->add_by(buffer.alt());

    buffer.swap();
  }
  // Communicate B new
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*m_bnew);

  // m_bnew now holds B^{n+1}
  // add_alpha_beta_cu(buffer.main(), *(this->B), *(this->m_bnew), alpha, beta);
  select(typename ExecPolicy<Conf>::exec_tag{}, buffer.main()[0]) =
      alpha * this->B->at(0) + beta * m_bnew->at(0);
  select(typename ExecPolicy<Conf>::exec_tag{}, buffer.main()[1]) =
      alpha * this->B->at(1) + beta * m_bnew->at(1);
  select(typename ExecPolicy<Conf>::exec_tag{}, buffer.main()[2]) =
      alpha * this->B->at(2) + beta * m_bnew->at(2);

  // buffer.main() now holds alpha*B^n + beta*B^{n+1}. Compute E explicitly from
  // this
  compute_e_update_explicit(dt);
  axis_boundary_e<Conf, ExecPolicy>(*(this->E), m_grid_sph);

  // Communicate E
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  this->B->copy_from(*m_bnew);

  // apply coordinate boundary condition
  if (this->m_comm == nullptr || this->m_comm->domain_info().is_boundary[1]) {
    damping_boundary<Conf, ExecPolicy>(*(this->E), *(this->B), m_damping_length, m_damping_coef);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::compute_divs_e_b() {
  vec_t<bool, Conf::dim * 2> is_boundary;
  is_boundary.set(true);
  if (m_comm != nullptr) {
    is_boundary = m_comm->domain_info().is_boundary;
  }
  ExecPolicy<Conf>::launch(
      [is_boundary] LAMBDA(auto divE, auto divB, auto e, auto b, auto gp) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();

        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                auto idx_mx = idx.dec_x();
                auto idx_my = idx.dec_y();

                divE[idx] = (e[0][idx] * gp.Ae[0][idx] -
                             e[0][idx_mx] * gp.Ae[0][idx_mx] +
                             e[1][idx] * gp.Ae[1][idx] -
                             e[1][idx_my] * gp.Ae[1][idx_my]) /
                            (gp.dV[idx] * grid.delta[0] * grid.delta[1]);
                auto idx_px = idx.inc_x();
                auto idx_py = idx.inc_y();

                divB[idx] = (b[0][idx_px] * gp.Ab[0][idx_px] -
                             b[0][idx] * gp.Ab[0][idx] +
                             b[1][idx_py] * gp.Ab[1][idx_py] -
                             b[1][idx] * gp.Ab[1][idx]) /
                            (gp.dV[idx] * grid.delta[0] * grid.delta[1]);

                if (is_boundary[0] && pos[0] == grid.guard[0])
                  divE[idx] = divB[idx] = 0.0f;
                if (is_boundary[1] &&
                    pos[0] == grid.dims[0] - grid.guard[0] - 1)
                  divE[idx] = divB[idx] = 0.0f;
                if (is_boundary[2] && pos[1] == grid.guard[1])
                  divE[idx] = divB[idx] = 0.0f;
                if (is_boundary[3] &&
                    pos[1] == grid.dims[1] - grid.guard[1] - 1)
                  divE[idx] = divB[idx] = 0.0f;
              }
            });
      },
      this->divE, this->divB, this->E, this->B, m_grid_sph.get_grid_ptrs());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::compute_flux() {
  if (Conf::dim == 2) {
    this->flux->init();

    ExecPolicy<Conf>::launch(
        [] LAMBDA(auto flux, auto b, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          // for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
            for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
                 n1++) {
              auto pos = index_t<Conf::dim>(n0, n1);
              auto idx = typename Conf::idx_t(pos, ext);
              flux[idx] = flux[idx.dec_y()] + b[0][idx] * gp.Ab[0][idx];
            }
          });
        },
        this->flux, this->B, m_grid_sph.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_spherical>::compute_EB_sqr() {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_spherical>::compute_b_update_explicit(double dt) {
  if constexpr (Conf::dim == 2) {
    const auto& E = this->E;
    // kernel_launch(
    ExecPolicy<Conf>::launch(
        [dt] LAMBDA(auto result, auto e, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          // gp is short for grid_ptrs
          // for (auto idx : grid_stride_range(Conf::begin(ext),
          // Conf::end(ext))) {
          ExecPolicy<Conf>::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                // auto idx = typename Conf::idx_t(n, ext);
                auto pos = get_pos(idx, ext);
                if (grid.is_in_bound(pos)) {
                  auto idx_py = idx.inc_y();
                  result[0][idx] -=
                      dt * fd<Conf::dim>::circ0(e, gp.le, idx, idx_py) /
                      gp.Ab[0][idx];

                  auto idx_px = idx.inc_x();
                  result[1][idx] -=
                      dt * fd<Conf::dim>::circ1(e, gp.le, idx, idx_px) /
                      gp.Ab[1][idx];
                  // Take care of axis boundary
                  Scalar theta = grid.template coord<1>(pos[1], true);
                  if (abs(theta) < 0.1f * grid.delta[1]) {
                    result[1][idx] = 0.0f;
                  }
                  // if (pos[0] == grid.guard[0] + 1 && pos[1] == grid.guard[1]
                  // + 1)
                  // {
                  //   printf("Eph is:\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n",
                  //          e[2][idx.dec_y(2).inc_x()],
                  //          e[2][idx.dec_y().inc_x()], e[2][idx.inc_x()],
                  //          e[2][idx.inc_y().inc_x()], e[2][idx.dec_y(2)],
                  //          e[2][idx.dec_y()], e[2][idx], e[2][idx.inc_y()],
                  //          e[2][idx.dec_y(2).dec_x()],
                  //          e[2][idx.dec_y().dec_x()], e[2][idx.dec_x()],
                  //          e[2][idx.inc_y().dec_x()]);
                  //   printf("circ E is: %f\n", circ2(e, gp.le, idx, idx,
                  //   idx_px, idx_py));
                  // }

                  result[2][idx] -=
                      dt *
                      fd<Conf::dim>::circ2(e, gp.le, idx, idx, idx_px, idx_py) /
                      gp.Ab[2][idx];
                }
              });
        },
        this->B, E, m_grid_sph.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_spherical>::compute_e_update_explicit(double dt) {
  if constexpr (Conf::dim == 2) {
    const auto& B = this->B;
    // kernel_launch(
    ExecPolicy<Conf>::launch(
        [dt] LAMBDA(auto result, auto b, auto j, auto gp) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          // using idx_t = typename Conf::idx_t;
          // gp is short for grid_ptrs
          // for (auto idx : grid_stride_range(Conf::begin(ext),
          // Conf::end(ext))) {
          ExecPolicy<Conf>::loop(
              Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
                // auto pos = idx.get_pos();
                auto pos = get_pos(idx, ext);
                if (grid.is_in_bound(pos)) {
                  auto idx_my = idx.dec_y();
                  result[0][idx] +=
                      dt * (fd<Conf::dim>::circ0(b, gp.lb, idx_my, idx) /
                                gp.Ae[0][idx] -
                            j[0][idx]);

                  auto idx_mx = idx.dec_x();
                  result[1][idx] +=
                      dt * (fd<Conf::dim>::circ1(b, gp.lb, idx_mx, idx) /
                                gp.Ae[1][idx] -
                            j[1][idx]);

                  result[2][idx] +=
                      dt * (fd<Conf::dim>::circ2(b, gp.lb, idx_mx, idx_my, idx,
                                                 idx) /
                                gp.Ae[2][idx] -
                            j[2][idx]);
                  // Take care of axis boundary, setting E_phi to zero
                  Scalar theta = grid.template coord<1>(pos[1], true);
                  if (abs(theta) < 0.1f * grid.delta[1]) {
                    result[2][idx] = 0.0f;
                  }
                  // if (pos[0] == grid.guard[0] + 1 && pos[1] == grid.guard[1]
                  // + 1)
                  // {
                  //   printf("Br is %f, %f, %f; Btheta is %f, %f, %f, %f\n",
                  //   b[0][idx_my],
                  //          b[0][idx], b[0][idx.inc_y()], b[1][idx_mx],
                  //          b[1][idx], b[1][idx.inc_x()], b[1][idx.inc_x(2)]);
                  //   printf("Ephi is %f, circ2_b is %f, jphi is %f\n",
                  //   result[2][idx],
                  //          circ2(b, gp.lb, idx_mx, idx_my, idx, idx),
                  //          j[2][idx]);
                  // }
                  // if (pos[0] == grid.guard[0] + 1 && pos[1] == grid.guard[1]
                  // + 1)
                  // {
                  //   printf("Br is:\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n",
                  //          b[0][idx.dec_y(2).inc_x()],
                  //          b[0][idx.dec_y().inc_x()], b[0][idx.inc_x()],
                  //          b[0][idx.inc_y().inc_x()], b[0][idx.dec_y(2)],
                  //          b[0][idx.dec_y()], b[0][idx], b[0][idx.inc_y()],
                  //          b[0][idx.dec_y(2).dec_x()],
                  //          b[0][idx.dec_y().dec_x()], b[0][idx.dec_x()],
                  //          b[0][idx.inc_y().dec_x()]);
                  //   printf("Bth is:\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n"
                  //          "%f, %f, %f, %f\n",
                  //          b[1][idx.dec_y(2).inc_x()],
                  //          b[1][idx.dec_y().inc_x()], b[1][idx.inc_x()],
                  //          b[1][idx.inc_y().inc_x()], b[1][idx.dec_y(2)],
                  //          b[1][idx.dec_y()], b[1][idx], b[1][idx.inc_y()],
                  //          b[1][idx.dec_y(2).dec_x()],
                  //          b[1][idx.dec_y().dec_x()], b[1][idx.dec_x()],
                  //          b[1][idx.inc_y().dec_x()]);
                  //   printf("circ B is: %f\n", circ2(b, gp.lb, idx_mx, idx_my,
                  //   idx, idx));
                  // }
                }
                // extra work for the theta = pi axis
                auto theta = grid.template coord<1>(pos[1], true);
                if (std::abs(theta - M_PI) < 0.1f * grid.delta[1]) {
                  auto idx_my = idx.dec_y();
                  result[0][idx] +=
                      dt * (fd<Conf::dim>::circ0(b, gp.lb, idx_my, idx) /
                                gp.Ae[0][idx] -
                            j[0][idx]);
                }
              });
        },
        this->E, B, this->J, m_grid_sph.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
  }
}

}  // namespace Aperture
