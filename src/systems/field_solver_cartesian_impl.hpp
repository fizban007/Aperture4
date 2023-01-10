/*
 * Copyright (c) 2022 Alex Chen.
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

#include "field_solver_base_impl.hpp"
#include "field_solver_cartesian.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/interpolation.hpp"

namespace Aperture {

namespace {

constexpr int diff_order = 2;

template <typename Conf>
using fd = finite_diff<Conf::dim, diff_order>;

// constexpr float cherenkov_factor = 1.025f;
constexpr float cherenkov_factor = 1.0f;
constexpr int n_pml_addon = 2;

// This is the profile of the pml conductivity
template <typename Float>
HD_INLINE Float
pml_sigma(Float x, Float dx, int n_pml) {
  // x is the distance into the pml, dx is the cell size, and n_pml is the
  // number of pml cells
  // return 4.0f * square(x / dx / (n_pml + n_pml_addon)) / dx;
  return 4.0f * cube(x / dx / (n_pml + n_pml_addon)) / dx;
}

// This is the profile of current damping in the pml, c.f. Lehe et al (2022)
template <typename Float>
HD_INLINE Float
pml_alpha(Float x, Float dx, int n_pml) {
  // x is the distance into the pml, dx is the cell size, and n_pml is the
  // number of pml cells
  // return math::exp(-4.0f * cube(math::abs(x) / dx) / (3.0f * square(n_pml +
  // n_pml_addon)));
  return math::exp(-4.0f * square(square(math::abs(x) / dx)) /
                   (4.0f * cube(n_pml + n_pml_addon)));
  // return 1.0;
}

}  // namespace

template <typename Conf, template <class> class ExecPolicy>
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::field_solver(
    const grid_t<Conf>& grid, const domain_comm<Conf, ExecPolicy>* comm)
    : field_solver_base<Conf>(grid), m_comm(comm) {
  ExecPolicy<Conf>::set_grid(this->m_grid);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::init() {
  field_solver_base<Conf>::init();

  sim_env().params().get_value("use_pml", m_use_pml);
  sim_env().params().get_value("pml_length", m_pml_length);
  sim_env().params().get_array("damping_boundary", m_damping);
  for (int i = 0; i < Conf::dim * 2; i++) {
    if (m_comm != nullptr && m_comm->domain_info().is_boundary[i] != true) {
      m_damping[i] = false;
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy,
             coord_policy_cartesian>::register_data_components() {
  auto type = ExecPolicy<Conf>::data_mem_type();

  field_solver_base<Conf>::register_data_components_impl(type);

  m_tmp_b1 = sim_env().register_data<vector_field<Conf>>(
      "tmp_b1", this->m_grid, field_type::face_centered, type);
  m_tmp_b2 = sim_env().register_data<vector_field<Conf>>(
      "tmp_b2", this->m_grid, field_type::face_centered, type);
  m_tmp_e1 = sim_env().register_data<vector_field<Conf>>(
      "tmp_e1", this->m_grid, field_type::vert_centered, type);
  m_tmp_e2 = sim_env().register_data<vector_field<Conf>>(
      "tmp_e2", this->m_grid, field_type::vert_centered, type);

  m_tmp_b1->skip_output(true);
  m_tmp_b2->skip_output(true);
  m_tmp_e1->skip_output(true);
  m_tmp_e2->skip_output(true);
  m_tmp_b1->include_in_snapshot(true);
  m_tmp_b2->include_in_snapshot(true);
  m_tmp_e1->include_in_snapshot(true);
  m_tmp_e2->include_in_snapshot(true);
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_e_update_pml(
    double dt) {
  using value_t = typename Conf::value_t;
  int pml_len = this->m_pml_length;
  vec_t<bool, Conf::dim * 2> damping(this->m_damping);
  auto& E = *(this->E);
  auto& B = *(this->B);
  auto& J = *(this->J);

  ExecPolicy<Conf>::launch(
      [dt, pml_len, damping] LAMBDA(auto result, auto e1, auto e2, auto b,
                                    auto stagger, auto j) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
        // {
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                // First compute the pml coefficients
                value_t sigma_0 = 0.0f, sigma_1 = 0.0f, sigma_2 = 0.0f;
                value_t alpha_0 = 1.0f, alpha_1 = 1.0f, alpha_2 = 1.0f;
                if (damping[0] && pos[0] < pml_len + grid.guard[0]) {
                  // value_t x = (pml_len + grid.guard[0] - pos[0]) *
                  // grid.delta[0];
                  value_t x = grid.coord(0, pml_len + grid.guard[0], true) -
                              grid.coord(0, pos[0], true);
                  sigma_0 = pml_sigma(x, grid.delta[0], pml_len);
                  alpha_1 *= pml_alpha(x, grid.delta[0], pml_len);
                  alpha_2 *= pml_alpha(x, grid.delta[0], pml_len);
                  value_t xs = grid.coord(0, pml_len + grid.guard[0], true) -
                               grid.coord(0, pos[0], false);
                  alpha_0 *= pml_alpha(xs, grid.delta[0], pml_len);
                } else if (damping[1] &&
                           pos[0] >= grid.guard[0] + grid.N[0] - pml_len) {
                  // value_t x = (pos[0] - (grid.guard[0] + grid.N[0] -
                  // pml_len)) *
                  //             grid.delta[0];
                  value_t x =
                      grid.coord(0, pos[0], true) -
                      grid.coord(0, grid.N[0] + grid.guard[0] - pml_len, true);
                  sigma_0 = pml_sigma(x, grid.delta[0], pml_len);
                  alpha_1 *= pml_alpha(x, grid.delta[0], pml_len);
                  alpha_2 *= pml_alpha(x, grid.delta[0], pml_len);
                  value_t xs =
                      grid.coord(0, pos[0], false) -
                      grid.coord(0, grid.N[0] + grid.guard[0] - pml_len, true);
                  alpha_0 *= pml_alpha(xs, grid.delta[0], pml_len);
                }
                if (Conf::dim > 1 && damping[2] &&
                    pos[1] < pml_len + grid.guard[1]) {
                  // value_t y = (pml_len + grid.guard[1] - pos[1]) *
                  // grid.delta[1];
                  value_t y = grid.coord(1, pml_len + grid.guard[1], true) -
                              grid.coord(1, pos[1], true);
                  sigma_1 = pml_sigma(y, grid.delta[1], pml_len);
                  alpha_0 *= pml_alpha(y, grid.delta[1], pml_len);
                  alpha_2 *= pml_alpha(y, grid.delta[1], pml_len);
                  value_t ys = grid.coord(1, pml_len + grid.guard[1], true) -
                               grid.coord(1, pos[1], false);
                  alpha_1 *= pml_alpha(ys, grid.delta[1], pml_len);
                } else if (Conf::dim > 1 && damping[3] &&
                           pos[1] >= grid.guard[1] + grid.N[1] - pml_len) {
                  // value_t y = (pos[1] - (grid.guard[1] + grid.N[1] -
                  // pml_len)) *
                  //             grid.delta[1];
                  value_t y =
                      grid.coord(1, pos[1], true) -
                      grid.coord(1, grid.N[1] + grid.guard[1] - pml_len, true);
                  sigma_1 = pml_sigma(y, grid.delta[1], pml_len);
                  alpha_0 *= pml_alpha(y, grid.delta[1], pml_len);
                  alpha_2 *= pml_alpha(y, grid.delta[1], pml_len);
                  value_t ys =
                      grid.coord(1, pos[1], false) -
                      grid.coord(1, grid.N[1] + grid.guard[1] - pml_len, true);
                  alpha_1 *= pml_alpha(ys, grid.delta[1], pml_len);
                }
                if (Conf::dim > 2 && damping[4] &&
                    pos[2] < pml_len + grid.guard[2]) {
                  // value_t z = (pml_len + grid.guard[2] - pos[2]) *
                  // grid.delta[2];
                  value_t z = grid.coord(2, pml_len + grid.guard[2], true) -
                              grid.coord(2, pos[2], true);
                  sigma_2 = pml_sigma(z, grid.delta[2], pml_len);
                  alpha_0 *= pml_alpha(z, grid.delta[2], pml_len);
                  alpha_1 *= pml_alpha(z, grid.delta[2], pml_len);
                  value_t zs = grid.coord(2, pml_len + grid.guard[2], true) -
                               grid.coord(2, pos[2], false);
                  alpha_2 *= pml_alpha(zs, grid.delta[2], pml_len);
                } else if (Conf::dim > 2 && damping[5] &&
                           pos[2] >= grid.guard[2] + grid.N[2] - pml_len) {
                  // value_t z = (pos[2] - (grid.guard[2] + grid.N[2] -
                  // pml_len)) *
                  //             grid.delta[2];
                  value_t z =
                      grid.coord(2, pos[2], true) -
                      grid.coord(2, grid.N[2] + grid.guard[2] - pml_len, true);
                  sigma_2 = pml_sigma(z, grid.delta[2], pml_len);
                  alpha_0 *= pml_alpha(z, grid.delta[2], pml_len);
                  alpha_1 *= pml_alpha(z, grid.delta[2], pml_len);
                  value_t zs =
                      grid.coord(2, pos[2], false) -
                      grid.coord(2, grid.N[2] + grid.guard[2] - pml_len, true);
                  alpha_2 *= pml_alpha(zs, grid.delta[2], pml_len);
                }

                // evolve E0
                if (sigma_1 > 0.0f || sigma_2 > 0.0f) {
                  e1[0][idx] +=
                      dt *
                      ((Conf::dim > 1 ? cherenkov_factor *
                                            diff<1>(b[2], idx, stagger[2],
                                                    order_tag<diff_order>{}) *
                                            grid.inv_delta[1]
                                      : 0.0f) -
                       e1[0][idx] * sigma_1);
                  // if (sigma_1 > 0.0f) {
                  //   e1[0][idx] -= (sigma_2 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[0][idx];
                  // }

                  e2[0][idx] +=
                      dt *
                      ((Conf::dim > 2 ? cherenkov_factor *
                                            (-diff<2>(b[1], idx, stagger[1],
                                                      order_tag<diff_order>{}) *
                                             grid.inv_delta[2])
                                      : 0.0f) -
                       e2[0][idx] * sigma_2);
                  // if (sigma_2 > 0.0f) {
                  //   e2[0][idx] -= (sigma_1 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[0][idx];
                  // }

                  // result[0][idx] = e1[0][idx] + e2[0][idx] - dt * j[0][idx] *
                  // (sigma_0 > 0.0f ? alpha : 1.0f);
                  result[0][idx] =
                      e1[0][idx] + e2[0][idx] - dt * j[0][idx] * alpha_0;
                  // result[0][idx] = e1[0][idx] + e2[0][idx];
                } else {
                  result[0][idx] +=
                      dt * (cherenkov_factor *
                                fd<Conf>::curl0(b, idx, stagger, grid) -
                            // j[0][idx] * (sigma_0 > 0.0f ? alpha : 1.0f));
                            j[0][idx] * alpha_0);
                }

                // evolve E1
                if (sigma_2 > 0.0f || sigma_0 > 0.0f) {
                  e1[1][idx] +=
                      dt *
                      ((Conf::dim > 2 ? cherenkov_factor *
                                            diff<2>(b[0], idx, stagger[0],
                                                    order_tag<diff_order>{}) *
                                            grid.inv_delta[2]
                                      : 0.0f) -
                       e1[1][idx] * sigma_2);
                  // if (sigma_2 > 0.0f) {
                  //   e1[1][idx] -= (sigma_0 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[1][idx];
                  // }

                  e2[1][idx] +=
                      dt *
                      ((Conf::dim > 0 ? cherenkov_factor *
                                            (-diff<0>(b[2], idx, stagger[2],
                                                      order_tag<diff_order>{}) *
                                             grid.inv_delta[0])
                                      : 0.0f) -
                       e2[1][idx] * sigma_0);
                  // if (sigma_0 > 0.0f) {
                  //   e2[1][idx] -= (sigma_2 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[1][idx];
                  // }

                  // result[1][idx] = e1[1][idx] + e2[1][idx] - dt * j[1][idx] *
                  // (sigma_1 > 0.0f ? alpha : 1.0f);
                  result[1][idx] =
                      e1[1][idx] + e2[1][idx] - dt * j[1][idx] * alpha_1;
                  // result[1][idx] = e1[1][idx] + e2[1][idx];
                } else {
                  result[1][idx] +=
                      dt * (cherenkov_factor *
                                fd<Conf>::curl1(b, idx, stagger, grid) -
                            // j[1][idx] * (sigma_1 > 0.0f ? alpha : 1.0f));
                            j[1][idx] * alpha_1);
                }

                // evolve E2
                if (sigma_0 > 0.0f || sigma_1 > 0.0f) {
                  e1[2][idx] +=
                      dt *
                      ((Conf::dim > 0 ? cherenkov_factor *
                                            diff<0>(b[1], idx, stagger[1],
                                                    order_tag<diff_order>{}) *
                                            grid.inv_delta[0]
                                      : 0.0f) -
                       e1[2][idx] * sigma_0);
                  // if (sigma_0 > 0.0f) {
                  //   e1[2][idx] -= (sigma_1 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[2][idx];
                  // }

                  e2[2][idx] +=
                      dt *
                      ((Conf::dim > 1 ? cherenkov_factor *
                                            (-diff<1>(b[0], idx, stagger[0],
                                                      order_tag<diff_order>{}) *
                                             grid.inv_delta[1])
                                      : 0.0f) -
                       e2[2][idx] * sigma_1);
                  // if (sigma_1 > 0.0f) {
                  //   e2[2][idx] -= (sigma_0 > 0.0f ? 0.5f : 1.0f) * dt * alpha
                  //   * j[2][idx];
                  // }

                  // result[2][idx] = e1[2][idx] + e2[2][idx] - dt * j[2][idx] *
                  // (sigma_2 > 0.0f ? alpha : 1.0f);
                  result[2][idx] =
                      e1[2][idx] + e2[2][idx] - dt * j[2][idx] * alpha_2;
                  // result[2][idx] = e1[2][idx] + e2[2][idx];
                } else {
                  result[2][idx] +=
                      dt * (cherenkov_factor *
                                fd<Conf>::curl2(b, idx, stagger, grid) -
                            // j[2][idx] * (sigma_2 > 0.0f ? alpha : 1.0f));
                            j[2][idx] * alpha_2);
                }
              }
            });
      },
      this->E, this->m_tmp_e1, this->m_tmp_e2, this->B, this->B->stagger_vec(),
      this->J);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_e_update(
    double dt) {
  using value_t = typename Conf::value_t;
  auto& E = *(this->E);
  auto& B = *(this->B);
  auto& J = *(this->J);

  ExecPolicy<Conf>::launch(
      [dt] LAMBDA(auto result, auto b, auto stagger, auto j) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
        // {
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                // evolve E0
                result[0][idx] +=
                    dt *
                    (cherenkov_factor * fd<Conf>::curl0(b, idx, stagger, grid) -
                     j[0][idx]);

                // evolve E1
                result[1][idx] +=
                    dt *
                    (cherenkov_factor * fd<Conf>::curl1(b, idx, stagger, grid) -
                     j[1][idx]);

                // evolve E2
                result[2][idx] +=
                    dt *
                    (cherenkov_factor * fd<Conf>::curl2(b, idx, stagger, grid) -
                     j[2][idx]);
              }
            });
      },
      this->E, this->B, this->B->stagger_vec(), this->J);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_b_update_pml(
    double dt) {
  using value_t = typename Conf::value_t;
  int pml_len = this->m_pml_length;
  vec_t<bool, Conf::dim * 2> damping(this->m_damping);
  auto& B = *(this->B);
  auto& E = *(this->E);

  ExecPolicy<Conf>::launch(
      [dt, pml_len, damping] LAMBDA(auto result, auto b1, auto b2, auto e,
                                    auto stagger) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
        // {
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                // First compute the pml coefficients
                value_t sigma_0 = 0.0f, sigma_1 = 0.0f, sigma_2 = 0.0f;
                if (damping[0] && pos[0] < pml_len + grid.guard[0]) {
                  // value_t x = (pml_len + grid.guard[0] - pos[0]) *
                  // grid.delta[0];
                  value_t x = grid.coord(0, pml_len + grid.guard[0], true) -
                              grid.coord(0, pos[0], false);
                  sigma_0 = pml_sigma(x, grid.delta[0], pml_len);
                } else if (damping[1] &&
                           pos[0] >= grid.guard[0] + grid.N[0] - pml_len) {
                  // value_t x = (pos[0] - (grid.guard[0] + grid.N[0] -
                  // pml_len)) *
                  //             grid.delta[0];
                  value_t x =
                      grid.coord(0, pos[0], false) -
                      grid.coord(0, grid.N[0] + grid.guard[0] - pml_len, true);
                  sigma_0 = pml_sigma(x, grid.delta[0], pml_len);
                }
                if (Conf::dim > 1 && damping[2] &&
                    pos[1] < pml_len + grid.guard[1]) {
                  // value_t y = (pml_len + grid.guard[1] - pos[1]) *
                  // grid.delta[1];
                  value_t y = grid.coord(1, pml_len + grid.guard[1], true) -
                              grid.coord(1, pos[1], false);
                  sigma_1 = pml_sigma(y, grid.delta[1], pml_len);
                } else if (Conf::dim > 1 && damping[3] &&
                           pos[1] >= grid.guard[1] + grid.N[1] - pml_len) {
                  // value_t y = (pos[1] - (grid.guard[1] + grid.N[1] -
                  // pml_len)) *
                  //             grid.delta[1];
                  value_t y =
                      grid.coord(1, pos[1], false) -
                      grid.coord(1, grid.N[1] + grid.guard[1] - pml_len, true);
                  sigma_1 = pml_sigma(y, grid.delta[1], pml_len);
                }
                if (Conf::dim > 2 && damping[4] &&
                    pos[2] < pml_len + grid.guard[2]) {
                  // value_t z = (pml_len + grid.guard[2] - pos[2]) *
                  // grid.delta[2];
                  value_t z = grid.coord(2, pml_len + grid.guard[2], true) -
                              grid.coord(2, pos[2], false);
                  sigma_2 = pml_sigma(z, grid.delta[2], pml_len);
                } else if (Conf::dim > 2 && damping[5] &&
                           pos[2] >= grid.guard[2] + grid.N[2] - pml_len) {
                  // value_t z = (pos[2] - (grid.guard[2] + grid.N[2] -
                  // pml_len)) *
                  //             grid.delta[2];
                  value_t z =
                      grid.coord(2, pos[2], false) -
                      grid.coord(2, grid.N[2] + grid.guard[2] - pml_len, true);
                  sigma_2 = pml_sigma(z, grid.delta[2], pml_len);
                }

                // evolve B0
                if (sigma_1 > 0.0f || sigma_2 > 0.0f) {
                  b1[0][idx] +=
                      dt *
                      (-(Conf::dim > 1 ? cherenkov_factor *
                                             diff<1>(e[2], idx, stagger[2],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[1]
                                       : 0.0f) -
                       b1[0][idx] * sigma_1);

                  b2[0][idx] +=
                      dt * (-(Conf::dim > 2
                                  ? cherenkov_factor *
                                        (-diff<2>(e[1], idx, stagger[1],
                                                  order_tag<diff_order>{}) *
                                         grid.inv_delta[2])
                                  : 0.0f) -
                            b2[0][idx] * sigma_2);

                  result[0][idx] = b1[0][idx] + b2[0][idx];
                } else {
                  result[0][idx] += -dt * cherenkov_factor *
                                    fd<Conf>::curl0(e, idx, stagger, grid);
                }

                // evolve B1
                if (sigma_2 > 0.0f || sigma_0 > 0.0f) {
                  b1[1][idx] +=
                      dt *
                      (-(Conf::dim > 2 ? cherenkov_factor *
                                             diff<2>(e[0], idx, stagger[0],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[2]
                                       : 0.0f) -
                       b1[1][idx] * sigma_2);

                  b2[1][idx] +=
                      dt * (-(Conf::dim > 0
                                  ? cherenkov_factor *
                                        (-diff<0>(e[2], idx, stagger[2],
                                                  order_tag<diff_order>{}) *
                                         grid.inv_delta[0])
                                  : 0.0f) -
                            b2[1][idx] * sigma_0);

                  result[1][idx] = b1[1][idx] + b2[1][idx];
                } else {
                  result[1][idx] += -dt * cherenkov_factor *
                                    fd<Conf>::curl1(e, idx, stagger, grid);
                }

                // evolve B2
                if (sigma_0 > 0.0f || sigma_1 > 0.0f) {
                  b1[2][idx] +=
                      dt *
                      (-(Conf::dim > 0 ? cherenkov_factor *
                                             diff<0>(e[1], idx, stagger[1],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[0]
                                       : 0.0f) -
                       b1[2][idx] * sigma_0);

                  b2[2][idx] +=
                      dt * (-(Conf::dim > 1
                                  ? cherenkov_factor *
                                        (-diff<1>(e[0], idx, stagger[0],
                                                  order_tag<diff_order>{}) *
                                         grid.inv_delta[1])
                                  : 0.0f) -
                            b2[2][idx] * sigma_1);

                  result[2][idx] = b1[2][idx] + b2[2][idx];
                } else {
                  result[2][idx] += -dt * cherenkov_factor *
                                    fd<Conf>::curl2(e, idx, stagger, grid);
                }
              }
            });
      },
      this->B, this->m_tmp_b1, this->m_tmp_b2, this->E, this->E->stagger_vec());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_b_update(
    double dt) {
  using value_t = typename Conf::value_t;
  auto& B = *(this->B);
  auto& E = *(this->E);

  ExecPolicy<Conf>::launch(
      [dt] LAMBDA(auto result, auto e, auto stagger) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto ext = grid.extent();
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                // evolve B0
                result[0][idx] += -dt * cherenkov_factor *
                                  fd<Conf>::curl0(e, idx, stagger, grid);

                // evolve B1
                result[1][idx] += -dt * cherenkov_factor *
                                  fd<Conf>::curl1(e, idx, stagger, grid);

                // evolve B2
                result[2][idx] += -dt * cherenkov_factor *
                                  fd<Conf>::curl2(e, idx, stagger, grid);
              }
            });
      },
      this->B, this->E, this->E->stagger_vec());
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_divs_e_b() {}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_flux() {
  if constexpr (Conf::dim == 2) {
    this->flux->init();
    auto ext = this->m_grid.extent();
    ExecPolicy<Conf>::launch(
        [ext] LAMBDA(auto flux, auto b) {
          auto& grid = ExecPolicy<Conf>::grid();
          ExecPolicy<Conf>::loop(0, grid.dims[0], [&] LAMBDA(auto n0) {
            for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
                 n1++) {
              auto pos = index_t<2>(n0, n1);
              auto idx = typename Config<2>::idx_t(pos, ext);
              flux[idx] = flux[idx.dec_y()] + b[0][idx] * grid.delta[1];
            }
          });
        },
        this->flux, this->Btotal);
    ExecPolicy<Conf>::sync();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::compute_EB_sqr() {
  this->E_sqr->init();
  this->B_sqr->init();

  auto ext = this->m_grid.extent();
  using value_t = typename Conf::value_t;
  ExecPolicy<Conf>::launch(
      [ext] LAMBDA(auto e_sqr, auto b_sqr, auto e, auto b) {
        auto& grid = ExecPolicy<Conf>::grid();
        auto interp = interp_t<1, Conf::dim>{};
        vec_t<value_t, 3> vert(0.0, 0.0, 0.0);
        ExecPolicy<Conf>::loop(
            Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
              auto pos = get_pos(idx, ext);
              if (grid.is_in_bound(pos)) {
                value_t E1 = interp(vert, e[0], idx, ext, stagger_t(0b110));
                value_t E2 = interp(vert, e[1], idx, ext, stagger_t(0b101));
                value_t E3 = interp(vert, e[2], idx, ext, stagger_t(0b011));
                value_t B1 = interp(vert, b[0], idx, ext, stagger_t(0b001));
                value_t B2 = interp(vert, b[1], idx, ext, stagger_t(0b010));
                value_t B3 = interp(vert, b[2], idx, ext, stagger_t(0b100));

                e_sqr[idx] = E1 * E1 + E2 * E2 + E3 * E3;
                b_sqr[idx] = B1 * B1 + B2 * B2 + B3 * B3;
              }
            });
      },
      this->E_sqr, this->B_sqr, this->Etotal, this->Btotal);
  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::update_explicit(
    double dt, double time) {
  Logger::print_detail("Running explicit Cartesian solver!");
  // dt *= 1.025;
  // if (time < TINY) {
  //   compute_e_update_explicit_pml_cu(*(this->E), *(this->m_tmp_e1),
  //   *(this->m_tmp_e2), *(this->B), *(this->J),
  //                                0.5f * dt, this->m_pml_length, damping);
  //   if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  // }

  if (this->m_update_b) {
    if (m_use_pml) {
      compute_b_update_pml(dt);
    } else {
      compute_b_update(dt);
    }
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));
  }
  if (this->m_update_e) {
    if (m_use_pml) {
      compute_e_update_pml(dt);
    } else {
      compute_e_update(dt);
    }
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  ExecPolicy<Conf>::sync();
}

template <typename Conf, template <class> class ExecPolicy>
void
field_solver<Conf, ExecPolicy, coord_policy_cartesian>::update_semi_implicit(
    double dt, double alpha, double beta, double time) {
  // FIXME: Running explicit update even for semi implicit
  update_explicit(dt, time);
}

}  // namespace Aperture
