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
#include "core/math.hpp"
#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "field_solver.h"
#include "framework/config.h"
#include "systems/helpers/field_solver_helper_cu.hpp"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/double_buffer.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

namespace {

constexpr int diff_order = 2;

template <typename Conf>
using fd = finite_diff<Conf::dim, diff_order>;

// constexpr float cherenkov_factor = 1.025f;
constexpr float cherenkov_factor = 1.0f;
constexpr int n_pml_addon = 1;

// This is the profile of the pml conductivity
template <typename Float>
HD_INLINE
Float pml_sigma(Float x, Float dx, int n_pml) {
  // x is the distance into the pml, dx is the cell size, and n_pml is the
  // number of pml cells
  return 2.0f * square(x / dx / (n_pml + n_pml_addon)) / dx;
}

// This is the profile of current damping in the pml, c.f. Lehe et al (2022)
template <typename Float>
HD_INLINE
Float pml_alpha(Float x, Float dx, int n_pml) {
  // x is the distance into the pml, dx is the cell size, and n_pml is the
  // number of pml cells
  return math::exp(-2.0f * cube(math::abs(x)) / (3.0f * square(n_pml + n_pml_addon) * cube(dx)));
}

template <typename Conf>
void
compute_e_update_explicit_pml_cu(vector_field<Conf> &result,
                                 vector_field<Conf> &e1, vector_field<Conf> &e2,
                                 const vector_field<Conf> &b,
                                 const vector_field<Conf> &j,
                                 typename Conf::value_t dt, int n_pml,
                                 const vec_t<bool, Conf::dim * 2> &damping) {
  using value_t = typename Conf::value_t;
  kernel_launch(
      [dt, n_pml] __device__(auto result, auto e1, auto e2, auto b, auto stagger, auto j,
                             auto damping) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            // First compute the pml coefficients
            value_t sigma_0 = 0.0f, sigma_1 = 0.0f, sigma_2 = 0.0f;
            value_t alpha = 1.0f;
            if (damping[0] && pos[0] < n_pml + grid.guard[0]) {
              // value_t x = (n_pml + grid.guard[0] - pos[0]) * grid.delta[0];
              value_t x = grid.pos(0, n_pml + grid.guard[0], true) -
                          grid.pos(0, pos[0], true);
              sigma_0 = pml_sigma(x, grid.delta[0], n_pml);
              alpha *= pml_alpha(x, grid.delta[0], n_pml);
            } else if (damping[1] &&
                       pos[0] >= grid.guard[0] + grid.N[0] - n_pml) {
              // value_t x = (pos[0] - (grid.guard[0] + grid.N[0] - n_pml)) *
              //             grid.delta[0];
              value_t x = grid.pos(0, pos[0], true) -
                          grid.pos(0, grid.N[0] + grid.guard[0] - n_pml, true);
              sigma_0 = pml_sigma(x, grid.delta[0], n_pml);
              alpha *= pml_alpha(x, grid.delta[0], n_pml);
            }
            if (Conf::dim > 1 && damping[2] && pos[1] < n_pml + grid.guard[1]) {
              // value_t y = (n_pml + grid.guard[1] - pos[1]) * grid.delta[1];
              value_t y = grid.pos(1, n_pml + grid.guard[1], true) -
                          grid.pos(1, pos[1], true);
              sigma_1 = pml_sigma(y, grid.delta[1], n_pml);
              alpha *= pml_alpha(y, grid.delta[1], n_pml);
            } else if (Conf::dim > 1 && damping[3] &&
                       pos[1] >= grid.guard[1] + grid.N[1] - n_pml) {
              // value_t y = (pos[1] - (grid.guard[1] + grid.N[1] - n_pml)) *
              //             grid.delta[1];
              value_t y = grid.pos(1, pos[1], true)
                  - grid.pos(1, grid.N[1] + grid.guard[1] - n_pml, true);
              sigma_1 = pml_sigma(y, grid.delta[1], n_pml);
              alpha *= pml_alpha(y, grid.delta[1], n_pml);
            }
            if (Conf::dim > 2 && damping[4] && pos[2] < n_pml + grid.guard[2]) {
              // value_t z = (n_pml + grid.guard[2] - pos[2]) * grid.delta[2];
              value_t z = grid.pos(2, n_pml + grid.guard[2], true)
                  - grid.pos(2, pos[2], true);
              sigma_2 = pml_sigma(z, grid.delta[2], n_pml);
              alpha *= pml_alpha(z, grid.delta[2], n_pml);
            } else if (Conf::dim > 2 && damping[5] &&
                       pos[2] >= grid.guard[2] + grid.N[2] - n_pml) {
              // value_t z = (pos[2] - (grid.guard[2] + grid.N[2] - n_pml)) *
              //             grid.delta[2];
              value_t z = grid.pos(2, pos[2], true)
                  - grid.pos(2, grid.N[2] + grid.guard[2] - n_pml, true);
              sigma_2 = pml_sigma(z, grid.delta[2], n_pml);
              alpha *= pml_alpha(z, grid.delta[2], n_pml);
            }

            // evolve E0
            if (sigma_1 > 0.0f || sigma_2 > 0.0f) {
              e1[0][idx] +=
                  dt * ((Conf::dim > 1 ? cherenkov_factor *
                                             diff<1>(b[2], idx, stagger[2],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[1]
                                       : 0.0f) -
                        e1[0][idx] * sigma_1);
              // if (sigma_1 > 0.0f) {
              //   e1[0][idx] -= (sigma_2 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[0][idx];
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
              //   e2[0][idx] -= (sigma_1 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[0][idx];
              // }

              result[0][idx] = e1[0][idx] + e2[0][idx] - dt * j[0][idx] * (sigma_0 > 0.0f ? alpha : 1.0f);
              // result[0][idx] = e1[0][idx] + e2[0][idx];
            } else {
              result[0][idx] += dt * (cherenkov_factor *
                                      fd<Conf>::curl0(b, idx, stagger, grid) -
                                  j[0][idx] * (sigma_0 > 0.0f ? alpha : 1.0f));
            }

            // evolve E1
            if (sigma_2 > 0.0f || sigma_0 > 0.0f) {
              e1[1][idx] +=
                  dt * ((Conf::dim > 2 ? cherenkov_factor *
                                             diff<2>(b[0], idx, stagger[0],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[2]
                                       : 0.0f) -
                        e1[1][idx] * sigma_2);
              // if (sigma_2 > 0.0f) {
              //   e1[1][idx] -= (sigma_0 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[1][idx];
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
              //   e2[1][idx] -= (sigma_2 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[1][idx];
              // }

              result[1][idx] = e1[1][idx] + e2[1][idx] - dt * j[1][idx] * (sigma_1 > 0.0f ? alpha : 1.0f);
              // result[1][idx] = e1[1][idx] + e2[1][idx];
            } else {
              result[1][idx] +=
                  dt *
                  (cherenkov_factor * fd<Conf>::curl1(b, idx, stagger, grid) -
                   j[1][idx] * (sigma_1 > 0.0f ? alpha : 1.0f));
            }

            // evolve E2
            if (sigma_0 > 0.0f || sigma_1 > 0.0f) {
              e1[2][idx] +=
                  dt * ((Conf::dim > 0 ? cherenkov_factor *
                                             diff<0>(b[1], idx, stagger[1],
                                                     order_tag<diff_order>{}) *
                                             grid.inv_delta[0]
                                       : 0.0f) -
                        e1[2][idx] * sigma_0);
              // if (sigma_0 > 0.0f) {
              //   e1[2][idx] -= (sigma_1 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[2][idx];
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
              //   e2[2][idx] -= (sigma_0 > 0.0f ? 0.5f : 1.0f) * dt * alpha * j[2][idx];
              // }

              result[2][idx] = e1[2][idx] + e2[2][idx] - dt * j[2][idx] * (sigma_2 > 0.0f ? alpha : 1.0f);
              // result[2][idx] = e1[2][idx] + e2[2][idx];
            } else {
              result[2][idx] +=
                  dt *
                  (cherenkov_factor * fd<Conf>::curl2(b, idx, stagger, grid) -
                   j[2][idx] * (sigma_2 > 0.0f ? alpha : 1.0f));
            }
          }
        }
      },
      result.get_ptrs(), e1.get_ptrs(), e2.get_ptrs(), b.get_ptrs(), b.stagger_vec(),
      j.get_ptrs(), damping);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_b_update_explicit_pml_cu(vector_field<Conf> &result,
                                 vector_field<Conf> &b1, vector_field<Conf> &b2,
                                 const vector_field<Conf> &e,
                                 typename Conf::value_t dt, int n_pml,
                                 const vec_t<bool, Conf::dim * 2> &damping) {
  using value_t = typename Conf::value_t;
  kernel_launch(
      [dt, n_pml] __device__(auto result, auto b1, auto b2, auto e, auto stagger,
                             auto damping) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            // First compute the pml coefficients
            value_t sigma_0 = 0.0f, sigma_1 = 0.0f, sigma_2 = 0.0f;
            if (damping[0] && pos[0] < n_pml + grid.guard[0]) {
              // value_t x = (n_pml + grid.guard[0] - pos[0]) * grid.delta[0];
              value_t x = grid.pos(0, n_pml + grid.guard[0], true)
                  - grid.pos(0, pos[0], true);
              sigma_0 = pml_sigma(x, grid.delta[0], n_pml);
            } else if (damping[1] &&
                       pos[0] >= grid.guard[0] + grid.N[0] - n_pml) {
              // value_t x = (pos[0] - (grid.guard[0] + grid.N[0] - n_pml)) *
              //             grid.delta[0];
              value_t x = grid.pos(0, pos[0], true)
                  - grid.pos(0, grid.N[0] + grid.guard[0] - n_pml, true);
              sigma_0 = pml_sigma(x, grid.delta[0], n_pml);
            }
            if (Conf::dim > 1 && damping[2] && pos[1] < n_pml + grid.guard[1]) {
              // value_t y = (n_pml + grid.guard[1] - pos[1]) * grid.delta[1];
              value_t y = grid.pos(1, n_pml + grid.guard[1], true)
                  - grid.pos(1, pos[1], true);
              sigma_1 = pml_sigma(y, grid.delta[1], n_pml);
            } else if (Conf::dim > 1 && damping[3] &&
                       pos[1] >= grid.guard[1] + grid.N[1] - n_pml) {
              // value_t y = (pos[1] - (grid.guard[1] + grid.N[1] - n_pml)) *
              //             grid.delta[1];
              value_t y = grid.pos(1, pos[1], true)
                  - grid.pos(1, grid.N[1] + grid.guard[1] - n_pml, true);
              sigma_1 = pml_sigma(y, grid.delta[1], n_pml);
            }
            if (Conf::dim > 2 && damping[4] && pos[2] < n_pml + grid.guard[2]) {
              // value_t z = (n_pml + grid.guard[2] - pos[2]) * grid.delta[2];
              value_t z = grid.pos(2, n_pml + grid.guard[2], true)
                  - grid.pos(2, pos[2], true);
              sigma_2 = pml_sigma(z, grid.delta[2], n_pml);
            } else if (Conf::dim > 2 && damping[5] &&
                       pos[2] >= grid.guard[2] + grid.N[2] - n_pml) {
              // value_t z = (pos[2] - (grid.guard[2] + grid.N[2] - n_pml)) *
              //             grid.delta[2];
              value_t z = grid.pos(2, pos[2], true)
                  - grid.pos(2, grid.N[2] + grid.guard[2] - n_pml, true);
              sigma_2 = pml_sigma(z, grid.delta[2], n_pml);
            }

            // evolve B0
            if (sigma_1 > 0.0f || sigma_2 > 0.0f) {
              b1[0][idx] +=
                  dt * (-(Conf::dim > 1 ? cherenkov_factor *
                                              diff<1>(e[2], idx, stagger[2],
                                                      order_tag<diff_order>{}) *
                                              grid.inv_delta[1]
                                        : 0.0f) -
                        b1[0][idx] * sigma_1);

              b2[0][idx] +=
                  dt *
                  (-(Conf::dim > 2 ? cherenkov_factor *
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
                  dt * (-(Conf::dim > 2 ? cherenkov_factor *
                                              diff<2>(e[0], idx, stagger[0],
                                                      order_tag<diff_order>{}) *
                                              grid.inv_delta[2]
                                        : 0.0f) -
                        b1[1][idx] * sigma_2);

              b2[1][idx] +=
                  dt *
                  (-(Conf::dim > 0 ? cherenkov_factor *
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
                  dt * (-(Conf::dim > 0 ? cherenkov_factor *
                                              diff<0>(e[1], idx, stagger[1],
                                                      order_tag<diff_order>{}) *
                                              grid.inv_delta[0]
                                        : 0.0f) -
                        b1[2][idx] * sigma_0);

              b2[2][idx] +=
                  dt *
                  (-(Conf::dim > 1 ? cherenkov_factor *
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
        }
      },
      result.get_ptrs(), b1.get_ptrs(), b2.get_ptrs(), e.get_ptrs(), e.stagger_vec(),
      damping);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_e_update_explicit_cu(vector_field<Conf> &result,
                             const vector_field<Conf> &e,
                             const vector_field<Conf> &b,
                             const vector_field<Conf> &j,
                             typename Conf::value_t dt) {
  kernel_launch(
      [dt] __device__(auto result, auto e, auto b, auto stagger, auto j) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            result[0][idx] =
                e[0][idx] + dt * (cherenkov_factor *
                                      fd<Conf>::curl0(b, idx, stagger, grid) -
                                  j[0][idx]);

            result[1][idx] =
                e[1][idx] + dt * (cherenkov_factor *
                                      fd<Conf>::curl1(b, idx, stagger, grid) -
                                  j[1][idx]);

            result[2][idx] =
                e[2][idx] + dt * (cherenkov_factor *
                                      fd<Conf>::curl2(b, idx, stagger, grid) -
                                  j[2][idx]);
          }
        }
      },
      result.get_ptrs(), e.get_ptrs(), b.get_ptrs(), b.stagger_vec(),
      j.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_b_update_explicit_cu(vector_field<Conf> &result,
                             const vector_field<Conf> &b,
                             const vector_field<Conf> &e,
                             typename Conf::value_t dt) {
  kernel_launch(
      [dt] __device__(auto result, auto b, auto e, auto stagger) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            result[0][idx] =
                b[0][idx] -
                dt * cherenkov_factor * fd<Conf>::curl0(e, idx, stagger, grid);

            result[1][idx] =
                b[1][idx] -
                dt * cherenkov_factor * fd<Conf>::curl1(e, idx, stagger, grid);

            result[2][idx] =
                b[2][idx] -
                dt * cherenkov_factor * fd<Conf>::curl2(e, idx, stagger, grid);
          }
        }
      },
      result.get_ptrs(), b.get_ptrs(), e.get_ptrs(), e.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_double_curl(vector_field<Conf> &result, const vector_field<Conf> &b,
                    typename Conf::value_t coef) {
  kernel_launch(
      [coef] __device__(auto result, auto b, auto stagger) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            result[0][idx] = -coef * (fd<Conf>::laplacian(b[0], idx, grid));
            result[1][idx] = -coef * (fd<Conf>::laplacian(b[1], idx, grid));
            result[2][idx] = -coef * (fd<Conf>::laplacian(b[2], idx, grid));
          }
          // if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && pos[1] ==
          // grid.dims[1] / 2) {
          //   printf("B3 result is %.9f, B3 input is %.9f\n", result[2][idx],
          //   b[2][idx]);
          // }
        }
      },
      result.get_ptrs(), b.get_ptrs(), b.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_implicit_rhs(vector_field<Conf> &result, const vector_field<Conf> &e,
                     const vector_field<Conf> &j, typename Conf::value_t alpha,
                     typename Conf::value_t beta, typename Conf::value_t dt) {
  kernel_launch(
      [alpha, beta, dt] __device__(auto result, auto e, auto j, auto stagger) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            result[0][idx] +=
                -dt * (fd<Conf>::curl0(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl0(j, idx, stagger, grid));
            result[1][idx] +=
                -dt * (fd<Conf>::curl1(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl1(j, idx, stagger, grid));
            result[2][idx] +=
                -dt * (fd<Conf>::curl2(e, idx, stagger, grid) -
                       dt * beta * fd<Conf>::curl2(j, idx, stagger, grid));
          }
          // if (pos[0] == grid.dims[0] - grid.guard[0] - 1 && pos[1] ==
          // grid.dims[1] / 2) {
          //   printf("J2 is %.9f, E2 is %.9f\n", j[1][idx], e[1][idx]);
          // }
        }
      },
      result.get_ptrs(), e.get_ptrs(), j.get_ptrs(), e.stagger_vec());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_divs_cu(scalar_field<Conf> &divE, scalar_field<Conf> &divB,
                const vector_field<Conf> &e, const vector_field<Conf> &b,
                const vec_t<bool, Conf::dim * 2> &is_boundary) {
  // vec_t<bool, Conf::dim * 2> boundary(is_boundary);
  kernel_launch(
      [] __device__(auto divE, auto divB, auto e, auto b, auto st_e, auto st_b,
                    auto is_boundary) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          // auto pos = idx.get_pos();
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            divE[idx] = fd<Conf>::div(e, idx, st_e, grid);
            divB[idx] = fd<Conf>::div(b, idx, st_b, grid);

            // Check boundary
            // if (is_boundary[0] && pos[0] == grid.guard[0])
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[1] && pos[0] == grid.dims[0] - grid.guard[0] - 1)
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[2] && pos[1] == grid.guard[1])
            //   divE[idx] = divB[idx] = 0.0f;
            // if (is_boundary[3] && pos[1] == grid.dims[1] - grid.guard[1] - 1)
            //   divE[idx] = divB[idx] = 0.0f;
          }
        }
      },
      divE.dev_ndptr(), divB.dev_ndptr(), e.get_ptrs(), b.get_ptrs(),
      e.stagger_vec(), b.stagger_vec(), is_boundary);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
compute_flux(scalar_field<Conf> &flux, const vector_field<Conf> &b,
             const grid_t<Conf> &grid) {}

template <>
void
compute_flux<Config<2>>(scalar_field<Config<2>> &flux,
                        const vector_field<Config<2>> &b,
                        const grid_t<Config<2>> &grid) {
  // if constexpr (Conf::dim == 2) {
  flux.init();
  auto ext = grid.extent();
  kernel_launch(
      [ext] __device__(auto flux, auto b) {
        auto &grid = dev_grid<2, typename Config<2>::value_t>();
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          for (int n1 = grid.guard[1]; n1 < grid.dims[1] - grid.guard[1];
               n1++) {
            auto pos = index_t<2>(n0, n1);
            auto idx = typename Config<2>::idx_t(pos, ext);
            flux[idx] = flux[idx.dec_y()] + b[0][idx] * grid.delta[1];
          }
        }
      },
      flux.dev_ndptr(), b.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
  // }
}

template <typename Conf>
void
compute_EB_sqr(scalar_field<Conf> &E_sqr, scalar_field<Conf> &B_sqr, const vector_field<Conf> &E,
               const vector_field<Conf> &B) {
  E_sqr.init();
  B_sqr.init();
  auto& grid = B.grid();
  kernel_launch(
      [] __device__(auto e_sqr, auto b_sqr, auto e, auto b) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        auto interp = interp_t<1, Conf::dim>{};
        using value_t = typename Conf::value_t;
        vec_t<value_t, 3> vert(0.0, 0.0, 0.0);

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            value_t E1 = interp(vert, e[0], idx, ext, stagger_t(0b110));
            value_t E2 = interp(vert, e[1], idx, ext, stagger_t(0b101));
            value_t E3 = interp(vert, e[2], idx, ext, stagger_t(0b011));
            value_t B1 = interp(vert, b[0], idx, ext, stagger_t(0b001));
            value_t B2 = interp(vert, b[1], idx, ext, stagger_t(0b010));
            value_t B3 = interp(vert, b[2], idx, ext, stagger_t(0b100));

            e_sqr[idx] = E1*E1 + E2*E2 + E3*E3;
            b_sqr[idx] = B1*B1 + B2*B2 + B3*B3;
          }
        }
      },
      E_sqr.dev_ndptr(), B_sqr.dev_ndptr(), E.get_ptrs(), B.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void
clean_field(vector_field<Conf> &f) {
  kernel_launch(
      [] __device__(auto f) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          for (int i = 0; i < 3; i++) {
            if (math::abs(f[i][idx]) < TINY) {
              f[i][idx] = 0.0f;
            }
          }
        }
      },
      f.get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

}  // namespace

template <typename Conf>
void
field_solver_cu<Conf>::init_tmp_fields() {
  if (this->m_use_implicit) {
    this->m_tmp_b1 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::face_centered, MemType::device_only);
    // this->m_tmp_e1 = std::make_unique<vector_field<Conf>>(
    //     this->m_grid, field_type::edge_centered, MemType::device_only);
    this->m_bnew = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::face_centered, MemType::device_only);
    // this->m_enew = std::make_unique<vector_field<Conf>>(
    //     this->m_grid, field_type::edge_centered, MemType::device_only);
    this->m_tmp_b2 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::face_centered, MemType::device_only);
  } else {
    this->m_tmp_b1 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::face_centered, MemType::device_only);
    this->m_tmp_b2 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::face_centered, MemType::device_only);
    this->m_tmp_e1 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::edge_centered, MemType::device_only);
    this->m_tmp_e2 = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::edge_centered, MemType::device_only);
    this->m_tmp_b1->init();
    this->m_tmp_b2->init();
    this->m_tmp_e1->init();
    this->m_tmp_e2->init();
  }
}

template <typename Conf>
void
field_solver_cu<Conf>::register_data_components() {
  this->register_data_impl(MemType::host_device);
}

template <typename Conf>
void
field_solver_cu<Conf>::compute_e_update_pml(vector_field<Conf>& E, const vector_field<Conf>& B,
                                            const vector_field<Conf>& J, double dt) {
  vec_t<bool, Conf::dim * 2> damping(this->m_damping);

  if (this->m_update_e) {
    compute_e_update_explicit_pml_cu(E, *(this->m_tmp_e1), *(this->m_tmp_e2), B, J,
                                 dt, this->m_pml_length, damping);
    // compute_e_update_explicit_cu(*(this->E), *(this->E), *(this->B), *(this->J),
    //                              dt);
    // Communicate the new E values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(E);
  }
}

template <typename Conf>
void
field_solver_cu<Conf>::compute_b_update_pml(vector_field<Conf>& B, const vector_field<Conf>& E, double dt) {
  vec_t<bool, Conf::dim * 2> damping(this->m_damping);

  if (this->m_update_b) {
    // compute_b_update_explicit_cu(*(this->B), *(this->B), *(this->E), dt);
    compute_b_update_explicit_pml_cu(B, *(this->m_tmp_b1), *(this->m_tmp_b2), E,
                                 dt, this->m_pml_length, damping);
    // Communicate the new B values to guard cells
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(B);
  }
}

template <typename Conf>
void
field_solver_cu<Conf>::compute_divs_e_b() {
  if (this->m_comm != nullptr) {
    vec_t<bool, Conf::dim * 2> is_boundary(
        this->m_comm->domain_info().is_boundary);
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  } else {
    // bool is_boundary[4] = {true, true, true, true};
    vec_t<bool, Conf::dim * 2> is_boundary;
    is_boundary = true;
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  }

}

template <typename Conf>
void
field_solver_cu<Conf>::update_explicit(double dt, double time) {
  Logger::print_detail("Running explicit Cartesian solver!");
  // dt *= 1.025;
  // if (time < TINY) {
  //   compute_e_update_explicit_pml_cu(*(this->E), *(this->m_tmp_e1), *(this->m_tmp_e2), *(this->B), *(this->J),
  //                                0.5f * dt, this->m_pml_length, damping);
  //   if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  // }

  compute_b_update_pml(*(this->B), *(this->E), dt);
  compute_e_update_pml(*(this->E), *(this->B), *(this->J), dt);

  auto step = sim_env().get_step();
  if (step % this->m_data_interval == 0) {
    compute_divs_e_b();
    // auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
    // auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
    compute_flux(*(this->flux), *(this->Btotal), this->m_grid);
    compute_EB_sqr(*(this->E_sqr), *(this->B_sqr), *(this->Etotal), *(this->Btotal));
  }
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Conf>
void
field_solver_cu<Conf>::update_semi_implicit(double dt, double alpha,
                                            double beta, double time) {
  Logger::print_detail("Running semi-implicit Cartesian solver!");
  this->m_tmp_b2->init();
  // set m_tmp_b1 to B
  this->m_tmp_b1->copy_from(*(this->B));

  // Assemble the RHS
  compute_double_curl(*(this->m_tmp_b2), *(this->m_tmp_b1),
                      -alpha * beta * dt * dt);
  if (this->m_comm != nullptr)
    this->m_comm->send_guard_cells(*(this->m_tmp_b2));
  this->m_tmp_b1->add_by(*(this->m_tmp_b2));

  // Send guard cells for m_tmp_b1

  compute_implicit_rhs(*(this->m_tmp_b1), *(this->E), *(this->J), alpha, beta,
                       dt);
  if (this->m_comm != nullptr)
    this->m_comm->send_guard_cells(*(this->m_tmp_b1));

  // Since we need to iterate, define a double buffer to switch quickly between
  // operand and result.
  // clean_field(*(this->m_tmp_b1));
  this->m_bnew->copy_from(*(this->m_tmp_b1));

  auto buffer = make_double_buffer(*(this->m_tmp_b1), *(this->m_tmp_b2));
  for (int i = 0; i < 6; i++) {
    compute_double_curl(buffer.alt(), buffer.main(), -beta * beta * dt * dt);

    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(buffer.alt());
    this->m_bnew->add_by(buffer.alt());

    buffer.swap();
  }
  // m_bnew now holds B^{n+1}
  // add_alpha_beta_cu(buffer.main(), *(this->B), *(this->m_bnew), alpha, beta);
  select_dev(buffer.main()[0]) =
      alpha * this->B->at(0) + beta * this->m_bnew->at(0);
  select_dev(buffer.main()[1]) =
      alpha * this->B->at(1) + beta * this->m_bnew->at(1);
  select_dev(buffer.main()[2]) =
      alpha * this->B->at(2) + beta * this->m_bnew->at(2);

  // buffer.main() now holds alpha*B^n + beta*B^{n+1}. Compute E explicitly from
  // this
  compute_e_update_explicit_cu(*(this->E), *(this->E), buffer.main(),
                               *(this->J), dt);

  // Communicate E
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  this->B->copy_from(*(this->m_bnew));

  clean_field(*(this->B));
  clean_field(*(this->E));

  if (this->m_comm != nullptr) {
    vec_t<bool, Conf::dim * 2> is_boundary(
        this->m_comm->domain_info().is_boundary);
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  } else {
    // bool is_boundary[4] = {true, true, true, true};
    vec_t<bool, Conf::dim * 2> is_boundary;
    is_boundary = true;
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  }

  auto step = sim_env().get_step();
  if (step % this->m_data_interval == 0) {
    // auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
    // auto& grid = dynamic_cast<const grid_curv_t<Conf>&>(this->m_grid);
    compute_flux(*(this->flux), *(this->Btotal), this->m_grid);
    compute_EB_sqr(*(this->E_sqr), *(this->B_sqr), *(this->Etotal), *(this->Btotal));
  }
}

template <typename Conf>
void
field_solver_cu<Conf>::update_semi_implicit_old(double dt, double alpha,
                                                double beta, double time) {
  compute_e_update_explicit_cu(*(this->m_enew), *(this->E), *(this->B),
                               *(this->J), dt);
  compute_b_update_explicit_cu(*(this->m_bnew), *(this->B), *(this->E), dt);

  if (this->m_comm != nullptr) {
    this->m_comm->send_guard_cells(*(this->m_enew));
    this->m_comm->send_guard_cells(*(this->m_bnew));
  }

  // Start iterating a few times
  const int n_iteration = 4;
  for (int i = 0; i < n_iteration; i++) {
    add_alpha_beta_cu(*(this->m_tmp_e1), *(this->E), *(this->m_enew), alpha,
                      beta);
    add_alpha_beta_cu(*(this->m_tmp_b1), *(this->B), *(this->m_bnew), alpha,
                      beta);

    compute_e_update_explicit_cu(*(this->m_enew), *(this->E), *(this->m_tmp_b1),
                                 *(this->J), dt);
    compute_b_update_explicit_cu(*(this->m_bnew), *(this->B), *(this->m_tmp_e1),
                                 dt);

    if (this->m_comm != nullptr) {
      this->m_comm->send_guard_cells(*(this->m_enew));
      this->m_comm->send_guard_cells(*(this->m_bnew));
    }
  }

  this->E->copy_from(*(this->m_enew));
  this->B->copy_from(*(this->m_bnew));

  if (this->m_comm != nullptr) {
    vec_t<bool, Conf::dim * 2> is_boundary(
        this->m_comm->domain_info().is_boundary);
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  } else {
    // bool is_boundary[4] = {true, true, true, true};
    vec_t<bool, Conf::dim * 2> is_boundary;
    is_boundary = true;
    compute_divs_cu(*(this->divE), *(this->divB), *(this->E), *(this->B),
                    is_boundary);
  }

  CudaSafeCall(cudaDeviceSynchronize());
}

INSTANTIATE_WITH_CONFIG(field_solver_cu);

}  // namespace Aperture
