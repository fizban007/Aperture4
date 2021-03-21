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

#include "core/math.hpp"
#include "core/random.h"
#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace {} // namespace

namespace Aperture {

template <typename Conf>
void harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                          rng_states_t &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = grid.sizes[1];

  // Initialize the magnetic field values
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return B0 * tanh(y / delta);
  });

  // Compute how many particles to initialize in every cell
  multi_array<int, Conf::dim> num_per_cell(ext);
  multi_array<int, Conf::dim> cum_num_per_cell(ext);

  kernel_launch(
      [delta, n_cs, n_upstream] __device__(auto num_per_cell) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            num_per_cell[idx] = 2 * n_upstream;

            value_t y = grid.template pos<1>(pos, 0.5f);
            value_t cs_y = 3.0f * delta;
            if (math::abs(y) < cs_y) {
              num_per_cell[idx] += 2 * n_cs;
            }
          }
        }
      },
      num_per_cell.dev_ndptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  thrust::device_ptr<int> p_num_per_cell(num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_cell(cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_cell, p_num_per_cell + ext.size(),
                         p_cum_num_per_cell);
  CudaCheckError();
  num_per_cell.copy_to_host();
  cum_num_per_cell.copy_to_host();
  int new_particles =
      (cum_num_per_cell[ext.size() - 1] + num_per_cell[ext.size() - 1]);
  Logger::print_info("Initializing {} particles", new_particles);

  // kernel_launch(
  using policy = exec_policy_cuda<Conf>;
  policy::launch(
      [delta, kT_cs, beta_d, n_cs, n_upstream, B0,
       q_e] __device__(auto ptc, rand_state *states, auto num_per_cell,
                       auto cum_num_per_cell) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        rng_t rng(states);
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = Conf::idx(cell, ext);
          auto pos = get_pos(idx, ext);

          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < num_per_cell[idx]; i += 2) {
              uint32_t offset_e = cum_num_per_cell[idx] + i;
              uint32_t offset_p = offset_e + 1;

              if (i < 2 * n_upstream) {
                ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                auto p1 = rng.gaussian<value_t>(2.0e-2f);
                auto p2 = rng.gaussian<value_t>(2.0e-2f);
                auto p3 = rng.gaussian<value_t>(2.0e-2f);
                ptc.p1[offset_e] = p1;
                ptc.p2[offset_e] = p2;
                ptc.p3[offset_e] = p3;
                ptc.E[offset_e] =
                    math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

                p1 = rng.gaussian<value_t>(2.0e-2f);
                p2 = rng.gaussian<value_t>(2.0e-2f);
                p3 = rng.gaussian<value_t>(2.0e-2f);
                // // p1 = curand_normal(&local_state) * 2.0e-2;
                // // p2 = curand_normal(&local_state) * 2.0e-2;
                // // p3 = curand_normal(&local_state) * 2.0e-2;
                ptc.p1[offset_p] = p1;
                ptc.p2[offset_p] = p2;
                ptc.p3[offset_p] = p3;
                ptc.E[offset_p] =
                    math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

                ptc.weight[offset_e] = ptc.weight[offset_p] = 1.0f / n_upstream;
                ptc.cell[offset_e] = idx.linear;
                ptc.cell[offset_p] = idx.linear;
                ptc.flag[offset_e] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial), PtcType::electron);
                ptc.flag[offset_p] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial), PtcType::positron);

              } else {
                auto y = grid.pos(1, pos[1], 0.5f);
                value_t j = -B0 / delta / square(cosh(y / delta));
                value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);

                ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                vec_t<value_t, 3> u_d =
                    rng.maxwell_juttner_drifting(kT_cs, beta_d);
                // auto u_d = rng.maxwell_juttner_3d(kT_cs);
                value_t sign = (y < 0 ? 1.0f : -1.0f);

                ptc.p1[offset_e] = u_d[1] * sign;
                ptc.p2[offset_e] = u_d[2] * sign;
                ptc.p3[offset_e] = u_d[0] * sign;
                ptc.E[offset_e] = math::sqrt(1.0f + u_d.dot(u_d));

                u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
                ptc.p1[offset_p] = -u_d[1] * sign;
                ptc.p2[offset_p] = -u_d[2] * sign;
                ptc.p3[offset_p] = -u_d[0] * sign;
                ptc.E[offset_p] = math::sqrt(1.0f + u_d.dot(u_d));

                ptc.weight[offset_e] = ptc.weight[offset_p] = w;
                ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
                ptc.flag[offset_e] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                    PtcType::electron);
                ptc.flag[offset_p] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                    PtcType::positron);
              }
            }
          }
        }
      },
      ptc, states, num_per_cell, cum_num_per_cell);
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.add_num(new_particles);
  ptc.sort_by_cell(grid.size());
  Logger::print_info("After initial condition, there are {} particles", ptc.number());
  // using value_t = typename Conf::value_t;
  // // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  // value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  // value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  // value_t beta_d =
  //     sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  // value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  // value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  // value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  // int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  // int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  // value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // // the upstream field strength
  // value_t B0 = math::sqrt(sigma);

  // // Initialize the magnetic field values
  // B.set_values(
  //     0, [B0, delta](auto x, auto y, auto z) { return B0 * tanh(y / delta); });

  // // Initialize the particles
  // auto num = ptc.number();
  // // Define a variable to hold the moving position in the photon array where we
  // // insert new photons
  // buffer<unsigned long long int> ptc_pos(1);
  // ptc_pos[0] = num;
  // ptc_pos.copy_to_device();
  // // auto policy = exec_policy_cuda<Conf>{};
  // using policy = exec_policy_cuda<Conf>;
  // policy::launch(
  //     [delta, kT_cs, beta_d, n_cs, n_upstream, B0,
  //      q_e] __device__(auto ptc, auto states, auto ptc_pos) {
  //       auto &grid = policy::grid();
  //       auto ext = grid.extent();
  //       rng_t rng(states);

  //       policy::loop(
  //           Conf::begin(ext), Conf::end(ext),
  //           [delta, kT_cs, beta_d, n_cs, n_upstream, B0, q_e, &grid, &ext,
  //            &rng] __device__(auto idx, auto &ptc, auto &ptc_pos) {
  //             auto pos = get_pos(idx, ext);
  //             if (!grid.is_in_bound(pos))
  //               return;
  //             // printf("cell %ld\n", idx.linear);

  //             // grid center position in y
  //             auto y = grid.pos(1, pos[1], 0.5f);
  //             value_t j = -B0 / delta / square(cosh(y / delta));
  //             value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
  //             // value_t w = math::abs(j) / q_e / n_cs;

  //             // Background plasma is everywhere
  //             for (int i = 0; i < n_upstream; i++) {
  //               auto offset = atomic_add(ptc_pos, 2);
  //               // auto offset = idx.linear * 2 * n_upstream + i * 2;
  //               auto offset_e = offset;
  //               auto offset_p = offset + 1;

  //               ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
  //               ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
  //               ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

  //               ptc.p1[offset_e] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.p2[offset_e] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.p3[offset_e] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.E[offset_e] = math::sqrt(1.0f + square(ptc.p1[offset_e]) +
  //                                            square(ptc.p2[offset_e]) +
  //                                            square(ptc.p3[offset_e]));

  //               ptc.p1[offset_p] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.p2[offset_p] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.p3[offset_p] = rng.gaussian<value_t>(2.0e-2);
  //               ptc.E[offset_p] = math::sqrt(1.0f + square(ptc.p1[offset_p]) +
  //                                            square(ptc.p2[offset_p]) +
  //                                            square(ptc.p3[offset_p]));

  //               // auto u = rng.maxwell_juttner_3d(1.0e-3);
  //               ptc.weight[offset_e] = ptc.weight[offset_p] = 1.0f / n_upstream;
  //               ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
  //               ptc.flag[offset_e] = set_ptc_type_flag(
  //                   flag_or(PtcFlag::initial), PtcType::electron);
  //               ptc.flag[offset_p] = set_ptc_type_flag(
  //                   flag_or(PtcFlag::initial), PtcType::positron);
  //             }

  //             value_t cs_y = 4.0f * delta;
  //             if (y > -cs_y && y < cs_y) {
  //               // Current sheet plasma
  //               for (int i = 0; i < n_cs; i++) {
  //                 // for (int i = 0; i < n_upstream; i++) {
  //                 auto offset = atomic_add(ptc_pos, 2);
  //                 // auto offset = idx.linear * 2 * n_upstream + i * 2;
  //                 auto offset_e = offset;
  //                 auto offset_p = offset + 1;

  //                 ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
  //                 ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
  //                 ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

  //                 auto u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
  //                 // auto u_d = rng.maxwell_juttner_3d(kT_cs);

  //                 ptc.p1[offset_e] = u_d[1];
  //                 ptc.p2[offset_e] = u_d[2];
  //                 ptc.p3[offset_e] = u_d[0];
  //                 ptc.E[offset_e] = math::sqrt(1.0f + u_d.dot(u_d));

  //                 // auto u_p = rng.maxwell_juttner_drifting(kT_cs, beta_d);
  //                 ptc.p1[offset_p] = -u_d[2];
  //                 ptc.p2[offset_p] = -u_d[1];
  //                 ptc.p3[offset_p] = -u_d[0];
  //                 ptc.E[offset_p] = math::sqrt(1.0f + u_d.dot(u_d));

  //                 ptc.weight[offset_e] = ptc.weight[offset_p] = w;
  //                 ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
  //                 ptc.flag[offset_e] = set_ptc_type_flag(
  //                     flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
  //                     PtcType::electron);
  //                 ptc.flag[offset_p] = set_ptc_type_flag(
  //                     flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
  //                     PtcType::positron);
  //               }
  //             }
  //           },
  //           ptc, ptc_pos);
  //     },
  //     ptc, states, ptc_pos);
  // policy::sync();
  // ptc_pos.copy_to_host();
  // ptc.set_num(ptc_pos[0]);
}

template <typename Conf>
void double_harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                                 rng_states_t &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = grid.sizes[1];

  // Initialize the magnetic field values
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    if (y < 0.0f) {
      return B0 * tanh((y + 0.25f * ysize) / delta);
    } else {
      return -B0 * tanh((y - 0.25f * ysize) / delta);
    }
  });

  // Compute how many particles to initialize in every cell
  multi_array<int, Conf::dim> num_per_cell(ext);
  multi_array<int, Conf::dim> cum_num_per_cell(ext);

  kernel_launch(
      [delta, n_cs, n_upstream] __device__(auto num_per_cell) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          if (grid.is_in_bound(pos)) {
            num_per_cell[idx] = 2 * n_upstream;

            value_t y = grid.template pos<1>(pos, 0.5f);
            value_t cs_y = 3.0f * delta;
            value_t y1 = 0.5 * grid.lower[1];
            value_t y2 = -0.5 * grid.lower[1];
            if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
              num_per_cell[idx] += 2 * n_cs;
            }
          }
        }
      },
      num_per_cell.dev_ndptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  thrust::device_ptr<int> p_num_per_cell(num_per_cell.dev_ptr());
  thrust::device_ptr<int> p_cum_num_per_cell(cum_num_per_cell.dev_ptr());

  thrust::exclusive_scan(p_num_per_cell, p_num_per_cell + ext.size(),
                         p_cum_num_per_cell);
  CudaCheckError();
  num_per_cell.copy_to_host();
  cum_num_per_cell.copy_to_host();
  int new_particles =
      (cum_num_per_cell[ext.size() - 1] + num_per_cell[ext.size() - 1]);
  Logger::print_info("Initializing {} particles", new_particles);

  // kernel_launch(
  using policy = exec_policy_cuda<Conf>;
  policy::launch(
      [delta, kT_cs, beta_d, n_cs, n_upstream, B0,
       q_e] __device__(auto ptc, rand_state *states, auto num_per_cell,
                       auto cum_num_per_cell) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        rng_t rng(states);
        for (auto cell : grid_stride_range(0, ext.size())) {
          auto idx = Conf::idx(cell, ext);
          auto pos = get_pos(idx, ext);

          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < num_per_cell[idx]; i += 2) {
              uint32_t offset_e = cum_num_per_cell[idx] + i;
              uint32_t offset_p = offset_e + 1;

              if (i < 2 * n_upstream) {
                ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                auto p1 = rng.gaussian<value_t>(2.0e-2f);
                auto p2 = rng.gaussian<value_t>(2.0e-2f);
                auto p3 = rng.gaussian<value_t>(2.0e-2f);
                ptc.p1[offset_e] = p1;
                ptc.p2[offset_e] = p2;
                ptc.p3[offset_e] = p3;
                ptc.E[offset_e] =
                    math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

                p1 = rng.gaussian<value_t>(2.0e-2f);
                p2 = rng.gaussian<value_t>(2.0e-2f);
                p3 = rng.gaussian<value_t>(2.0e-2f);
                // // p1 = curand_normal(&local_state) * 2.0e-2;
                // // p2 = curand_normal(&local_state) * 2.0e-2;
                // // p3 = curand_normal(&local_state) * 2.0e-2;
                ptc.p1[offset_p] = p1;
                ptc.p2[offset_p] = p2;
                ptc.p3[offset_p] = p3;
                ptc.E[offset_p] =
                    math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

                ptc.weight[offset_e] = ptc.weight[offset_p] = 1.0f / n_upstream;
                ptc.cell[offset_e] = idx.linear;
                ptc.cell[offset_p] = idx.linear;
                ptc.flag[offset_e] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial), PtcType::electron);
                ptc.flag[offset_p] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial), PtcType::positron);

              } else {
                auto y = grid.pos(1, pos[1], 0.5f);
                value_t j;
                value_t ysize = grid.sizes[1];
                if (y < 0.0f) {
                  j = -B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
                } else {
                  j = B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
                }
                value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
                ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
                ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
                ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

                vec_t<value_t, 3> u_d =
                    rng.maxwell_juttner_drifting(kT_cs, beta_d);
                // auto u_d = rng.maxwell_juttner_3d(kT_cs);
                value_t sign = (y < 0 ? 1.0f : -1.0f);

                ptc.p1[offset_e] = u_d[1] * sign;
                ptc.p2[offset_e] = u_d[2] * sign;
                ptc.p3[offset_e] = u_d[0] * sign;
                ptc.E[offset_e] = math::sqrt(1.0f + u_d.dot(u_d));

                u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
                ptc.p1[offset_p] = -u_d[1] * sign;
                ptc.p2[offset_p] = -u_d[2] * sign;
                ptc.p3[offset_p] = -u_d[0] * sign;
                ptc.E[offset_p] = math::sqrt(1.0f + u_d.dot(u_d));

                ptc.weight[offset_e] = ptc.weight[offset_p] = w;
                ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
                ptc.flag[offset_e] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                    PtcType::electron);
                ptc.flag[offset_p] = set_ptc_type_flag(
                    flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
                    PtcType::positron);
              }
            }
          }
        }
      },
      ptc, states, num_per_cell, cum_num_per_cell);
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.add_num(new_particles);
  ptc.sort_by_cell(grid.size());
  Logger::print_info("After initial condition, there are {} particles", ptc.number());

  // Logger::print_debug("states has size {}", states.states().size());
  // Initialize the particles
  // auto num = ptc.number();
  // Define a variable to hold the moving position in the photon array where we
  // insert new photons
  // buffer<int> ptc_pos(1, MemType::host_device);
  // // ptc_pos[0] = 0;
  // ptc_pos[0] = n_upstream * grid.size() * 2;
  // ptc_pos.copy_to_device();
  // CudaSafeCall(cudaDeviceSynchronize());
  // using policy = exec_policy_cuda<Conf>;
  // // policy::launch(
  // // kernel_launch({1, 1},
  // kernel_launch(
  //     [delta, kT_cs, beta_d, n_cs, n_upstream, B0,
  //      q_e] __device__(auto ptc, rand_state *states, int *ptc_pos) {
  //       auto &grid = policy::grid();
  //       auto ext = grid.extent();
  //       rng_t rng(states);
  //       // auto id = threadIdx.x + blockIdx.x * blockDim.x;
  //       // curandState local_state = states[id];

  //       // printf("rng %f\n", rng.uniform<float>());
  //       for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext)))
  //       {
  //         auto pos = get_pos(idx, ext);
  //         if (!grid.is_in_bound(pos))
  //           continue;
  //         // printf("cell %ld\n", idx.linear);

  //         // grid center position in y
  //         auto y = grid.pos(1, pos[1], 0.5f);
  //         value_t j;
  //         value_t ysize = grid.sizes[1];
  //         if (y < 0.0f) {
  //           j = -B0 / delta / square(cosh((y + 0.25f * ysize) / delta));
  //         } else {
  //           j = B0 / delta / square(cosh((y - 0.25f * ysize) / delta));
  //         }
  //         value_t w = math::abs(j) / q_e / n_cs / (2.0f * beta_d);
  //         // value_t w = math::abs(j) / q_e / n_cs;

  //         // Background plasma is everywhere
  //         for (int n = 0; n < n_upstream; n++) {
  //         // for (int n = 0; n < 1; n++) {
  //           // unsigned int offset = Aperture::atomic_add(ptc_pos, 2);
  //           // int offset = atomicAdd(&ptc_pos[0], 2);
  //           uint64_t offset = idx.linear * n_upstream * 2 + n * 2;
  //           // if (offset + 1 > 240000000) {
  //           // printf("n_upstream is %d, pos is (%d, %d), offset is %lu\n",
  //           n_upstream, pos[0], pos[1], offset);
  //           // }
  //           size_t offset_e = offset;
  //           size_t offset_p = offset + 1;

  //           ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
  //           ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
  //           ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

  //           auto p1 = rng.gaussian<value_t>(2.0e-2f);
  //           auto p2 = rng.gaussian<value_t>(2.0e-2f);
  //           auto p3 = rng.gaussian<value_t>(2.0e-2f);
  //           ptc.p1[offset_e] = p1;
  //           ptc.p2[offset_e] = p2;
  //           ptc.p3[offset_e] = p3;
  //           ptc.E[offset_e] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           // p1 = rng.gaussian<value_t>(2.0e-2f);
  //           // p2 = rng.gaussian<value_t>(2.0e-2f);
  //           // p3 = rng.gaussian<value_t>(2.0e-2f);
  //           // // p1 = curand_normal(&local_state) * 2.0e-2;
  //           // // p2 = curand_normal(&local_state) * 2.0e-2;
  //           // // p3 = curand_normal(&local_state) * 2.0e-2;
  //           ptc.p1[offset_p] = p1;
  //           ptc.p2[offset_p] = p2;
  //           ptc.p3[offset_p] = p3;
  //           ptc.E[offset_p] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           ptc.weight[offset_e] = ptc.weight[offset_p] = 1.0f / n_upstream;
  //           ptc.cell[offset_e] = idx.linear;
  //           ptc.cell[offset_p] = idx.linear;
  //           ptc.flag[offset_e] =
  //               set_ptc_type_flag(flag_or(PtcFlag::initial),
  //               PtcType::electron);
  //           ptc.flag[offset_p] =
  //               set_ptc_type_flag(flag_or(PtcFlag::initial),
  //               PtcType::positron);
  //         }

  //         // printf("idx is %ld\n", idx.linear);
  //         value_t cs_y = 3.0f * delta;
  //         value_t y1 = 0.5 * grid.lower[1];
  //         value_t y2 = -0.5 * grid.lower[1];
  //         // Current sheet plasma
  //         if (math::abs(y - y1) < cs_y || math::abs(y - y2) < cs_y) {
  //           for (int i = 0; i < n_cs; i++) {
  //             // for (int i = 0; i < n_upstream; i++) {
  //             size_t offset = atomic_add(ptc_pos, 2);
  //             // if (offset > 240000000) {
  //               // printf("delta is %f, pos is (%d, %d), offset is %lu,
  //               ptc_pos is %d\n", delta, pos[0], pos[1], offset, *ptc_pos);
  //             // }
  //             // auto offset = idx.linear * 2 * n_upstream + i * 2;
  //             size_t offset_e = offset;
  //             size_t offset_p = offset + 1;

  //             ptc.x1[offset_e] = ptc.x1[offset_p] = rng.uniform<value_t>();
  //             ptc.x2[offset_e] = ptc.x2[offset_p] = rng.uniform<value_t>();
  //             ptc.x3[offset_e] = ptc.x3[offset_p] = rng.uniform<value_t>();

  //             vec_t<value_t, 3> u_d = rng.maxwell_juttner_drifting(kT_cs,
  //             beta_d);
  //             // auto u_d = rng.maxwell_juttner_3d(kT_cs);
  //             value_t sign = (y < 0 ? 1.0f : -1.0f);

  //             ptc.p1[offset_e] = u_d[1] * sign;
  //             ptc.p2[offset_e] = u_d[2] * sign;
  //             ptc.p3[offset_e] = u_d[0] * sign;
  //             ptc.E[offset_e] = math::sqrt(1.0f + u_d.dot(u_d));

  //             // u_d = rng.maxwell_juttner_drifting(kT_cs, beta_d);
  //             ptc.p1[offset_p] = -u_d[2] * sign;
  //             ptc.p2[offset_p] = u_d[1] * sign;
  //             ptc.p3[offset_p] = -u_d[0] * sign;
  //             ptc.E[offset_p] = math::sqrt(1.0f + u_d.dot(u_d));

  //             ptc.weight[offset_e] = ptc.weight[offset_p] = w;
  //             ptc.cell[offset_e] = ptc.cell[offset_p] = idx.linear;
  //             ptc.flag[offset_e] = set_ptc_type_flag(
  //                 flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
  //                 PtcType::electron);
  //             ptc.flag[offset_p] = set_ptc_type_flag(
  //                 flag_or(PtcFlag::initial, PtcFlag::exclude_from_spectrum),
  //                 PtcType::positron);
  //           }
  //         }
  //         // states[id] = local_state;
  //       }
  //       // Conf::begin(ext), Conf::end(ext),
  //       // [delta, kT_cs, beta_d, n_cs, n_upstream, B0, q_e, &grid, &ext]
  //       // __device__(auto idx, auto &ptc, auto &ptc_pos, auto &rng) { ptc,
  //       // ptc_pos, rng);
  //     },
  //     ptc.get_dev_ptrs(), states.states().dev_ptr(), ptc_pos.dev_ptr());
  // // policy::sync();
  // CudaSafeCall(cudaDeviceSynchronize());
  // ptc_pos.copy_to_host();
  // ptc.set_num(ptc_pos[0]);
  // Logger::print_info("Initial condition contains {} particles", ptc_pos[0]);
}

template void harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                              particle_data_t &ptc,
                                              rng_states_t &states);

template void double_harris_current_sheet<Config<2>>(vector_field<Config<2>> &B,
                                                     particle_data_t &ptc,
                                                     rng_states_t &states);

} // namespace Aperture
