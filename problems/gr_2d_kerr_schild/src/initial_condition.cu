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

#include "data/fields.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/grid_ks.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/wald_solution.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void initial_nonrotating_vacuum_wald(vector_field<Conf> &B0,
                                     vector_field<Conf> &D0,
                                     const grid_ks_t<Conf> &grid) {
  Scalar Bp = 1.0;
  sim_env().params().get_value("Bp", Bp);

  kernel_launch(
      [Bp] __device__(auto B, auto D, auto a) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
          auto r_s =
              grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
          auto th = grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
          auto th_s =
              grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));
          if (math::abs(th_s) < TINY)
            th_s = (th_s < 0.0f ? -1.0f : 1.0f) * 0.01 * grid.delta[1];

          B[2][idx] = 0.0f;

          auto r2 = r_s * r_s;
          B[0][idx] = Bp * r_s * r_s * math::sin(2.0f * th) /
                      Metric_KS::sqrt_gamma(a, r_s, th);

          auto sth2 = square(math::sin(th_s));
          auto cth2 = square(math::cos(th_s));
          auto sth = math::sin(th_s);
          auto cth = math::cos(th_s);
          r2 = r * r;
          B[1][idx] = -2.0 * Bp * r * square(math::sin(th_s)) /
                      Metric_KS::sqrt_gamma(a, r, th_s);
          // if (pos[1] == 2 && pos[0] == 10)
          //   printf("Bth is %f, gamma is %f, th_s is %f\n", B[1][idx],
          //   Metric_KS::sqrt_gamma(a, r, th_s), th_s);

          r2 = r_s * r_s;
          D[2][idx] = (Metric_KS::sq_gamma_beta(0.0f, r_s, sth, cth) /
                       Metric_KS::ag_33(0.0f, r_s, sth, cth)) *
                      2.0 * Bp * r_s * square(math::sin(th)) /
                      Metric_KS::sqrt_gamma(a, r_s, th);
        }
      },
      B0.get_ptrs(), D0.get_ptrs(), grid.a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                         const grid_ks_t<Conf> &grid) {
  typename Conf::value_t Bp = 1.0;
  sim_env().params().get_value("Bp", Bp);

  kernel_launch(
      [Bp] __device__(auto B, auto D, auto a) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
          auto r_s =
              grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
          auto th = grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
          auto th_s =
              grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));

          B[0][idx] = gr_wald_solution_B(a, r_s, th, Bp, 0);
          B[1][idx] = gr_wald_solution_B(a, r, th_s, Bp, 1);
          B[2][idx] = gr_wald_solution_B(a, r, th, Bp, 2);

          D[0][idx] = gr_wald_solution_D(a, r, th_s, Bp, 0);
          D[1][idx] = gr_wald_solution_D(a, r_s, th, Bp, 1);
          D[2][idx] = gr_wald_solution_D(a, r_s, th_s, Bp, 2);
        }
      },
      B0.get_ptrs(), D0.get_ptrs(), grid.a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf>
void initial_vacuum_monopole(vector_field<Conf> &B, vector_field<Conf> &D,
                             const grid_ks_t<Conf> &grid) {
  Scalar Bp = 1.0;
  sim_env().params().get_value("Bp", Bp);

  kernel_launch(
      [Bp] __device__(auto B, auto D, auto a) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
          auto r_s =
              grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
          auto th = grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
          auto th_s =
              grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));
          if (math::abs(th_s) < TINY)
            th_s = (th_s < 0.0f ? -1.0f : 1.0f) * 0.01 * grid.delta[1];

          // Both B^\phi and D^\phi are zero
          B[2][idx] = 0.0f;
          D[2][idx] = 0.0f;

          auto sth = math::sin(th);
          auto cth = math::cos(th);
          auto sth_s = math::sin(th_s);
          auto cth_s = math::cos(th_s);

          auto Br = [a] __device__(auto r, auto sth, auto cth) {
            return ((a * a + r * r) * (r * r - square(a * cth)) * sth /
                    square(r * r + square(a * cth))) /
                   Metric_KS::sqrt_gamma(a, r, sth, cth);
          };

          auto Bth = [a] __device__(auto r, auto sth, auto cth) {
            return -(2.0 * r * cth * square(a * sth) /
                     square(r * r + square(a * cth))) /
                   Metric_KS::sqrt_gamma(a, r, sth, cth);
          };

          auto Er = [a] __device__(auto r, auto sth, auto cth) {
            return -2.0 * a * r * cth / square(r * r + square(a * cth));
          };

          auto Eth = [a] __device__(auto r, auto sth, auto cth) {
            return -a * sth * (r * r - square(a * cth)) /
                   square(r * r + square(a * cth));
          };

          B[0][idx] = Bp * Br(r_s, sth, cth);
          B[1][idx] = Bp * Bth(r, sth_s, cth_s);

          D[0][idx] =
              Bp *
              (Metric_KS::gu11(a, r, sth_s, cth_s) * Er(r, sth_s, cth_s) -
               Metric_KS::gu13(a, r, sth_s, cth_s) *
                   Metric_KS::sq_gamma_beta(a, r, sth_s, cth_s) *
                   Bth(r, sth_s, cth_s)) /
              Metric_KS::alpha(a, r, sth_s, cth_s);
          D[1][idx] = Bp *
                      (Metric_KS::gu22(a, r_s, sth, cth) * Eth(r_s, sth, cth)) /
                      Metric_KS::alpha(a, r_s, sth, cth);
          D[2][idx] =
              Bp *
              (Metric_KS::gu13(a, r_s, sth_s, cth_s) * Er(r_s, sth_s, cth_s) -
               Metric_KS::gu33(a, r_s, sth_s, cth_s) *
                   Metric_KS::sq_gamma_beta(a, r_s, sth_s, cth_s) *
                   Bth(r_s, sth_s, cth_s)) /
              Metric_KS::alpha(a, r_s, sth_s, cth_s);
        }
      },
      B.get_ptrs(), D.get_ptrs(), grid.a);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template void
initial_nonrotating_vacuum_wald(vector_field<Config<2, float>> &B0,
                                vector_field<Config<2, float>> &D0,
                                const grid_ks_t<Config<2, float>> &grid);
template void
initial_nonrotating_vacuum_wald(vector_field<Config<2, double>> &B0,
                                vector_field<Config<2, double>> &D0,
                                const grid_ks_t<Config<2, double>> &grid);

template void initial_vacuum_wald(vector_field<Config<2, float>> &B0,
                                  vector_field<Config<2, float>> &D0,
                                  const grid_ks_t<Config<2, float>> &grid);
template void initial_vacuum_wald(vector_field<Config<2, double>> &B0,
                                  vector_field<Config<2, double>> &D0,
                                  const grid_ks_t<Config<2, double>> &grid);

template void initial_vacuum_monopole(vector_field<Config<2>> &B,
                                      vector_field<Config<2>> &D,
                                      const grid_ks_t<Config<2>> &grid);

} // namespace Aperture
