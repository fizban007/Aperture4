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

#include "core/math.hpp"
#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

struct alfven_wave_solution {
  Scalar sinth = 0.1;
  Scalar lambda_x = 1.0;
  Scalar y0 = 1.0;
  Scalar delta_y = 1.0;
  Scalar B0 = 5000;

  Scalar costh;
  Scalar lambda;
  Scalar delta_eta;
  Scalar eta0;

  Scalar xi(Scalar x, Scalar y) const { return x * sinth + y * costh; }

  Scalar eta(Scalar x, Scalar y) const { return x * costh - y * sinth; }

  alfven_wave_solution(Scalar sinth_, Scalar lambda_x_, Scalar y0_,
                       Scalar delta_y_, Scalar B0_)
      : sinth(sinth_),
        lambda_x(lambda_x_),
        y0(y0_),
        delta_y(delta_y_),
        B0(B0_) {
    costh = math::sqrt(1.0f - sinth * sinth);
    lambda = lambda_x / sinth;
    delta_eta = delta_y * sinth;
    eta0 = eta(0.0, y0);
  }

  Scalar wave_arg(Scalar t, Scalar x, Scalar y) const {
    return 2.0 * M_PI *
           clamp<Scalar>((xi(x, y) - t + eta(x, y) * costh / sinth) / lambda,
                         0.0, 1.0);
  }

  Scalar width_arg(Scalar x, Scalar y) const {
    return clamp<Scalar>((eta(x, y) - eta0) / delta_eta, 0.0, 1.0);
  }

  Scalar width_prof(Scalar w) const { return square(math::sin(M_PI * w)); }

  Scalar d_width(Scalar w) const {
    return 2.0 * M_PI * math::sin(M_PI * w) * math::cos(M_PI * w) / delta_eta;
  }

  Scalar Bz(Scalar t, Scalar x, Scalar y) const {
    return B0 * math::sin(wave_arg(t, x, y)) * width_prof(width_arg(x, y));
  }

  Scalar Ex(Scalar t, Scalar x, Scalar y) const {
    return costh * (-Bz(t, x, y));
  }

  Scalar Ey(Scalar t, Scalar x, Scalar y) const {
    return sinth * (Bz(t, x, y));
  }

  Scalar Jx(Scalar t, Scalar x, Scalar y) const {
    return -B0 * sinth *
           (2.0f * M_PI * math::cos(wave_arg(t, x, y)) * costh / lambda /
                sinth +
            d_width(width_arg(x, y)));
  }

  Scalar Jy(Scalar t, Scalar x, Scalar y) const {
    return -B0 * costh *
           (2.0f * M_PI * math::cos(wave_arg(t, x, y)) * costh / lambda /
                sinth +
            d_width(width_arg(x, y)));
  }
};

template <typename Conf>
void
set_initial_condition(sim_environment &env, vector_field<Conf> &B0,
                      particle_data_t &ptc, curand_states_t &states, int mult,
                      Scalar weight) {
  auto Bp = env.params().get_as<double>("Bp", 1000.0);
  auto muB = env.params().get_as<double>("muB", 1.0);
  B0.set_values(0,
                [Bp, muB](Scalar x, Scalar y, Scalar z) { return Bp * muB; });
  B0.set_values(1, [Bp, muB](Scalar x, Scalar y, Scalar z) {
    return Bp * math::sqrt(1.0 - muB);
  });
  // pusher->fill_multiplicity(mult, weight);
  // ptc->append_dev({0.0f, 0.0f, 0.0f}, {0.0f, 100.0f, 0.0f}, 200 + 258 *
  // grid->dims[0],
  //                 100.0, set_ptc_type_flag(0, PtcType::positron));

  auto num = ptc.number();
  kernel_launch(
      [num, mult, weight] __device__(auto ptc, auto states) {
        auto &grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = Conf::idx(n, ext);
          auto pos = idx.get_pos();
          if (pos[0] > grid.dims[0] / 2) continue;
          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < mult; i++) {
              uint32_t offset = num + idx.linear * mult * 2 + i * 2;

              ptc.x1[offset] = ptc.x1[offset + 1] = rng();
              ptc.x2[offset] = ptc.x2[offset + 1] = rng();
              ptc.x3[offset] = ptc.x3[offset + 1] = rng();
              ptc.p1[offset] = ptc.p1[offset + 1] = 0.0;
              ptc.p2[offset] = ptc.p2[offset + 1] = 0.0;
              ptc.p3[offset] = ptc.p3[offset + 1] = 0.0;
              ptc.E[offset] = ptc.E[offset + 1] = 1.0;
              ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
              Scalar x = grid.template pos<0>(pos[0], ptc.x1[offset]);
              ptc.weight[offset] = ptc.weight[offset + 1] = cube(
                  math::abs(0.5f * grid.sizes[0] - x) * 2.0f / grid.sizes[0]);
              ptc.flag[offset] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                   PtcType::electron);
              ptc.flag[offset + 1] = set_ptc_type_flag(
                  flag_or(PtcFlag::primary), PtcType::positron);
            }
          }
        }
      },
      ptc.dev_ptrs(), states.states());
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.set_num(num + mult * 2 * B0.grid().extent().size());
}

template <typename Conf>
void
initial_condition_wave(sim_environment &env, vector_field<Conf> &B,
                       vector_field<Conf> &E, vector_field<Conf> &B0,
                       particle_data_t &ptc, curand_states_t &states, int mult,
                       Scalar weight) {
  Scalar sinth = env.params().get_as<double>("muB", 0.1);
  Scalar Bp = env.params().get_as<double>("Bp", 5000.0);
  Scalar Bwave = 0.1 * Bp;

  alfven_wave_solution wave(sinth, 0.5, 4.0, 2.0, Bwave);

  B0.set_values(
      0, [Bp, sinth](Scalar x, Scalar y, Scalar z) { return Bp * sinth; });
  B0.set_values(1, [Bp, sinth](Scalar x, Scalar y, Scalar z) {
    return Bp * math::sqrt(1.0 - sinth * sinth);
  });
  B.set_values(
      2, [wave](Scalar x, Scalar y, Scalar z) { return wave.Bz(0.0, x, y); });
  E.set_values(
      0, [wave](Scalar x, Scalar y, Scalar z) { return wave.Ex(0.0, x, y); });
  E.set_values(
      1, [wave](Scalar x, Scalar y, Scalar z) { return wave.Ey(0.0, x, y); });

  // Compute injection number per cell
  multi_array<int, Conf::dim> num_per_cell;
  multi_array<int, Conf::dim> cum_num_per_cell;

  // Actually inject particles
  auto num = ptc.number();
  kernel_launch(
      [num, mult, weight] __device__(auto ptc, auto states) {
        auto &grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = idx_col_major_t<Conf::dim>(n, ext);
          auto pos = idx.get_pos();
          auto idx_row = idx_row_major_t<Conf::dim>(pos, ext);
          if (pos[0] > grid.dims[0] / 2) continue;
          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < mult; i++) {
              uint32_t offset = num + idx_row.linear * mult * 2 + i * 2;

              ptc.x1[offset] = ptc.x1[offset + 1] = rng();
              ptc.x2[offset] = ptc.x2[offset + 1] = rng();
              ptc.x3[offset] = ptc.x3[offset + 1] = rng();
              ptc.p1[offset] = ptc.p1[offset + 1] = 0.0;
              ptc.p2[offset] = ptc.p2[offset + 1] = 0.0;
              ptc.p3[offset] = ptc.p3[offset + 1] = 0.0;
              ptc.E[offset] = ptc.E[offset + 1] = 1.0;
              ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
              Scalar x = grid.template pos<0>(pos[0], ptc.x1[offset]);
              ptc.weight[offset] = ptc.weight[offset + 1] = cube(
                  math::abs(0.5f * grid.sizes[0] - x) * 2.0f / grid.sizes[0]);
              ptc.flag[offset] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                   PtcType::electron);
              ptc.flag[offset + 1] = set_ptc_type_flag(
                  flag_or(PtcFlag::primary), PtcType::positron);
            }
          }
        }
      },
      ptc.dev_ptrs(), states.states());
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.set_num(num + mult * 2 * B0.grid().extent().size() / 2);
}

template void set_initial_condition<Config<2>>(sim_environment &env,
                                               vector_field<Config<2>> &B0,
                                               particle_data_t &ptc,
                                               curand_states_t &states,
                                               int mult, Scalar weight);

template void initial_condition_wave<Config<2>>(
    sim_environment &env, vector_field<Config<2>> &B,
    vector_field<Config<2>> &E, vector_field<Config<2>> &B0,
    particle_data_t &ptc, curand_states_t &states, int mult, Scalar weight);

}  // namespace Aperture
