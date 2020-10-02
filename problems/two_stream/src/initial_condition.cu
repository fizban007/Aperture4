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
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
initial_condition_two_stream(sim_environment &env, vector_field<Conf> &B,
                             vector_field<Conf> &E, vector_field<Conf> &B0,
                             particle_data_t &ptc, curand_states_t &states, int mult,
                             Scalar p_init) {
  Scalar Bp = env.params().get_as<double>("Bp", 5000.0);
  Scalar q_e = env.params().get_as<double>("q_e", 1.0);

  // Scalar j = env.params().get_as<double>("j_parallel", 100.0);
  Scalar p0 = env.params().get_as<double>("p_0", 0.0);
  Scalar gamma0 = math::sqrt(1.0 + p0 * p0);
  Scalar beta0 = p0 / gamma0;
  Scalar j = (mult - 5) * q_e * beta0;
  Scalar delta_p = env.params().get_as<double>("delta_p", 0.0);
  // Scalar p0 = env.params().get_as<double>("p_0", 0.0);
  auto& grid = B.grid();

  B0.set_values(
      0, [Bp](Scalar x, Scalar y, Scalar z) { return Bp; });
  B.set_values(
      2, [j](Scalar x, Scalar y, Scalar z) { return 0.5 * j * (2.0 * y - 1.0); });
  // E.set_values(
      // 0, [Bp](Scalar x, Scalar y, Scalar z) { return 0.01*Bp; });

  auto num = ptc.number();

  kernel_launch(
      [mult, num, q_e, p0, delta_p, Bp] __device__(auto ptc, auto states, auto w) {
        // int mult = 1;
        auto &grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        Scalar gamma0 = sqrt(1.0f + p0 * p0);
        Scalar beta0 = p0 / gamma0;
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = idx_col_major_t<Conf::dim>(n, ext);
          auto pos = idx.get_pos();
          auto idx_row = idx_row_major_t<Conf::dim>(pos, ext);
          // if (pos[0] > grid.dims[0] * 0.99) continue;
          if (grid.is_in_bound(pos)) {
            for (int i = 0; i < mult; i++) {
              uint32_t offset = num + idx_row.linear * mult * 2 + i * 2;
              ptc.x1[offset] = ptc.x1[offset + 1] = rng();
              ptc.x2[offset] = ptc.x2[offset + 1] = rng();
              ptc.x3[offset] = ptc.x3[offset + 1] = 0.0f;

              Scalar x = grid.template pos<0>(pos[0], ptc.x1[offset]);
              Scalar y = grid.template pos<1>(pos[1], ptc.x2[offset]);

              // ptc.p1[offset] = ptc.p1[offset + 1] = 0.0f;
              // Scalar u = rng();
              // if (u < 0.5f) {
              //   // ptc.p1[offset] = ptc.p1[offset + 1] = p_init;
              //   ptc.p1[offset] = p_init + 0.1f * p_init * (rng() - 0.5f);
              //   ptc.p1[offset + 1] = p_init + 0.1f * p_init * (rng() - 0.5f);
              // } else {
              //   ptc.p1[offset] = -p_init + 0.1f * p_init * (rng() - 0.5f);
              //   ptc.p1[offset + 1] = -p_init + 0.1f * p_init * (rng() - 0.5f);
              // }
              ptc.p1[offset + 1] = rng.gaussian(delta_p);
              Scalar p = rng.gaussian(0.3 * delta_p);
              Scalar g = sqrt(1.0f + p * p);
              if (i < 5)
                ptc.p1[offset] = rng.gaussian(delta_p);
              else
                ptc.p1[offset] = gamma0 * (p - g * beta0);
              ptc.p2[offset] = ptc.p2[offset + 1] = 0.0f;
              ptc.p3[offset] = ptc.p3[offset + 1] = 0.0f;
              ptc.E[offset] = math::sqrt(1.0f + ptc.p1[offset] * ptc.p1[offset]);
              ptc.E[offset + 1] = math::sqrt(1.0f + ptc.p1[offset + 1] * ptc.p1[offset + 1]);
              ptc.weight[offset] = ptc.weight[offset + 1] = w;

              ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;

              // Scalar x = grid.template pos<0>(pos[0], ptc.x1[offset]);
              ptc.flag[offset] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                   PtcType::electron);
              ptc.flag[offset + 1] = set_ptc_type_flag(
                  flag_or(PtcFlag::primary), PtcType::positron);
            }
          }
        }
      },
      ptc.dev_ptrs(), states.states(), 1.0);
  CudaSafeCall(cudaDeviceSynchronize());
  ptc.set_num(num + 2 * mult * grid.extent().size());
}

template void initial_condition_two_stream<Config<2>>(
    sim_environment &env, vector_field<Config<2>> &B,
    vector_field<Config<2>> &E, vector_field<Config<2>> &B0,
    particle_data_t &ptc, curand_states_t &states, int mult, Scalar p_init);


} // namespace Aperture
