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

template <typename Conf>
void set_initial_condition(sim_environment &env, vector_field<Conf> &B0,
                           particle_data_t &ptc, curand_states_t &states,
                           int mult, Scalar weight) {
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
          if (pos[0] > grid.dims[0] / 2)
            continue;
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
              ptc.weight[offset] = ptc.weight[offset + 1] =
                  cube(math::abs(0.5f * grid.sizes[0] - x) * 2.0f / grid.sizes[0]);
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

template void set_initial_condition<Config<2>>(sim_environment &env,
                                               vector_field<Config<2>> &B0,
                                               particle_data_t &ptc,
                                               curand_states_t &states,
                                               int mult, Scalar weight);

} // namespace Aperture
