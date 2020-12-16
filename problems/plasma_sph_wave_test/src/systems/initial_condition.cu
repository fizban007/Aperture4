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

#include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "initial_condition.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
set_initial_condition(sim_environment& env, const grid_sph_t<Conf>& grid,
                      int mult, double weight, double Bp) {
  particle_data_t* ptc;
  vector_field<Conf> *B0, *B;
  curand_states_t* states;
  env.get_data("particles", &ptc);
  env.get_data("B0", &B0);
  env.get_data("B", &B);
  env.get_data("rand_states", &states);

  if (ptc != nullptr && states != nullptr) {
    auto num = ptc->number();
    using idx_t = typename Conf::idx_t;

    kernel_launch(
        [num] __device__(auto ptc, auto states, auto mult, auto weight) {
          auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
          auto ext = grid.extent();
          int id = threadIdx.x + blockIdx.x * blockDim.x;
          cuda_rng_t rng(&states[id]);
          for (auto n : grid_stride_range(0, ext.size())) {
            auto idx = idx_t(n, ext);
            auto pos = idx.get_pos();
            if (grid.is_in_bound(pos)) {
              for (int i = 0; i < mult; i++) {
                uint32_t offset = num + idx.linear * mult * 2 + i * 2;

                ptc.x1[offset] = ptc.x1[offset + 1] = rng();
                ptc.x2[offset] = ptc.x2[offset + 1] = rng();
                ptc.x3[offset] = ptc.x3[offset + 1] = 0.0;
                Scalar theta = grid.template pos<1>(pos[1], ptc.x2[offset]);
                ptc.p1[offset] = ptc.p1[offset + 1] = 0.0;
                ptc.p2[offset] = ptc.p2[offset + 1] = 0.0;
                ptc.p3[offset] = ptc.p3[offset + 1] = 0.0;
                ptc.E[offset] = ptc.E[offset + 1] = 1.0;
                ptc.cell[offset] = ptc.cell[offset + 1] = idx.linear;
                ptc.weight[offset] = ptc.weight[offset + 1] = sin(theta) * weight;
                ptc.flag[offset] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                     PtcType::electron);
                ptc.flag[offset + 1] = set_ptc_type_flag(flag_or(PtcFlag::primary),
                                                         PtcType::positron);
              }
            }
          }
        },
        ptc->dev_ptrs(), states->states(), mult, weight);
    CudaSafeCall(cudaDeviceSynchronize());
    ptc->set_num(num + mult * 2 * grid.extent().size());
    Logger::print_info("ptc has number {}", ptc->number());
  }

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    return Bp / (r * r);
    // return Bp * 2.0 * cos(theta) / cube(r);
  });
  // B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
  //   Scalar r = std::exp(x);
  //   // return Bp / (r * r);
  //   return Bp * sin(theta) / cube(r);
  // });
  B->copy_from(*B0);
}

template void set_initial_condition<Config<2>>(
    sim_environment& env, const grid_sph_t<Config<2>>& grid, int mult,
    double weight, double Bp);

}  // namespace Aperture
