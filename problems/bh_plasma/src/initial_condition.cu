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

#include "data/fields.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/grid_ks.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include "systems/physics/wald_solution.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
initial_vacuum_wald(sim_environment &env, vector_field<Conf> &B0,
                    vector_field<Conf> &D0, const grid_ks_t<Conf> &grid) {
  Scalar Bp = 1.0;
  env.params().get_value("Bp", Bp);
  Scalar a = 0.0;
  env.params().get_value("bh_spin", a);

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

template void initial_vacuum_wald(sim_environment &env,
                                  vector_field<Config<2>> &B0,
                                  vector_field<Config<2>> &D0,
                                  const grid_ks_t<Config<2>> &grid);

}  // namespace Aperture
