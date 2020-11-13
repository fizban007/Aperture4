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
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
initial_nonrotating_vacuum_wald(sim_environment& env, vector_field<Conf>& B0,
                                const grid_ks_t<Conf>& grid) {
  Scalar Bp = 1.0;
  env.params().get_value("Bp", Bp);

  kernel_launch(
      [Bp] __device__(auto B, auto a) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          auto r = grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], false));
          auto r_s =
              grid_ks_t<Conf>::radius(grid.template pos<0>(pos[0], true));
          auto th = grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], false));
          auto th_s =
              grid_ks_t<Conf>::theta(grid.template pos<1>(pos[1], true));
          if (th_s < TINY) th_s += 1.0e-4;

          B[2][idx] = 0.0f;

          auto r2 = r_s * r_s;
          B[0][idx] =
              0.5f * Bp *
              (a * a + r2 - 2.0f * r_s +
               (8.0f * r_s * square(a * a + r2)) /
                   square(a * a + 2.0f * r2 + a * a * math::cos(2.0f * th))) *
              math::sin(2.0f * th) / Metric_KS::sqrt_gamma(a, r_s, th);
          // B[0][idx] = 2.0 * Bp * math::cos(th);

          auto sth2 = square(math::sin(th_s));
          auto cth2 = square(math::cos(th_s));
          r2 = r * r;
          B[1][idx] =
              -0.5f * Bp *
              (2.0f * r * (a * a + r2) * (r2 + a * a * math::cos(2.0f * th_s)) *
                   sth2 -
               2.0f * a * a * sth2 * sth2 *
                   (r2 - a * a * r + a * a * (r - 1.0f) * cth2)) /
              (square(r2 + a * a * cth2) * Metric_KS::sqrt_gamma(a, r, th_s));
          // B[1][idx] = 2.0 * Bp * math::sin(th_s) / r;
        }
      },
      B0.get_ptrs(), 0.0f);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template void initial_nonrotating_vacuum_wald(sim_environment& env,
                                              vector_field<Config<2>>& B0,
                                              const grid_ks_t<Config<2>>& grid);

}  // namespace Aperture
