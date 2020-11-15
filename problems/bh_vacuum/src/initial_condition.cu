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
                                vector_field<Conf>& D0,
                                const grid_ks_t<Conf>& grid) {
  Scalar Bp = 1.0;
  env.params().get_value("Bp", Bp);

  kernel_launch(
      [Bp] __device__(auto B, auto D, auto a) {
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
          if (math::abs(th_s) < TINY)
            th_s = (th_s < 0.0f ? -1.0f : 1.0f) * 0.01 * grid.delta[1];

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
          auto sth = math::sin(th_s);
          auto cth = math::cos(th_s);
          r2 = r * r;
          B[1][idx] =
              -0.5f * Bp *
              (2.0f * r * (a * a + r2) * (r2 + a * a * math::cos(2.0f * th_s)) *
                   sth2 -
               2.0f * a * a * sth2 * sth2 *
                   (r2 - a * a * r + a * a * (r - 1.0f) * cth2)) /
              (square(r2 + a * a * cth2) * Metric_KS::sqrt_gamma(a, r, sth, cth));
          // B[1][idx] = 2.0 * Bp * math::sin(th_s) / r;
          // if (pos[1] == 2 && pos[0] == 10)
          //   printf("Bth is %f, gamma is %f, th_s is %f\n", B[1][idx],
          //   Metric_KS::sqrt_gamma(a, r, th_s), th_s);

          r2 = r_s * r_s;
          // D[2][idx] = -(Metric_KS::beta1(a, r_s, sth, cth) /
          // Metric_KS::alpha(a, r_s, sth, cth) / Metric_KS::sqrt_gamma(a, r,
          // sth, cth))
          D[2][idx] = -(Metric_KS::sq_gamma_beta(a, r_s, sth, cth) /
                        Metric_KS::ag_33(a, r_s, sth, cth)) *
                      (-0.5f * Bp) *
                      (2.0f * r_s * (a * a + r2) *
                           (r2 + a * a * math::cos(2.0f * th_s)) * sth2 -
                       2.0f * a * a * sth2 * sth2 *
                           (r2 - a * a * r_s + a * a * (r_s - 1.0f) * cth2)) /
                      (square(r2 + a * a * cth2) *
                       Metric_KS::sqrt_gamma(a, r_s, sth, cth));
        }
      },
      B0.get_ptrs(), D0.get_ptrs(), 0.0f);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template void initial_nonrotating_vacuum_wald(sim_environment& env,
                                              vector_field<Config<2>>& B0,
                                              vector_field<Config<2>>& D0,
                                              const grid_ks_t<Config<2>>& grid);

}  // namespace Aperture
