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
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_injector_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace Aperture {

template <typename Conf>
void
initial_condition_single_stream(vector_field<Conf> &B, vector_field<Conf> &E,
                                particle_data_t &ptc, rng_states_t &states) {
  using value_t = typename Conf::value_t;
  Scalar Bp = sim_env().params().get_as<double>("Bp", 5000.0);
  Scalar q_e = sim_env().params().get_as<double>("q_e", 1.0);

  Scalar p0 = sim_env().params().get_as<double>("p_0", 0.0);
  Scalar gamma0 = math::sqrt(1.0 + p0 * p0);
  Scalar beta0 = p0 / gamma0;
  // Scalar j = (mult - 5) * q_e * beta0;
  int n = sim_env().params().get_as<int64_t>("multiplicity", 10);
  auto &grid = B.grid();

  B.set_values(1, [Bp](Scalar x, Scalar y, Scalar z) { return Bp; });
  // E.set_values(
  // 0, [Bp](Scalar x, Scalar y, Scalar z) { return 0.01*Bp; });

  auto injector =
      sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  injector->init();

  injector->inject(
      // Injection criterion
      [] __device__(auto &pos, auto &grid, auto &ext) { return (pos[1] < grid.dims[1] / 2 - 10); },
      // Number injected
      [n] __device__(auto &pos, auto &grid, auto &ext) { return 2 * n; },
      // Initialize particles
      [p0] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                      PtcType type) {
        if (type == PtcType::electron)
          return vec_t<value_t, 3>(p0, 0.0, 0.0);
        else
          return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      // Particle weight
      [n] __device__(auto &x_global) { return 1.0 / n; });
}

template void initial_condition_single_stream(vector_field<Config<2>> &B,
                                              vector_field<Config<2>> &E,
                                              particle_data_t &ptc,
                                              rng_states_t &states);

}  // namespace Aperture
