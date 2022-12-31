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

// #include "data/curand_states.h"
#include "data/rng_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "initial_condition.h"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
set_initial_condition(const grid_sph_t<Conf>& grid) {
  using value_t = typename Conf::value_t;

  particle_data_t* ptc;
  vector_field<Conf>*B0, *B;
  rng_states_t<exec_tags::device>* states;
  sim_env().get_data("particles", &ptc);
  sim_env().get_data("B0", &B0);
  sim_env().get_data("B", &B);
  sim_env().get_data("rng_states", &states);

  double Bp = sim_env().params().get_as<double>("Bp", 10000.0);

  int mult = sim_env().params().get_as<int64_t>("multiplicity", 10);
  value_t rho0 = sim_env().params().get_as<double>("rho0", Bp * 0.01);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  double weight = rho0 / mult / q_e;

  if (ptc != nullptr && states != nullptr) {
    auto num = ptc->number();
    using idx_t = typename Conf::idx_t;

  auto injector = sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);

  injector->inject(
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      [mult] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * mult;
      },
      [] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                    PtcType type) {
        return vec_t<typename Conf::value_t, 3>(0.0f, 0.0f, 0.0f);
      },
      // [weight] __device__(auto &pos, auto &grid, auto &ext) {
      [weight] __device__(auto& x_global) {
        value_t r = grid_sph_t<Conf>::radius(x_global[0]);
        // weight scaling as 1/r meaning rho will scale as 1/r^4
        return weight * math::sin(x_global[1]) / r;
      });
  }

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });
  B->copy_from(*B0);
}

template void set_initial_condition<Config<2>>(
    const grid_sph_t<Config<2>>& grid);

}  // namespace Aperture
