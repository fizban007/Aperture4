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
initial_condition_plasma(const grid_sph_t<Conf>& grid) {
  using value_t = typename Conf::value_t;

  vector_field<Conf> *B0, *B;
  rng_states_t<exec_tags::device>* states;
  sim_env().get_data("B0", &B0);
  sim_env().get_data("B", &B);
  // sim_env().get_data("rng_states", &states);

  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t Bp = sim_env().params().get_as<double>("Bp", 1000.0);
  value_t rho0 = sim_env().params().get_as<double>("rho0", 10000.0);
  // value_t omega = sim_env().params().get_as<double>("omega", 100.0);
  int mult = sim_env().params().get_as<int64_t>("multiplicity", 10);

  auto injector = sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);

  injector->inject_pairs(
      // Injection criterion
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      // Number injected
      [mult] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * mult;
      },
      // Injected distribution
      [] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                    PtcType type) {
        return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      // Particle weights
      [mult, rho0, q_e] __device__(auto &x_global) {
        value_t r = grid_sph_t<Conf>::radius(x_global[0]);
        // weight scaling as r meaning rho will scale as 1/r^2
        value_t rho_gj = rho0 * r;
        return rho_gj / q_e / mult * sin(x_global[1]);
      });

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    // return Bp / (r * r);
    return Bp * sin(theta) / cube(r);
  });
  B->copy_from(*B0);
}

template <typename Conf>
void
initial_condition_vacuum(const grid_sph_t<Conf>& grid, double Bp) {
  vector_field<Conf> *B, *B0;
  sim_env().get_data("B", &B);
  sim_env().get_data("B0", &B0);
  if (B == nullptr)
    Logger::print_err("B is nullptr!!!");

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    return Bp / (r * r);
    // return Bp * 2.0 * cos(theta) / (r * r * r);
  });
  B->copy_from(*B0);
}

template void initial_condition_plasma<Config<2>>(
    const grid_sph_t<Config<2>>& grid);

template void initial_condition_vacuum<Config<2>>(
    const grid_sph_t<Config<2>>& grid, double Bp);

}  // namespace Aperture
