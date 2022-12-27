/*
 * Copyright (c) 2021 Alex Chen.
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

#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/gather_momentum_space.h"
#include "systems/ptc_injector_cuda.hpp"
#include "systems/ptc_updater_base.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include <iostream>

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Conf::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_cuda> comm;
  grid_t<Conf> grid(comm);

  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_cuda, coord_policy_cartesian, ptc_physics_policy_empty>>(
      grid, comm);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_cuda>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_cuda>>(grid, &comm);

  env.init();

  particle_data_t *ptc;
  rng_states_t<exec_tags::device> *states;
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);

  auto injector =
      sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  injector->init();

  int n = sim_env().params().get_as<int64_t>("multiplicity", 10);
  float kT = sim_env().params().get_as<double>("kT", 1.0);
  injector->inject(
      // Injection criterion
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      // Number injected
      [n] __device__(auto &pos, auto &grid, auto &ext) { return 2 * n; },
      // Initialize particles
      [kT] __device__(auto &pos, auto &grid, auto &ext, rand_state &state,
                      PtcType type) {
        vec_t<value_t, 3> u = rng_maxwell_juttner_3d(state, kT);
        return u;
      },
      // Particle weight
      [n] __device__(auto &x_global) { return 1.0 / n; });

  env.update();

  return 0;
}
