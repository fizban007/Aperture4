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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver.h"
#include "systems/gather_momentum_space.h"
#include "systems/ptc_updater_base.h"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template <typename Conf>
void initial_condition_single_stream(vector_field<Conf> &B,
                                     vector_field<Conf> &E,
                                     particle_data_t &ptc, rng_states_t &states);

}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf> comm;
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian>>(grid,
                                                                       comm);
  auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta;
  particle_data_t *ptc;
  rng_states_t *states;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("Edelta", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rand_states", &states);

  // set_initial_condition(env, *B0, *ptc, *states, 10, 1.0);
  initial_condition_single_stream(*Bdelta, *Edelta, *ptc, *states);

  env.run();
  return 0;
}
