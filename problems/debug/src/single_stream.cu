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

#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_base.h"
#include "systems/gather_momentum_space.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_sync_cooling.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/phys_policy_IC_cooling.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template <typename Conf>
void initial_condition_single_stream(vector_field<Conf> &B,
                                     vector_field<Conf> &E,
                                     particle_data_t &ptc,
                                     rng_states_t<exec_tags::device> &states);

}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf, exec_policy_gpu> comm;
  // grid_t<Conf> grid(comm);
  auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  // auto pusher = env.register_system<
  //     ptc_updater<Conf, exec_policy_gpu, coord_policy_cartesian>>(grid,
  //                                                                      comm);
  auto pusher = env.register_system<ptc_updater<
      // Conf, exec_policy_gpu, coord_policy_cartesian,
      // phys_policy_IC_cooling>>( Conf, exec_policy_gpu,
      // coord_policy_cartesian_sync_cooling,
      //     phys_policy_IC_cooling>>(
      Conf, exec_policy_gpu, coord_policy_cartesian_sync_cooling>>(grid, &comm);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_gpu, coord_policy_cartesian, IC_radiation_scheme>>(
      grid, &comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_gpu>>(grid);
  // auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_gpu>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta;
  particle_data_t *ptc;
  rng_states_t<exec_tags::device> *states;
  // env.get_data("B0", &B0);
  env.get_data("B", &Bdelta);
  env.get_data("E", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);

  // set_initial_condition(env, *B0, *ptc, *states, 10, 1.0);
  initial_condition_single_stream(*Bdelta, *Edelta, *ptc, *states);

  env.run();
  return 0;
}
