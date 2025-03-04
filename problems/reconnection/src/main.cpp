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

#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
// #include "systems/compute_lorentz_factor.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver.h"
#include "systems/gather_tracked_ptc.h"
// #include "systems/gather_momentum_space.h"
// #include "systems/legacy/ptc_updater_old.h"
#include "systems/policies/coord_policy_cartesian_sync_cooling.hpp"
#include "systems/ptc_updater.h"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template <typename Conf>
void harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                          rng_states_t &states);

template <typename Conf>
void double_harris_current_sheet(vector_field<Conf> &B, vector_field<Conf> &J0, particle_data_t &ptc,
                                 rng_states_t &states);

} // namespace Aperture

int main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::info);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  domain_comm<Conf> comm;
  // auto grid = env.register_system<grid_t<Conf>>(env, comm);
  grid_t<Conf> grid(comm);
  // auto pusher = env.register_system<ptc_updater_old_cu<Conf>>(grid, &comm);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_gpu, coord_policy_cartesian>>(grid, &comm);
  // auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  // auto momentum =
      // env.register_system<gather_momentum_space<Conf, exec_policy_gpu>>(grid);
  auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  auto bc = env.register_system<boundary_condition<Conf>>(grid, &comm);
  // auto rad = env.register_system<ph_freepath_dev<Conf>>(*grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta, *J0;
  particle_data_t *ptc;
  // curand_states_t *states;
  rng_states_t *states;
  env.get_data("B0", &B0);
  env.get_data("J0", &J0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("Edelta", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);
  // env.get_data("rand_states", &states);

  harris_current_sheet(*Bdelta, *ptc, *states);

  env.run();
  return 0;
}
