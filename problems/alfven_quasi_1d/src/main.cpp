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
#include "systems/ph_freepath_dev.h"
#include "systems/ptc_updater.h"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {
template <typename Conf>
void set_initial_condition(vector_field<Conf> &B0, particle_data_t &ptc,
                           curand_states_t &states, int mult, Scalar weight);

template <typename Conf>
void initial_condition_wave(vector_field<Conf> &B, vector_field<Conf> &E,
                            vector_field<Conf> &B0, particle_data_t &ptc,
                            curand_states_t &states, int mult, Scalar weight);

template <typename Conf>
void initial_condition_standing_alfven(
    vector_field<Conf> &B, vector_field<Conf> &E, vector_field<Conf> &B0,
    particle_data_t &ptc, curand_states_t &states, int mult, Scalar weight);

}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto& env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  domain_comm<Conf> comm;
  // auto grid = env.register_system<grid_t<Conf>>(env, comm);
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<ptc_updater_cu<Conf>>(grid, &comm);
  auto lorentz =
      env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space_cu<Conf>>(grid);
  auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  auto bc = env.register_system<boundary_condition<Conf>>(grid);
  // auto rad = env.register_system<ph_freepath_dev<Conf>>(*grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta;
  particle_data_t *ptc;
  curand_states_t *states;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("Edelta", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rand_states", &states);

  // set_initial_condition(*B0, *ptc, *states, 10, 1.0);
  initial_condition_wave(*Bdelta, *Edelta, *B0, *ptc, *states, 10, 1.0);

  env.run();
  return 0;
}
