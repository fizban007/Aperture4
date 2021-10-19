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
#include "systems/field_solver_sph.h"
// #include "systems/legacy/ptc_updater_sph.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/gather_momentum_space.h"
#include "systems/ptc_updater_base.h"
#include "initial_condition.h"
// #include "systems/ptc_injector.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto& env = sim_environment::instance(&argc, &argv, false);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // domain_comm<Conf> comm;
  // grid_sph_t<Conf> grid(comm);
  grid_sph_t<Conf> grid;
  grid.init();

  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_spherical>>(grid);
  auto solver =
      env.register_system<field_solver_sph_cu<Conf>>(grid);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_cuda>>(grid);
  auto bc = env.register_system<boundary_condition<Conf>>(grid);
  auto exporter = env.register_system<data_exporter<Conf>>(grid);

  env.init();

  initial_condition_plasma(grid);

  env.run();
  return 0;
}
