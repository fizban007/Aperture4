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
#include "systems/field_solver_default.h"
#include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env, *comm);
  auto solver =
      env.register_system<field_solver_default<Conf>>(env, *grid, comm);
  auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);

  env.init();
  env.run();
  return 0;
}
