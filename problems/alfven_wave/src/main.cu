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
// #include "systems/ptc_updater_sph.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/initial_condition.h"
#include "systems/ptc_injector_mult.h"
#include "systems/ptc_updater_impl.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/ptc_physics_policy_gravity_sph.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv, false);

  // env.params().add("log_level", (int64_t)LogLevel::detail);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_sph_t<Conf>>();
  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_gpu, coord_policy_spherical, ptc_physics_policy_gravity_sph>>(*grid);
  auto solver =
      env.register_system<field_solver_sph_cu<Conf>>(*grid);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(*grid);
  // auto injector =
  //     env.register_system<ptc_injector_mult<Conf>>(*grid);
  auto bc = env.register_system<boundary_condition<Conf>>(*grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_gpu>>(*grid);

  env.init();

  set_initial_condition(*grid);

  env.run();
  return 0;
}
