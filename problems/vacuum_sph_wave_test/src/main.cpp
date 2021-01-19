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
#include "systems/grid_sph.h"
// #include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_sph_t<Conf>>(env, *comm);
  auto solver =
      env.register_system<field_solver_sph_cu<Conf>>(env, *grid, comm.get());
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm.get());
  auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);

  env.init();

  // Set initial condition
  vector_field<Conf> *B, *B0;
  env.get_data("B", &B);
  env.get_data("B0", &B0);
  if (B == nullptr)
    Logger::print_err("B is nullptr!!!");

  double Bp = 100.0;

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    return Bp / (r * r);
    // return Bp * 2.0 * cos(theta) / (r * r * r);
  });
  // B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
  //   Scalar r = std::exp(x);
  //   // return Bp / (r * r);
  //   return Bp * sin(theta) / (r * r * r);
  // });
  B->copy_from(*B0);

  env.run();
  return 0;
}
