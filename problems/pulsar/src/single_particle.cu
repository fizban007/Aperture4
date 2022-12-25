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

#include "core/math.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_sph.h"
#include "systems/ph_freepath_dev.h"
#include "systems/ptc_updater_pulsar.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);
  double Omega = 0.16667;
  env.params().get_value("omega", Omega);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  // auto grid = env.register_system<grid_sph_t<Conf>>(env);
  grid_sph_t<Conf> grid(env);
  grid.init();
  auto pusher = env.register_system<ptc_updater_pulsar<Conf>>(env, grid);
  auto lorentz =
      env.register_system<compute_lorentz_factor_cu<Conf>>(env, grid);
  // auto rad = env.register_system<ph_freepath_dev<Conf>>(env, *grid);
  // auto solver = env.register_system<field_solver_sph_cu<Conf>>(env, grid);

  // auto bc = env.register_system<boundary_condition<Conf>>(env, grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, grid);

  env.init();

  double Bp = 10000.0;
  env.params().get_value("Bp", Bp);

  vector_field<Conf> *B0, *E0;
  particle_data_t *ptc;
  env.get_data("B", &B0);
  env.get_data("E", &E0);
  env.get_data("particles", &ptc);

  // Set dipole initial magnetic field
  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });
  E0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return -Bp * sin(theta) / (cube(r) * r);
  });
  E0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * 2.0 * cos(theta) / (cube(r) * r);
  });
  // B->copy_from(*B0);

  // Add a single particle to the magnetosphere
  Scalar p0 = -100.0f;
  for (int i = 0; i < 1; i++) {
    ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, {p0, 0.0f, 0.0f}, 800 + 500 * grid.dims[0],
                    100.0);
  }

  env.run();
  return 0;
}
