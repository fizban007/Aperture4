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
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_sph.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/resonant_scattering_scheme.hpp"
// #include "systems/boundary_condition.h"
#include <iostream>

namespace Aperture {
  template class radiative_transfer<Config<2>, exec_policy_dynamic,
                                  coord_policy_spherical, resonant_scattering_scheme>;
}

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Config<2>::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_spherical>>(
          grid, &comm);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_dynamic, coord_policy_spherical,
      resonant_scattering_scheme>>( grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  double Bp = 10000.0;
  env.params().get_value("Bp", Bp);

  // Set initial condition
  // set_initial_condition(env, *grid, 0, 1.0, Bp);
  vector_field<Conf> *B0, *B;
  particle_data_t *ptc;
  // env.get_data("B0", &B0);
  env.get_data("B", &B);
  env.get_data("particles", &ptc);

  // Set dipole initial magnetic field
  B->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // Add a single particle to the magnetosphere
  Scalar p0 = 1000.0f;
  Scalar r0 = 1.1f;
  Scalar th0 = 0.3f;
  for (int i = 0; i < 5000; i++) {
    // ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, {p0, 0.0f, 0.0f}, 10 + 60 * grid->dims[0],
    //                 100.0);
    ptc_append_global(exec_policy_dynamic<Conf>::exec_tag{}, *ptc, grid,
                      {grid_sph_t<Conf>::from_radius(r0), th0, 0.0f},
                      {p0, 0.0f, 0.0f}, 1.0f, flag_or(PtcFlag::tracked));
  }

  env.run();
  return 0;
}
