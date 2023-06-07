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
#include "cxxopts.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.hpp"
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
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include <fstream>
#include <memory>
#include <vector>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  // auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_spherical_sync_cooling>>(
          // coord_policy_spherical>>(
          grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  // auto rad = env.register_system<radiative_transfer<
  //     Conf, exec_policy_dynamic, coord_policy_spherical,
  //     IC_radiation_scheme>>( grid, &comm);
  auto solver =
      // env.register_system<field_solver<Conf, exec_policy_dynamic,
      // coord_policy_spherical>>(
      env.register_system<field_solver<Conf, exec_policy_dynamic,
                                       coord_policy_spherical>>(
          grid, &comm);
  auto bc = env.register_system<boundary_condition<Conf, exec_policy_dynamic>>(
      grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *B0;
  particles_t *ptc;
  env.get_data("B0", &B0);
  env.get_data("particles", &ptc);

  // Read parameters
  float Bp = 1.0e2;
  float twist_omega = 0.01;
  float qe = 1.0;
  int ppc = 100;
  env.params().get_value("Bp", Bp);
  env.params().get_value("ppc", ppc);
  env.params().get_value("twist_omega", twist_omega);
  env.params().get_value("qe", qe);

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

  // Fill the magnetosphere with some multiplicity
  pusher->fill_multiplicity(ppc, Bp * twist_omega / qe / ppc);

  env.run();
  return 0;
}
