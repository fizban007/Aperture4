/*
 * Copyright (c) 2023 Alex Chen.
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

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<1> Conf;
  auto &env = sim_environment::instance(&argc, &argv, true);
  typedef typename Conf::value_t value_t;
  using exec_policy = exec_policy_dynamic<Conf>;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                      &comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  // Prepare initial conditions
  value_t rho_b = 1.0, rho_0 = 1.0, q_e = 1.0, p0 = 1.0;
  value_t T_b = 0.1;
  int mult = 10;
  env.params().get_value("rho_b", rho_b);
  env.params().get_value("rho_0", rho_0);
  env.params().get_value("q_e", q_e);
  env.params().get_value("p0", p0);
  env.params().get_value("multiplicity", mult);
  env.params().get_value("T_b", T_b);

  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [mult] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * mult; },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [p0, T_b] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto beta = p0 / math::sqrt(p0*p0 + 1.0);
        return rng_maxwell_juttner_drifting(state, T_b, beta);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [rho_b, mult, q_e] LAMBDA(auto &x_global, PtcType type) {
        if (type == PtcType::electron)
          return rho_b / mult / q_e;
        else
          return value_t(0.0);
      });
  injector.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [mult] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * mult; },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [p0, T_b] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto beta = p0 / math::sqrt(p0*p0 + 1.0);
        return rng_maxwell_juttner_drifting(state, T_b, -beta);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [rho_b, mult, q_e] LAMBDA(auto &x_global, PtcType type) {
        if (type == PtcType::electron)
          return rho_b / mult / q_e;
        else
          return value_t(0.0);
      });

  env.run();
  return 0;
}
