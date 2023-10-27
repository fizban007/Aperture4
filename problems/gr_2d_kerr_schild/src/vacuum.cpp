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
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks.h"
#include "systems/grid_ks.h"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"

using namespace std;

namespace Aperture {
template <typename Conf>
void initial_nonrotating_vacuum_wald(vector_field<Conf> &B0,
                                     vector_field<Conf> &D0,
                                     const grid_ks_t<Conf> &grid);

template <typename Conf>
void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                         const grid_ks_t<Conf> &grid);
}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  using exec_policy = exec_policy_dynamic<Conf>;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_ks_t<Conf> grid(comm);

  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid,
                                                                       &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  // Prepare initial condition here
  vector_field<Conf> *B0, *D0, *B, *D;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);

  initial_vacuum_wald(*B, *D, grid);
  // initial_nonrotating_vacuum_wald(*B, *D, grid);
  // B->add_by(*B0, -1.0);
  // D->add_by(*D0, -1.0);

  env.run();

  return 0;
}
