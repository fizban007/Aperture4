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
  auto& env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf> comm;
  grid_ks_t<Conf> grid(comm);

  auto solver =
      env.register_system<field_solver_gr_ks_cu<Conf>>(grid, &comm);
  // auto bc = env.register_system<boundary_condition<Conf>>(grid);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  // Prepare initial condition here
  vector_field<Conf> *B0, *D0, *B, *D;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);

  initial_vacuum_wald(*B0, *D0, grid);
  B->copy_from(*B0);
  D->copy_from(*D0);

  env.run();

  return 0;
}
