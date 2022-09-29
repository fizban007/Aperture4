/*
 * Copyright (c) 2022 Alex Chen.
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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/ptc_updater_base.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_gca_lite.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_base_impl.hpp"

namespace Aperture {

template class ptc_updater_new<Config<3>, exec_policy_cuda, coord_policy_cartesian_gca_lite>;

}

using namespace Aperture;

int main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;

  domain_comm<Conf> comm;
  grid_t<Conf> grid(comm);

  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian_gca_lite>>(grid, comm);
  // auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  vector_field<Conf> *B, *E;
  particle_data_t *ptc;
  rng_states_t *states;

  env.get_data("B", &B);
  env.get_data("E", &E);
  env.get_data("particles", &ptc);

  // Set initial condition


  env.run();

  return 0;
}
