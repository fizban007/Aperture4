/*
 * Copyright (c) 2021 Alex Chen.
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

#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/gather_momentum_space.h"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/ptc_updater_base.h"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template <typename Conf>
void harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                          rng_states_t<exec_tags::device> &states);

template <typename Conf>
void double_harris_current_sheet(vector_field<Conf> &B, particle_data_t &ptc,
                                 rng_states_t<exec_tags::device> &states);

}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_gpu> comm;
  auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_gpu, coord_policy_cartesian>>(grid, &comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_gpu>>(grid);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_gpu, coord_policy_cartesian>>(grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_gpu>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta;
  particle_data_t *ptc;
  // curand_states_t *states;
  rng_states_t<exec_tags::device> *states;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);
  // env.get_data("rand_states", &states);

  double_harris_current_sheet(*Bdelta, *ptc, *states);

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  Logger::print_info("GPU memory: free = {} GiB, total = {} GiB",
                     free_mem / 1.0e9, total_mem / 1.0e9);
  env.run();
  return 0;
}
