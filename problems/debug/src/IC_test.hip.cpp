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

#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_base.h"
#include "systems/gather_momentum_space.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/phys_policy_IC_cooling.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_base_impl.hpp"
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "utils/hdf_wrapper.h"
#include <iostream>

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_gpu> comm;
  grid_t<Conf> grid(comm);
  size_t max_ptc_num = 10000;
  env.register_data<particle_data_t>("particles", max_ptc_num, MemType::device_managed);

  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_gpu, coord_policy_cartesian, ptc_physics_policy_empty>>(
      grid, &comm);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_gpu, coord_policy_cartesian, IC_radiation_scheme>>(
      grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_gpu>>(grid, &comm);

  env.init();

  particle_data_t *ptc;
  env.get_data("particles", &ptc);

  Logger::print_info("initializing a single particle");
  ptc_append(exec_tags::host{}, *ptc, {0.0f, 0.0f, 0.0f}, {1000.0f, 0.0f, 0.0f}, 4 + 4 * 36, 1.0f,
              set_ptc_type_flag(0, PtcType::electron));
  gpuDeviceSynchronize();
  Logger::print_info("finished initializing a single particle");

  std::vector<double> gammas(env.get_max_steps() / 10);
  for (int n = 0; n < env.get_max_steps(); n++) {
    Logger::print_info("gamma is {}, cell is {}", ptc->E[0], ptc->cell[0]);
    if (n % 10 == 0) {
      gammas[n / 10] = ptc->E[0];
    }
    env.update();
  }
  Logger::print_info("gamma[0] is {}", gammas[0]);

  H5File outfile = hdf_create("ptc_output.h5");
  outfile.write(gammas.data(), gammas.size(), "gamma");
  outfile.close();

  return 0;
}
