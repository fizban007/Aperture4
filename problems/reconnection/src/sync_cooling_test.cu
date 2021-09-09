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

#include "framework/config.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_cuda.hpp"
// #include "systems/policies/sync_cooling_policy.hpp"
#include "systems/policies/phys_policy_sync_cooling.hpp"
#include "systems/ptc_updater_base_impl.hpp"

namespace Aperture {

template class ptc_updater_new<Config<2>, exec_policy_cuda,
                               coord_policy_cartesian, phys_policy_sync_cooling>;

}

using namespace Aperture;

int
main(int argc, char* argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto& env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf> comm;

  auto grid = env.register_system<grid_t<Conf>>(comm);
  auto pusher = env.register_system<ptc_updater_new<
      Conf, exec_policy_cuda, coord_policy_cartesian, phys_policy_sync_cooling>>(
      *grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(*grid, &comm);

  env.init();

  vector_field<Conf> *B;
  env.get_data("B", &B);
  particle_data_t* ptc;
  env.get_data("particles", &ptc);

  auto Bp = sim_env().params().get_as("Bp", 1000.0);

  B->set_values(0, [Bp] (auto x, auto y, auto z) {
      return Bp;
    });

  ptc->append_dev({0.5f, 0.5f, 0.5f}, {10.0f, 10.0f, 0.0f},
                  grid->dims[0] / 2 + (grid->dims[1] / 2) * grid->dims[0], 1.0f,
                  set_ptc_type_flag(0, PtcType::electron));

  env.run();

  return 0;
}
