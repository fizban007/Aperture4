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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_updater.h"

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;
  using exec_policy = exec_policy_dynamic<Conf>;
  using exec_tag = typename exec_policy::exec_tag;

  // Specify config parameters
  env.params().add("log_level", (int64_t)LogLevel::debug);
  env.params().add("N", std::vector<int64_t>({64, 64}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("periodic_boundary", std::vector<bool>({true, true}));
  env.params().add("size", std::vector<double>({1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));
  env.params().add("max_tracked_num", 1000);
  env.params().add("max_ptc_num", 1000);
  env.params().add("dt", 1.0e-2);
  env.params().add("max_steps", 1000);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                      &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  // Prepare initial conditions
  particle_data_t *ptc;
  vector_field<Conf> *B;
  sim_env().get_data("particles", &ptc);
  sim_env().get_data("B", &B);

  double B0 = 1000.0;
  B->set_values(2, [B0] (auto x, auto y, auto z) {
    return B0;
  });

  for (int i = 0; i < 10; i++) {
    ptc_append_global(
        exec_tag{}, *ptc, grid, {0.4, 0.5, 0.0}, {0.0, 10.0, 0.0}, 1.0,
        set_ptc_type_flag(flag_or(PtcFlag::tracked), PtcType::positron));
  }

  env.run();
  return 0;
}
