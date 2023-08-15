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

#include "core/math.hpp"
#include "cxxopts.hpp"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/policies/coord_policy_cartesian_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/grid.h"
// #include "systems/policies/exec_policy_host.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/range.hpp"
#include "utils/util_functions.h"
#include <fstream>
#include <memory>
#include <vector>

namespace Aperture {

template class ptc_updater<Config<3>, exec_policy_dynamic,
                               coord_policy_cartesian_sync_cooling>;


}

using namespace Aperture;

using vec3 = vec_t<double, 3>;

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  // auto& grid = *(env.register_system<grid_t<Conf>>(comm));
  grid_t<Conf> grid(comm);
  auto pusher = env.register_system<ptc_updater<
      // Conf, exec_policy_dynamic, coord_policy_cartesian_sync_cooling>>(
        Conf, exec_policy_dynamic, coord_policy_cartesian>>(
      grid, &comm);
  // auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  vector_field<Conf> *B;
  env.get_data("B", &B);
  float Bp = 10.0f, p0 = 10.0f;
  int ppc = 100;
  env.params().get_value("B0", Bp);
  env.params().get_value("p0", p0);
  env.params().get_value("ppc", ppc);

  B->set_values(2, [Bp] (auto x, auto y, auto z) {
      return Bp;
    });

  // auto injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_dynamic>>(grid);
  ptc_injector_dynamic<Conf> injector(grid);

  injector.inject_pairs(
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      // Number injected
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * ppc;
      },
      // Initialize particles
      [] LAMBDA(auto &x_global, rand_state &state,
                                       PtcType type) {
        auto th = rng_uniform<float>(state) * M_PI;
        auto ph = rng_uniform<float>(state) * 2.0 * M_PI;
        double p = 100.0;
        return vec_t<float, 3>(p * math::sin(th) * math::cos(ph),
                               p * math::sin(th) * math::sin(ph),
                               p * math::cos(th));
      },
      // Particle weight
      [ppc] LAMBDA(auto &x_global, PtcType type) {
        return 1.0 / ppc;
      });
  env.run();

  return 0;
}
