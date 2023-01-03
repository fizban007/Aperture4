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

#include "core/math.hpp"
#include "cxxopts.hpp"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/gather_momentum_space.h"
#include "systems/policies/coord_policy_cartesian_impl_cooling.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
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

namespace Aperture {

template class ptc_updater<Config<2>, exec_policy_gpu,
                               coord_policy_cartesian_impl_cooling>;


}

using namespace Aperture;

using vec3 = vec_t<double, 3>;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf> comm;
  auto& grid = *(env.register_system<grid_t<Conf>>(comm));
  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_gpu, coord_policy_cartesian_impl_cooling>>(
      grid, comm);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_gpu, coord_policy_cartesian, IC_radiation_scheme>>(
      grid, &comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_gpu>>(grid);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B;
  particle_data_t *ptc;
  rng_states_t *states;
  env.get_data("B", &B);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);
  float Bp = 10.0f, p0 = 10.0f;
  env.params().get_value("B0", Bp);
  env.params().get_value("p0", p0);

  B->set_values(2, [Bp] (auto x, auto y, auto z) {
      return Bp;
    });

  // int N = 1000;
  // for (int n = 0; n < N; n++) {
  //   ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.5f}, {p0, 0.0f, 0.0f},
  //                   grid.dims[0] / 2 +
  //                       (grid.dims[1] / 2) * grid.dims[0],
  //                   1.0f, set_ptc_type_flag(0, PtcType::electron));
  // }
  auto injector =
      sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);

  int n_upstream = 1;
  float alpha = 2.0;
  injector->inject(
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      [p0, alpha] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                               PtcType type) {
        // Power law distribution of particles
        auto u = rng.uniform<float>();
        double p = 1.0 * pow(1.0 - u, -1.0 / (alpha - 1.0));
        while (p > 1.0e3) {
          u = rng.uniform<float>();
          p = 1.0 * pow(1.0 - u, -1.0 / (alpha - 1.0));
        }

        auto th = rng.uniform<float>() * M_PI;
        auto ph = rng.uniform<float>() * 2.0 * M_PI;

        return vec_t<float, 3>(p * math::sin(th) * math::cos(ph),
                p * math::sin(th) * math::sin(ph),
                p * math::cos(th));
      },
      // [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
      [n_upstream] __device__(auto &x_global) {
        // value_t rho = rho_bg / square(grid_polar_t<Conf>::radius(x_global[0]));
        // value_t rho = rho_bg / grid_polar_t<Conf>::radius(x_global[0]);
        return 1.0f / n_upstream;
      });
  env.run();
  return 0;
}
