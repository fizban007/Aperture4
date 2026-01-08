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
// #include "cxxopts.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include <fstream>
#include <memory>
#include <vector>

using namespace std;
using namespace Aperture;

namespace Aperture {

template class ptc_updater<Config<2>, exec_policy_dynamic,
                           coord_policy_cartesian_sync_cooling>;

}  // namespace Aperture

int
main(int argc, char* argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);


  domain_comm<Conf, exec_policy_dynamic> comm;
  auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_dynamic, coord_policy_cartesian, IC_radiation_scheme>>(
      grid, &comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_cartesian_sync_cooling>>(
          grid, &comm);

  env.init();

  vector_field<Conf> *B;
  env.get_data("B", &B);

  int ppc = 100;
  float kT = 0.1;
  float B0 = 1.0e2;
  float sigma = 10.0;
  float pitch = M_PI * 0.5;
  env.params().get_value("ppc", ppc);
  env.params().get_value("B0", B0);
  env.params().get_value("kT", kT);
  env.params().get_value("sigma", sigma);
  env.params().get_value("pitch", pitch);
  float rho = B0*B0 / sigma;

  B->set_values(2, [B0](auto x, auto y, auto z) { return B0; });

  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return true;
      },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * ppc;
      },
      [kT, pitch] LAMBDA(auto &x_global,
                rand_state &state,
                PtcType type) {
        return rng_maxwell_juttner_3d(state, kT);
        // auto th = rng_uniform(state) * M_PI;
        // auto cth = math::cos(th);
        // auto sth = math::sin(th);
        // auto cth = rng_uniform(state);
        // auto sth = math::sqrt(1.0f - cth*cth);
        // auto ph = rng_uniform(state) * 2.0f * M_PI;
        // return vec_t<typename Conf::value_t, 3>(kT * sth * math::cos(ph),
        //                                         kT * sth * math::sin(ph),
        //                                         kT * cth);
        // return vec_t<typename Conf::value_t, 3>(0.0, 0.0, kT);
      },
      [rho, ppc] LAMBDA(auto &x_global, PtcType type) {
        return rho / ppc;
      });

  env.run();

  return 0;
}
