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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
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

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  // auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  // auto rad = env.register_system<radiative_transfer<
  //     Conf, exec_policy_dynamic, coord_policy_spherical, IC_radiation_scheme>>(
  //     grid, &comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_spherical_sync_cooling>>(
                                      // coord_policy_spherical>>(
          grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *B;
  particles_t *ptc;
  env.get_data("B", &B);
  env.get_data("particles", &ptc);

  // Read parameters
  float Bp = 1.0e2;
  float kT = 0.1;
  float rho = 1.0;
  int ppc = 100;
  env.params().get_value("Bp", Bp);
  env.params().get_value("rho", rho);
  env.params().get_value("ppc", ppc);
  env.params().get_value("kT", kT);
  // Set dipole initial magnetic field
  B->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // ptc_append_global(exec_tags::device{}, *ptc, grid,
  //                   {grid_sph_t<Conf>::from_radius(2.0), 0.5 * M_PI + 0.02, 0.0},
  //                   {10.0, 10.0, 0.0}, 1.0f, flag_or(PtcFlag::tracked));
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto x_global = grid.coord_global(pos);
        if (grid_sph_t<Conf>::radius(x_global[0]) > 2.0 &&
            grid_sph_t<Conf>::radius(x_global[0]) < 3.0 &&
            grid_sph_t<Conf>::theta(x_global[1]) > M_PI * 0.5 - 0.5 &&
            grid_sph_t<Conf>::theta(x_global[1]) < M_PI * 0.5 + 0.5) {
          return true;
        } else {
          return false;
        }
      },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      [kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto p = rng_maxwell_juttner_3d(state, kT);
        grid_sph_t<Conf>::vec_from_cart(p, x_global);
        return p;
        // return rng_maxwell_juttner_3d(state, kT);
      },
      [rho, ppc] LAMBDA(auto &x_global, PtcType type) {
        return rho / ppc *
            math::sin(grid_sph_t<Conf>::theta(x_global[1]));
      });

  env.run();

  return 0;
}
