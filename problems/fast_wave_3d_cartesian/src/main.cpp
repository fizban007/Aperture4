/*
 * Copyright (c) 2024 Alex Chen.
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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_cartesian.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid.h"
#include "systems/policies/coord_policy_cartesian.hpp"
// #include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  using value_t = typename Conf::value_t;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_cartesian>>(
          grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  // auto moments =
  //     env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *B0, *B, *E;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &E);

  // Read parameters
  float Bp = 1.0e4;
  float qe = 1.0;
  float kT = 1.0e-3;
  float rho0 = 1.0;
  float gamma = 1.0;
  int ppc = 20;
  env.params().get_value("Bp", Bp);
  env.params().get_value("ppc", ppc);
  env.params().get_value("qe", qe);
  env.params().get_value("kT", kT);
  env.params().get_value("rho0", rho0);
  env.params().get_value("gamma", gamma);

  // Set background magnetic field in the z direction
  B0->set_values(2, [Bp](Scalar x, Scalar y, Scalar z) {
    return Bp;
  });

  int kx_f = 4;
  int kz_f = 3;
  float deltaB = 1.0;
  float seed_amplitude = 1.0e-6;
  value_t sizes[Conf::dim];
  uint32_t N[Conf::dim];
  env.params().get_value("kx_f", kx_f);
  env.params().get_value("kz_f", kz_f);
  env.params().get_value("deltaB", deltaB);
  env.params().get_value("seed_amplitude", seed_amplitude);
  env.params().get_array("size", sizes);
  env.params().get_array("N", N);

  Logger::print_info("deltaB is {}", deltaB);

  // Initialize a fast wave propagating in the x-z plane
  E->set_values(1, [deltaB, kx_f, kz_f, sizes](Scalar x, Scalar y, Scalar z) {
    value_t kz = kz_f * 2.0 * M_PI / sizes[2];
    value_t kx = kx_f * 2.0 * M_PI / sizes[0];
    return deltaB * math::cos(kx * x + kz * z);
  });

  B->set_values(0, [deltaB, kx_f, kz_f, sizes](Scalar x, Scalar y, Scalar z) {
    value_t kz = kz_f * 2.0 * M_PI / sizes[2];
    value_t kx = kx_f * 2.0 * M_PI / sizes[0];
    value_t theta = math::atan2(kx, kz);
    return deltaB * math::cos(kx * x + kz * z) * math::cos(theta);
  });

  B->set_values(2, [deltaB, kx_f, kz_f, sizes](Scalar x, Scalar y, Scalar z) {
    value_t kz = kz_f * 2.0 * M_PI / sizes[2];
    value_t kx = kx_f * 2.0 * M_PI / sizes[0];
    value_t theta = math::atan2(kx, kz);
    return -deltaB * math::cos(kx * x + kz * z) * math::sin(theta);
  });

  // initialize seed alfven waves
  B->set_values(1, [seed_amplitude, sizes](Scalar x, Scalar y, Scalar z) {
    value_t k1x = 2 * 2.0 * M_PI / sizes[0];    
    value_t k1z = 4 * 2.0 * M_PI / sizes[2];    
    value_t k2x = 2 * 2.0 * M_PI / sizes[0];    
    value_t k2z = -1 * 2.0 * M_PI / sizes[2];
    return seed_amplitude * (math::cos(k1x * x + k1z * z) + math::cos(k2x * x + k2z * z));
  });

  E->set_values(0, [seed_amplitude, sizes](Scalar x, Scalar y, Scalar z) {
    value_t k1x = 2 * 2.0 * M_PI / sizes[0];    
    value_t k1z = 4 * 2.0 * M_PI / sizes[2];    
    value_t k2x = 2 * 2.0 * M_PI / sizes[0];    
    value_t k2z = -1 * 2.0 * M_PI / sizes[2];
    return seed_amplitude * (math::cos(k1x * x + k1z * z) + math::cos(k2x * x + k2z * z));
  });

  // Fill the box with pairs
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      [kT, gamma] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return rng_maxwell_juttner_3d<value_t>(state, kT);
      },
      [rho0, qe, ppc] LAMBDA(auto &x_global, PtcType type) {
        return rho0 / qe / ppc;
      });

  env.run();

  return 0;
}
