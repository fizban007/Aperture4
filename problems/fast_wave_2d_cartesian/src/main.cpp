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

#include "boundary_condition.hpp"

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
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
  // auto bc = env.register_system<boundary_condition<Conf, exec_policy_dynamic>>(
  //     grid, &comm);
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
  float gamma = 5.0;
  int ppc = 20;
  env.params().get_value("Bp", Bp);
  env.params().get_value("ppc", ppc);
  env.params().get_value("qe", qe);
  env.params().get_value("kT", kT);
  env.params().get_value("rho0", rho0);
  env.params().get_value("gamma", gamma);

  // Set dipole initial magnetic field
  B0->set_values(1, [Bp](Scalar x, Scalar y, Scalar z) {
    return Bp;
  });

  float omega = 5.0;
  float theta = M_PI / 4.0;
  float deltaB = 1.0;
  env.params().get_value("omega", omega);
  env.params().get_value("theta", theta);
  env.params().get_value("deltaB", deltaB);

  E->set_values(2, [deltaB, theta, omega](Scalar x, Scalar y, Scalar z) {
    value_t ky = omega * math::cos(theta);
    value_t kx = omega * math::sin(theta);
    return deltaB * math::cos(kx * x + ky * y);
  });

  B->set_values(0, [deltaB, theta, omega](Scalar x, Scalar y, Scalar z) {
    value_t ky = omega * math::cos(theta);
    value_t kx = omega * math::sin(theta);
    return deltaB * math::cos(kx * x + ky * y) * math::cos(theta);
  });

  B->set_values(1, [deltaB, theta, omega](Scalar x, Scalar y, Scalar z) {
    value_t ky = omega * math::cos(theta);
    value_t kx = omega * math::sin(theta);
    return -deltaB * math::cos(kx * x + ky * y) * math::sin(theta);
  });

  comm.send_guard_cells(*E);
  comm.send_guard_cells(*B);

  // Fill the box with pairs
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      [kT, gamma] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        // return rng_maxwell_juttner_3d(state, kT);
        value_t beta = sqrt(1.0 - 1.0 / (gamma * gamma));
        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting<value_t>(state, kT, beta);
        value_t sign = 1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[0] * sign;
        auto p3 = u_d[2] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      [rho0, qe, ppc] LAMBDA(auto &x_global, PtcType type) {
        if (type == PtcType::electron) {
          return rho0 / qe / ppc;
        } else {
          return 0.0f;
        }
      });

  env.run();

  return 0;
}
