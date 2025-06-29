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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_sph.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"

#include "boundary_condition.hpp"

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Config<2>::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_spherical_sync_cooling>>(
          // coord_policy_spherical>>(
          grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_spherical>>(grid,
                                                                       &comm);
  auto bc = env.register_system<boundary_condition<Conf, exec_policy_dynamic>>(
      grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta, *Btotal, *Etotal;
  env.get_data("B0", &B0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("Edelta", &Edelta);
  env.get_data("B", &Btotal);
  env.get_data("E", &Etotal);

  Etotal->set_fullres_output(2, true);
  Etotal->set_fullres_output(1, true);
  Btotal->set_fullres_output(2, true);
  Btotal->set_fullres_output(1, true);

  // Read parameters
  float Bp = 1.0e4;
  float qe = 1.0;
  float kT = 1.0e-3;
  float rho0 = 1.0e4;
  int ppc = 20;
  env.params().get_value("Bp", Bp);
  env.params().get_value("ppc", ppc);
  env.params().get_value("qe", qe);
  env.params().get_value("kT", kT);
  env.params().get_value("rho0", rho0);

  // Set dipole initial magnetic field
  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B0->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // float w0 = 0.1;
  // float freq = 5.0;
  // int num_lambda = 4;
  // sim_env().params().get_value("w0", w0);
  // sim_env().params().get_value("wave_freq", freq);
  // sim_env().params().get_value("num_lambda", num_lambda);

  // float lower[2] = {0.0, 0.0};
  // sim_env().params().get_array("lower", lower);

  // Bdelta->set_values(1, [Bp, w0, freq, num_lambda, lower](Scalar x, Scalar th, Scalar phi) {
  //   auto r = grid_sph_t<Conf>::radius(x);
  //   auto r_lower = grid_sph_t<Conf>::radius(lower[0]);
  //   Scalar phase = 2.0 * M_PI * freq * (r - r_lower);
  //   if (phase < 2.0 * M_PI * num_lambda) {
  //     return Bp * w0 * math::sin(phase) * math::sin(th) / r;
  //   } else {
  //     return (value_t)0.0;
  //   }
  // });

  // Edelta->set_values(2, [Bp, w0, freq, num_lambda, lower](Scalar x, Scalar th, Scalar phi) {
  //   auto r = grid_sph_t<Conf>::radius(x);
  //   auto r_lower = grid_sph_t<Conf>::radius(lower[0]);
  //   Scalar phase = 2.0 * M_PI * freq * (r - r_lower);
  //   if (phase < 2.0 * M_PI * num_lambda) {
  //     return -Bp * w0 * math::sin(phase) * math::sin(th) / r;
  //   } else {
  //     return (value_t)0.0;
  //   }
  // });

  // Fill the magnetosphere with pairs
  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2 * ppc; },
      [kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        return rng_maxwell_juttner_3d<value_t>(state, kT);
      },
      [rho0, qe, ppc] LAMBDA(auto &x_global, PtcType type) {
        auto &grid = static_cast<const grid_sph_t<Conf> &>(
            exec_policy_dynamic<Conf>::grid());
        auto r = grid.radius(x_global[0]);
        auto th = grid.theta(x_global[1]);
        return rho0 * math::sin(th) / qe / ppc;
      });

  env.run();

  return 0;
}
