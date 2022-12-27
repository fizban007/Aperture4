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

#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver.h"
#include "systems/gather_momentum_space.h"
#include "systems/ptc_injector_cuda.hpp"
// #include "systems/legacy/ptc_updater_old.h"
#include "systems/policies/coord_policy_cartesian_impl_cooling.hpp"
#include "systems/ptc_updater_base.h"
#include <iostream>

using namespace std;
using namespace Aperture;

template <typename Conf>
void init_upstream(vector_field<Conf> &E,
                   vector_field<Conf> &B, particle_data_t &ptc,
                   rng_states_t &states) {
  using value_t = typename Conf::value_t;
  // auto delta = sim_env().params().get_as<double>("current_sheet_delta", 5.0);
  value_t B_g = sim_env().params().get_as<double>("guide_field", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 1.0e3);
  value_t kT_cs = sim_env().params().get_as<double>("current_sheet_kT", 1.0);
  value_t kT_upstream = sim_env().params().get_as<double>("upstream_kT", 0.01);
  value_t beta_d =
      sim_env().params().get_as<double>("current_sheet_drift", 0.5);
  value_t gamma_d = 1.0f / math::sqrt(1.0f - beta_d * beta_d);

  value_t delta = 2.0f * kT_cs / (math::sqrt(sigma) * gamma_d * beta_d);
  value_t n_d = gamma_d * sigma / (4.0f * kT_cs);

  int n_cs = sim_env().params().get_as<int64_t>("current_sheet_n", 15);
  int n_upstream = sim_env().params().get_as<int64_t>("upstream_n", 5);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  // Our unit for length will be upstream c/\omega_p, therefore sigma determines
  // the upstream field strength
  value_t B0 = math::sqrt(sigma);
  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = grid.sizes[1];

  // Initialize the magnetic field values
  B.set_values(0, [B0, delta, ysize](auto x, auto y, auto z) {
    return B0;
  });
  E.set_values(2, [B0, delta, ysize](auto x, auto y, auto z) {
    return 0.1 * B0;
  });

  auto injector = sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  injector->init();

  injector->inject(
      [] __device__(auto &pos, auto &grid, auto &ext) { return true; },
      [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
        return 2 * n_upstream;
      },
      [kT_upstream] __device__(auto &pos, auto &grid, auto &ext, rng_t<exec_tags::device> &rng,
                               PtcType type) {
        auto p1 = rng.gaussian<value_t>(2.0f * kT_upstream);
        auto p2 = rng.gaussian<value_t>(2.0f * kT_upstream);
        auto p3 = rng.gaussian<value_t>(2.0f * kT_upstream);
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [n_upstream] __device__(auto &pos, auto &grid, auto &ext) {
      [n_upstream] __device__(auto& x_global) {
        return 1.0 / n_upstream;
      });

  Logger::print_info("After initial condition, there are {} particles", ptc.number());
}

int main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::info);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  domain_comm<Conf> comm;
  // auto grid = env.register_system<grid_t<Conf>>(env, comm);
  grid_t<Conf> grid(comm);
  // auto pusher = env.register_system<ptc_updater_old_cu<Conf>>(grid, &comm);
  auto pusher = env.register_system<ptc_updater<
      Conf, exec_policy_cuda, coord_policy_cartesian_impl_cooling>>(grid, comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto momentum =
      env.register_system<gather_momentum_space<Conf, exec_policy_cuda>>(grid);
  auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  // auto rad = env.register_system<ph_freepath_dev<Conf>>(*grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *E0, *B0, *Bdelta, *Edelta;
  particle_data_t *ptc;
  // curand_states_t *states;
  rng_states_t *states;
  env.get_data("B0", &B0);
  env.get_data("E0", &E0);
  env.get_data("Bdelta", &Bdelta);
  env.get_data("Edelta", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);
  // env.get_data("rand_states", &states);

  init_upstream(*E0, *B0, *ptc, *states);

  env.run();
  return 0;
}
