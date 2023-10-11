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

#include "core/random.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/gather_tracked_ptc.h"
// #include "systems/gather_momentum_space.h"
// #include "systems/legacy/ptc_updater_old.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiation/IC_radiation_scheme.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template <typename Conf>
class ptc_physics_policy_gravity_y_sym {
 public:
  using value_t = typename Conf::value_t;

  void init() { sim_env().params().get_value("gravity", m_g); }

  template <typename PtcContext, typename IntT>
  HD_INLINE void operator()(const Grid<Conf::dim, value_t>& grid,
                            PtcContext& context,
                            const vec_t<IntT, Conf::dim>& pos,
                            value_t dt) const {
    auto y = grid.coord(1, pos[1], context.x[1]);
    context.p[1] -= sgn(y) * m_g * dt;
    context.gamma = sqrt(1.0f + context.p.dot(context.p));
  }

 private:
  value_t m_g = 1.0f;
};

template class ptc_updater<Config<2>, exec_policy_dynamic,
                           coord_policy_cartesian, ptc_physics_policy_gravity_y_sym>;

template <typename Conf>
void
Parker_initial_condition(vector_field<Conf> &B, particle_data_t &ptc,
                         rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;

  // In this weird unit system, L = 1 = \sigma/g. Gravity is the same as
  // magnetization
  value_t B0 = sim_env().params().get_as<double>("B0", 100.0);
  value_t kT = sim_env().params().get_as<double>("kT", 1.0e-3);
  value_t g = sim_env().params().get_as<double>("gravity", 1.0);

  value_t global_sizes[Conf::dim];
  value_t global_lower[Conf::dim];

  sim_env().params().get_array("size", global_sizes);
  sim_env().params().get_array("lower", global_lower);

  int ppc = sim_env().params().get_as<int64_t>("ppc", 100);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);

  auto &grid = B.grid();
  auto ext = grid.extent();
  value_t ysize = global_sizes[1];

  // y is already in units of L
  B.set_values(0, [B0](auto x, auto y, auto z) {
    return B0 * math::exp(-math::abs(y) / 2.0);
  });

  ptc_injector_dynamic<Conf> injector(grid);

  // Background (upstream) particles
  injector.inject_pairs(
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 2 * ppc;
      },
      [kT, g, B0] LAMBDA(auto &x_global,
                         rand_state &state, PtcType type) {
        value_t beta = g / B0 / 2.0 * math::exp(math::abs(x_global[1])/2.0);
        beta *= (type == PtcType::electron ? -1.0 : 1.0);
        // return rng_maxwell_juttner_3d(state, kT);
        return rng_maxwell_juttner_drifting(state, kT, beta);
        // return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      [ppc, q_e, g, B0] LAMBDA(auto &x_global, PtcType type) {
        // electrons and positions each account for half of mass density
        value_t rho = B0*B0 / g / 2.0 * math::exp(-math::abs(x_global[1]));
        return rho / q_e / ppc;
      });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}



}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  // sim_environment env(&argc, &argv);
  auto &env = sim_environment::instance(&argc, &argv);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  domain_comm<Conf, exec_policy_dynamic> comm;
  auto &grid = *(env.register_system<grid_t<Conf>>(comm));
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_cartesian,
                                      ptc_physics_policy_gravity_y_sym>>(
          grid, &comm);
  auto moments =
      env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  // auto momentum =
  //     env.register_system<gather_momentum_space<Conf,
  //     exec_policy_gpu>>(grid);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                      &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  vector_field<Conf> *Bdelta;
  particle_data_t *ptc;
  rng_states_t<exec_tags::device> *states;
  env.get_data("Bdelta", &Bdelta);
  //env.get_data("Edelta", &Edelta);
  env.get_data("particles", &ptc);
  env.get_data("rng_states", &states);

  // Set initial conditions
  Parker_initial_condition(*Bdelta, *ptc, *states);

#ifdef GPU_ENABLED
  size_t free_mem, total_mem;
  gpuMemGetInfo(&free_mem, &total_mem);
  Logger::print_info("GPU memory: free = {} GiB, total = {} GiB",
                     free_mem / 1.0e9, total_mem / 1.0e9);
#endif
  env.run();
  return 0;
}
