/*
 * Copyright (c) 2020 Alex Chen.
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

#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_ks.h"
// #include "systems/legacy/ptc_updater_gr_ks.h"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_updater.h"
#include "utils/util_functions.h"

using namespace std;

namespace Aperture {

template <typename Conf>
void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                         const grid_ks_t<Conf> &grid);
}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2, Scalar> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv, false);

  env.params().add("log_level", (int64_t)LogLevel::debug);
  env.params().add("Bp", 2.0);
  env.params().add("bh_spin", 0.98);
  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({3, 3}));
  env.params().add("size", std::vector<double>({2.0, M_PI}));
  env.params().add("lower", std::vector<double>({-0.02, 0.0}));
  env.params().add("fld_output_interval", 100l);
  env.params().add("ptc_output_interval", 100l);
  env.params().add("max_ptc_num", 10000l);
  env.params().add("max_tracked_num", 10000l);
  // Note dt needs to be smaller than shortest cell crossing time
  env.params().add("dt", 1.0e-3);
  env.params().add("max_steps", 1000);

  // domain_comm<Conf> comm(env);
  grid_ks_t<Conf> grid;

  // auto bc = env.register_system<boundary_condition<Conf>>(env, grid);
  // auto solver =
  //     env.register_system<field_solver_gr_ks_cu<Conf>>(env, grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid);
  // auto injector =
  //     env.register_system<bh_injector<Conf>>(env, grid);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid);

  env.init();

  // Prepare initial condition here
  // vector_field<Conf> *B, *D, *B0, *D0;
  // env.get_data("B", &B);
  // env.get_data("E", &D);

  particle_data_t *ptc;
  env.get_data("particles", &ptc);

  value_t r0 = 2.0;
  value_t th0 = 0.5 * M_PI;
  value_t ph0 = 0.0;
  value_t u_r0 = 1.0;
  value_t u_th0 = 0.0;
  value_t u_ph0 = 1.0;
  // Add a single electron with the above initial conditions
  ptc_append_global(exec_policy_dynamic<Conf>::exec_tag{}, *ptc, grid,
                    {grid_ks_t<Conf>::from_radius(r0), th0, ph0},
                    {u_r0, u_th0, u_ph0}, 1.0f, flag_or(PtcFlag::tracked));
  GpuSafeCall(gpuDeviceSynchronize());

  // initial_vacuum_wald(*B, *D, grid);

  // vec_t<value_t, 3> x_global(math::log(4.0), M_PI * 0.5 - 0.2, 0.0);
  // index_t<2> pos;
  // vec_t<value_t, 3> x;
  // grid.from_global(x_global, pos, x);
  // auto ext = grid.extent();
  // typename Conf::idx_t idx(pos, ext);

  // for (int i = 0; i < 1; i++) {
  //   ptc->append(exec_tags::device{}, x, {0.57367008, 0.0, 1.565}, idx.linear,
  //               1000.0, set_ptc_type_flag(0, PtcType::positron));
  //   // ptc->append(exec_tags::device{}, x, {0.0, 0.0, 0.0}, idx.linear, 1000.0,
  //   //                 set_ptc_type_flag(0, PtcType::positron));
  //   // ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, , uint32_t cell)
  // }

  env.run();

  return 0;
}
