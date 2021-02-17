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
#include "cuda_runtime_api.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks.h"
#include "systems/grid_ks.h"
// #include "systems/legacy/ptc_updater_gr_ks.h"
#include "systems/ptc_updater_base.h"
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

  auto& env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);
  env.params().add("Bp", 2.0);
  env.params().add("bh_spin", 0.9);

  // domain_comm<Conf> comm(env);
  grid_ks_t<Conf> grid;

  // auto bc = env.register_system<boundary_condition<Conf>>(env, grid);
  // auto solver =
  //     env.register_system<field_solver_gr_ks_cu<Conf>>(env, grid);
  // auto pusher = env.register_system<ptc_updater_gr_ks_cu<Conf>>(grid);
  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_gr_ks_sph>>(grid);
  // auto injector =
  //     env.register_system<bh_injector<Conf>>(env, grid);
  auto exporter = env.register_system<data_exporter<Conf>>(grid);

  env.init();

  // Prepare initial condition here
  vector_field<Conf> *B, *D, *B0, *D0;
  particle_data_t *ptc;
  env.get_data("B", &B);
  env.get_data("E", &D);
  env.get_data("particles", &ptc);

  initial_vacuum_wald(*B, *D, grid);

  vec_t<value_t, 3> x_global(math::log(4.0), M_PI * 0.5 - 0.2, 0.0);
  index_t<2> pos;
  vec_t<value_t, 3> x;
  grid.from_global(x_global, pos, x);
  auto ext = grid.extent();
  typename Conf::idx_t idx(pos, ext);

  for (int i = 0; i < 1; i++) {
    ptc->append_dev(x, {0.57367008, 0.0, 1.565}, idx.linear, 1000.0,
                    set_ptc_type_flag(0, PtcType::positron));
    // ptc->append_dev(x, {0.0, 0.0, 0.0}, idx.linear, 1000.0,
    //                 set_ptc_type_flag(0, PtcType::positron));
    // ptc->append_dev({0.5f, 0.5f, 0.0f}, , uint32_t cell)
  }
  CudaSafeCall(cudaDeviceSynchronize());

  env.run();

  return 0;
}
