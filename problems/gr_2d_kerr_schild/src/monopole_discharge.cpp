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
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/radiative_transfer.h"
#include "systems/radiation/gr_ks_ic_radiation_scheme.hpp"
#include "utils/util_functions.h"

using namespace std;

namespace Aperture {

template <typename Conf>
void initial_vacuum_monopole(vector_field<Conf> &B, vector_field<Conf> &D,
                             const grid_ks_t<Conf> &grid);
} // namespace Aperture

using namespace Aperture;

int main(int argc, char *argv[]) {
  typedef Config<2, Scalar> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf> comm;
  grid_ks_t<Conf> grid(comm);

  auto solver = env.register_system<field_solver_gr_ks_cu<Conf>>(grid, &comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_gpu, coord_policy_gr_ks_sph>>(grid,
                                                                       comm);
  auto injector = env.register_system<bh_injector<Conf>>(grid);
  // auto rt = env.register_system<
  //     radiative_transfer<Conf, exec_policy_gpu, coord_policy_gr_ks_sph,
  //                        gr_ks_ic_radiation_scheme>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_gpu>>(grid, &comm);

  env.init();

  // Prepare initial condition here
  vector_field<Conf> *B, *D, *B0, *D0;
  particle_data_t *ptc;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);
  // env.get_data("B", &B);
  // env.get_data("E", &D);
  env.get_data("particles", &ptc);

  value_t Bp = 1000.0;
  env.params().get_value("Bp", Bp);

  // initial_vacuum_monopole(*B, *D, grid);
  initial_vacuum_monopole(*B0, *D0, grid);
  B->copy_from(*B0);

  pusher->fill_multiplicity(10, 1.0, 0.1);
  // Logger::print_info("number of particles is {}", ptc->number());
  // vec_t<value_t, 3> x_global(math::log(4.0), M_PI * 0.5 - 0.2, 0.0);
  // index_t<2> pos;
  // vec_t<value_t, 3> x;
  // grid.from_global(x_global, pos, x);
  // auto ext = grid.extent();
  // typename Conf::idx_t idx(pos, ext);

  // for (int i = 0; i < 1; i++) {
  //   ptc->append(exec_tags::device{}, {0.5, 0.5, 0.0}, {2000.0, 0.0, 1000.0}, 200 + grid.dims[0] * 400, 1000.0,
  //                   set_ptc_type_flag(0, PtcType::positron));
  //   // ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, , uint32_t cell)
  // }
  // CudaSafeCall(cudaDeviceSynchronize());

  // index_t<2> pos(200, 768);
  // auto ext = grid.extent();
  // typename Conf::idx_t idx(pos, ext);

  // for (int i = 0; i < 1; i++) {
  //   ptc->append(exec_tags::device{}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, idx.linear,
  //   1000.0,
  //                   set_ptc_type_flag(0, PtcType::positron));
  //   // ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, , uint32_t cell)
  // }
  // CudaSafeCall(cudaDeviceSynchronize());

  env.run();

  return 0;
}
