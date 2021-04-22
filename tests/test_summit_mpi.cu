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

#include "core/particles.h"
#include "framework/environment.h"
#include "framework/config.h"
#include "systems/grid.h"
#include "systems/domain_comm_async.h"
#include "utils/logger.h"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // sim_environment env(&argc, &argv);
  auto& env = sim_environment::instance(&argc, &argv, true);
  typedef Config<3> Conf;

  env.params().add("log_level", int64_t(LogLevel::debug));
  env.params().add("N", std::vector<int64_t>({100, 100, 100}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("nodes", std::vector<int64_t>({2, 2, 2}));
  env.params().add("lower", std::vector<double>({1.0, 2.0, 0.0}));
  env.params().add("size", std::vector<double>({100.0, 10.0, 30.0}));
  env.params().add("periodic_boundary", std::vector<bool>({true, true, true}));
  env.params().add("ptc_buffer_size", int64_t(100));
  env.params().add("ph_buffer_size", int64_t(100));

  auto comm = env.register_system<domain_comm_async<Conf>>();
  auto grid = env.register_system<grid_t<Conf>>(*comm);

  typename Conf::multi_array_t v(grid->extent());
  v.assign_dev(comm->rank());
  size_t free_mem, total_mem;
  for (int i = 0; i < 100; i++) {
    comm->send_guard_cells(v, *grid);
    cudaMemGetInfo( &free_mem, &total_mem );
    Logger::print_info("GPU memory: free = {} GiB, total = {} GiB", free_mem/1.0e9, total_mem/1.0e9);
  }
  v.copy_to_host();
}
