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
#include "data/fields.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "utils/logger.h"
#include "systems/policies/exec_policy_gpu.hpp"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // sim_environment env(&argc, &argv);
  auto& env = sim_environment::instance(&argc, &argv, true);
  typedef Config<2> Conf;

  env.params().add("log_level", int64_t(LogLevel::detail));
  env.params().add("N", std::vector<int64_t>({10, 10}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("ranks", std::vector<int64_t>({2, 2}));
  env.params().add("lower", std::vector<double>({1.0, 2.0}));
  env.params().add("size", std::vector<double>({100.0, 10.0}));
  env.params().add("periodic_boundary", std::vector<bool>({true, true}));
  env.params().add("ptc_buffer_size", int64_t(100));
  env.params().add("ph_buffer_size", int64_t(100));

  // auto comm = env.register_system<domain_comm<Conf, exec_policy_host>>();
  domain_comm<Conf, exec_policy_gpu> comm;
  auto grid = env.register_system<grid_t<Conf>>(comm);

  particles_t ptc(100, MemType::device_only);
  photons_t ph(100, MemType::device_only);
  ptc.set_segment_size(10);
  ph.set_segment_size(10);
  int N1 = grid->dims[0];
  ptc.set_num(18);
  if (comm.rank() == 0) {
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 1 + (N1 - 2) * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, (N1 - 1) + 3 * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, (N1 - 1) + 3 * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {3.0, 1.0, 0.0}, 1 + (N1 - 1) * N1);
    ptc_append(exec_tags::device{}, ptc, {0.5, 0.5, 0.5}, {4.0, -1.0, 0.0}, (N1 - 1) + 0 * N1);
    ptc_append(exec_tags::device{}, ph, {0.1, 0.2, 0.3}, {1.0, 1.0, 1.0}, 2 + 8 * N1, 0.0);
  }
  Logger::print_debug_all("initially Rank {} has {} particles:", comm.rank(),
                          ptc.number());
  // ptc.sort_by_cell(grid->size());
  // ph.sort_by_cell(grid->size());
  comm.send_particles(ptc, *grid);
  comm.send_particles(ph, *grid);
  ptc_sort_by_cell(exec_tags::device{}, ptc, grid->size());
  ptc_sort_by_cell(exec_tags::device{}, ph, grid->size());

  Logger::print_debug_all("Rank {} has {} particles:", comm.rank(),
                          ptc.number());
  // ptc.copy_to_host();
  // for (unsigned int i = 0; i < ptc.number(); i++) {
  //   auto c = ptc.cell[i];
  //   Logger::print_debug_all("cell {}, {}", c % N1, c / N1);
  // }
  Logger::print_debug_all("Rank {} has {} photons:", comm.rank(), ph.number());

  // typename Conf::multi_array_t v(grid->extent());
  vector_field<Conf> f(*grid);
  auto& v = f[2];
  v.assign(comm.rank());
  // comm.send_guard_cells(v, *grid);
  comm.send_guard_cells(f);
  v.copy_to_host();

  for (int n = 0; n < comm.size(); n++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (n == comm.rank()) {
      std::cout << "This is the initial content from rank " << n << std::endl;
      for (int j = 0; j < grid->dims[1]; j++) {
        for (int i = 0; i < grid->dims[0]; i++) {
          std::cout << v(i, j) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  v.assign(comm.rank());
  // comm.send_add_guard_cells(v, *grid);
  // comm.send_add_array_guard_cells_single_dir(v, *grid, 0, -1);
  // comm.send_add_array_guard_cells_single_dir(v, *grid, 0, 1);
  comm.send_add_guard_cells(f);

  v.copy_to_host();

  for (int n = 0; n < comm.size(); n++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (n == comm.rank()) {
      std::cout << "This is the content from rank " << n << std::endl;
      for (int j = 0; j < grid->dims[1]; j++) {
        for (int i = 0; i < grid->dims[0]; i++) {
          std::cout << v(i, j) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  return 0;
}
