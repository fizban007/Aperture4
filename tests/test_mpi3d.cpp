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

using namespace Aperture;

int
main(int argc, char* argv[]) {
  // sim_environment env(&argc, &argv);
  auto& env = sim_environment::instance(&argc, &argv, true);
  typedef Config<3> Conf;

  env.params().add("log_level", int64_t(LogLevel::debug));
  env.params().add("N", std::vector<int64_t>({20, 20, 30}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("ranks", std::vector<int64_t>({2, 2, 3}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));
  env.params().add("size", std::vector<double>({10.0, 10.0, 10.0}));
  env.params().add("periodic_boundary", std::vector<bool>({true, true, true}));
  env.params().add("ptc_buffer_size", int64_t(100));
  env.params().add("ph_buffer_size", int64_t(100));

  auto comm = env.register_system<domain_comm<Conf>>();
  auto grid = env.register_system<grid_t<Conf>>(*comm);

  particles_t ptc(100, MemType::device_managed);
  photons_t ph(100, MemType::device_managed);
  ptc.set_segment_size(100);
  ph.set_segment_size(100);
  int N1 = grid->dims[0];
  int N2 = grid->dims[1];
  int N3 = grid->dims[2];
  if (comm->rank() == 0) {
    // This particle should go up one z rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {1.0, 0.0, 0.0}, 5 + 7 * N1 + 12 * N1 * N2);
    // This particle should go down one z rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {2.0, 0.0, 0.0}, 4 + 8 * N1 + 1 * N1 * N2);
    // This particle should go up one y rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {3.0, 0.0, 0.0}, 4 + (N2 - 2) * N1 + 5 * N1 * N2);
    // This particle should go down one y rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {4.0, 0.0, 0.0}, 5 + 1 * N1 + 5 * N1 * N2);

    // Corners
    // This particle should go up one z rank and one x rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {5.0, 0.0, 0.0}, (N1 - 1) + 8 * N1 + (N3 - 1) * N1 * N2);
    // This particle should go up one z rank, one x rank, and go down one y rank
    ptc.append(exec_tags::device{}, {0.5, 0.5, 0.5}, {6.0, 0.0, 0.0}, (N1 - 1) + 1 * N1 + (N3 - 1) * N1 * N2);
  }
  ptc.sort_by_cell(grid->size());
  if (ptc.number() > 0) {
    Logger::print_debug_all("Initially Rank {}, {}, {} has {} particles:", comm->domain_info().mpi_coord[0],
                            comm->domain_info().mpi_coord[1],
                            comm->domain_info().mpi_coord[2],
                            ptc.number());
  }

  comm->send_particles(ptc, *grid);
  comm->send_particles(ph, *grid);
  ptc.sort_by_cell(grid->size());
  ph.sort_by_cell(grid->size());

  if (ptc.number() > 0) {
    Logger::print_debug_all("Rank {}, {}, {} has {} particles:", comm->domain_info().mpi_coord[0],
                            comm->domain_info().mpi_coord[1],
                            comm->domain_info().mpi_coord[2],
                            ptc.number());
    for (int i = 0; i < ptc.number(); i++) {
      Logger::print_debug_all("Particle tag {}", ptc.p1[i]);
    }
  }

  return 0;
}
