/*
 * Copyright (c) 2022 Alex Chen.
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

#include "catch2/catch_all.hpp"
#include "data/particle_data.h"
#include "data/fields.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  auto& env = sim_env();
  using Conf = Config<3>;

  env.params().add("log_level", (int64_t)LogLevel::detail);
  env.params().add("N", std::vector<int64_t>({64, 64, 128}));
  env.params().add("nodes", std::vector<int64_t>({2, 2, 2}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 1.0, 2.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));
  env.params().add<int64_t>("downsample", 2);

  domain_comm<Conf> comm;
  grid_t<Conf> grid(comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);
  // scalar_field<Conf> fs(grid, MemType::device_managed);
  auto fs = env.register_data<scalar_field<Conf>>("scalar", grid, MemType::device_managed);
  fs->include_in_snapshot(true);
  // vector_field<Conf> fv(grid, MemType::device_managed);
  auto fv = env.register_data<vector_field<Conf>>("vector", grid, MemType::device_managed);
  fv->include_in_snapshot(true);
  auto ptc = env.register_data<particle_data_t>("ptc", 1000, MemType::device_managed);
  ptc->include_in_snapshot(true);
  auto states = env.register_data<rng_states_t>("rng_states");
  states->include_in_snapshot(true);

  env.init();

  fs->set_values(0, [](auto x1, auto x2, auto x3) { return 3.0f; });
  fv->set_values(0, [](auto x1, auto x2, auto x3) {
    return 1.0f;
  });
  fv->set_values(1, [](auto x1, auto x2, auto x3) {
    return 2.0f;
  });
  fv->set_values(2, [](auto x1, auto x2, auto x3) {
    return 3.0f;
  });

  // particle_data_t ptc(1000, MemType::device_managed);
  for (int i = 0; i < 100; i++) {
    ptc->append({0.1, 0.2, 0.3}, {1.0, 2.0, 3.0},
                32 + 32 * 68 + 64 * 68 * 68);
  }

  // rng_states_t states;
  // states->init();

  // data_exporter<Conf> exporter(grid, &comm);
  // exporter.init();

  exporter->write_snapshot("Data/snapshot_mpi.h5", 0, 0.0);
  // auto outfile = hdf_create("Data/snapshot_mpi.h5", H5CreateMode::trunc_parallel);
  // Logger::print_info("writing scalar field");
  // exporter.write(fs, "scalar", outfile, true);

  // Logger::print_info("writing vector field");
  // exporter.write(fv, "vector", outfile, true);

  // Logger::print_info("writing particles");
  // exporter.write(ptc, "ptc", outfile, true);

  // Logger::print_info("writing rng_states");
  // exporter.write(states, "rng_states", outfile, true);

  return 0;
}
