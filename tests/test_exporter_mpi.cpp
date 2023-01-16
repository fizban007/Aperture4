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

#include "data/momentum_space.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/policies/exec_policy_host.hpp"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  auto& env = sim_env();
  using Conf = Config<3>;

  env.params().add("log_level", (int64_t)LogLevel::detail);
  env.params().add("N", std::vector<int64_t>({256, 256, 384}));
  env.params().add("ranks", std::vector<int64_t>({2, 2, 3}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 2.0, 3.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));
  env.params().add<int64_t>("downsample", 2);

  env.init();

  domain_comm<Conf, exec_policy_host> comm;
  grid_t<Conf> grid(comm);
  int num_bins[4] = {32, 32, 32, 32};
  float lowers[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float uppers[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  momentum_space<Conf> mom(grid, 4, num_bins, lowers, uppers, false);
  vector_field<Conf> vf(grid);
  vf.assign(3.0);
  data_exporter<Conf, exec_policy_host> exporter(grid, &comm);
  exporter.init();

  Logger::print_info("writing momenta");

  auto outfile = hdf_create("Data/momenta_mpi.h5", H5CreateMode::trunc_parallel);
  exporter.write(mom, "momentum", outfile);
  exporter.write(vf, "vector", outfile);

  return 0;
}
