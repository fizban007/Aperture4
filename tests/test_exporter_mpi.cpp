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

#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"

using namespace Aperture;

int
main(int argc, char* argv[]) {
  auto& env = sim_env();
  using Conf = Config<3>;

  env.params().add("N", std::vector<int64_t>({256, 256, 384}));
  env.params().add("nodes", std::vector<int64_t>({2, 2, 3}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 2.0, 3.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));
  env.params().add<int64_t>("downsample", 2);

  grid_t<Conf> grid;
  data_exporter<Conf> exporter(grid);
  exporter.init();
  // exporter.write_grid();

  return 0;
}
