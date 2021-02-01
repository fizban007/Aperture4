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

#include "framework/environment.h"
#include "framework/config.h"
#include "systems/ptc_updater_sph.h"
#include "utils/timer.h"
#include <fstream>
#include <iomanip>

using namespace Aperture;

int main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  Logger::print_info("value_t has size {}", sizeof(typename Conf::value_t));
  // sim_environment env;
  auto& env = sim_env();
  env.params().add("N", std::vector<int64_t>({512, 512}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("size", std::vector<double>({2.0, 3.14}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));
  env.params().add("max_ptc_num", 60000000l);

  auto grid = env.register_system<grid_sph_t<Conf>>();
  // auto pusher = env.register_system<ptc_updater_sph_cu<Conf>>(env, *grid);
  auto pusher = env.register_system<ptc_updater_cu<Conf>>(*grid);

  env.init();

  particle_data_t* ptc;
  env.get_data("particles", &ptc);
  pusher->fill_multiplicity(100);
  ptc->sort_by_cell_dev(grid->extent().size());
  Logger::print_info("There are {} particles in the array", ptc->number());

  int N = 100;
  double t = 0.0;
  for (int i = 0; i < N; i++) {
    timer::stamp();
    pusher->move_and_deposit(0.1, 2);
    double dt = 0.001 * timer::get_duration_since_stamp("us");
    t += dt;
    if (i % 10 == 0)
      Logger::print_info("Deposit took {}ms", dt);
  }
  t /= N;
  Logger::print_info("Ran deposit {} times, average time {}ms", N, t);
  Logger::print_info("Time per particle: {}ns", t / ptc->number() * 1.0e6);

  return 0;
}
