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

#include "catch.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/ptc_updater.h"
#include <fstream>
#include <iomanip>

using namespace Aperture;

TEST_CASE("Particle push in a uniform B field", "[pusher][.]") {
  Logger::init(0, LogLevel::debug);
  typedef Config<2> Conf;
  // sim_environment env;
  auto& env = sim_env();
  env.params().add("log_level", 2l);
  env.params().add("N", std::vector<int64_t>({64, 64, 64}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));

  auto ptc = env.register_data<particle_data_t>("particles", 10000,
                                                MemType::host_device);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>();
  auto pusher = env.register_system<ptc_updater<Conf>>(*grid);

  env.init();

  vector_field<Conf>* B;
  env.get_data("B", &B);
  (*B)[2].assign(10000.0);
  REQUIRE((*B)[2](20, 34) == Approx(10000.0f));

  ptc->append(vec_t<Scalar, 3>(0.0, 0.0, 0.0), vec_t<Scalar, 3>(0.0, 1000.0, 0.0),
              grid->get_idx(20, 34).linear, 0);
  ptc->weight[0] = 1.0;

  double dt = 0.001;
  uint32_t N = 4000;
  std::vector<double> x1(N);
  std::vector<double> x2(N);

  SECTION("Boris push") {
    pusher->use_pusher(Pusher::boris);
    for (uint32_t i = 0; i < N; i++) {
      auto pos = grid->pos_global(grid->idx_at(ptc->cell[0]).get_pos(),
                                  vec_t<Scalar, 3>(ptc->x1[0], ptc->x2[0], 0.0));
      x1[i] = pos[0];
      x2[i] = pos[1];

      pusher->push_default(dt);
      pusher->move_and_deposit(dt, i);
    }

    std::ofstream output("pusher_boris_result.csv");
    for (uint32_t i = 0; i < N; i++) {
      output << std::setprecision(12) << x1[i] << ", " << x2[i] << ", " << ptc->E[0] << std::endl;
    }
  }

  SECTION("Vay push") {
    pusher->use_pusher(Pusher::vay);
    for (uint32_t i = 0; i < N; i++) {
      auto pos = grid->pos_global(grid->idx_at(ptc->cell[0]).get_pos(),
                                  vec_t<Scalar, 3>(ptc->x1[0], ptc->x2[0], 0.0));
      x1[i] = pos[0];
      x2[i] = pos[1];

      pusher->push_default(dt);
      pusher->move_and_deposit(dt, i);
    }

    std::ofstream output("pusher_vay_result.csv");
    for (uint32_t i = 0; i < N; i++) {
      output << std::setprecision(12) << x1[i] << ", " << x2[i] << ", " << ptc->E[0] << std::endl;
    }
  }

  SECTION("Higuera push") {
    pusher->use_pusher(Pusher::higuera);
    for (uint32_t i = 0; i < N; i++) {
      auto pos = grid->pos_global(grid->idx_at(ptc->cell[0]).get_pos(),
                                  vec_t<Scalar, 3>(ptc->x1[0], ptc->x2[0], 0.0));
      x1[i] = pos[0];
      x2[i] = pos[1];

      pusher->push_default(dt);
      pusher->move_and_deposit(dt, i);
    }

    std::ofstream output("pusher_higuera_result.csv");
    for (uint32_t i = 0; i < N; i++) {
      output << std::setprecision(12) << x1[i] << ", " << x2[i] << ", " << ptc->E[0] << std::endl;
    }
  }
}
