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

#include "catch2/catch_all.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/legacy/ptc_updater_sph.h"
#include "systems/ptc_updater_base.h"
#include "utils/timer.h"
#include <fstream>
#include <iomanip>
#include <vector>

using namespace Aperture;

TEST_CASE("Comparing two spherical pushers", "[ptc_updater]") {
  typedef Config<2> Conf;

  auto& env = sim_env();
  env.reset();

  int64_t max_ptc_num = 10000;
  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("size", std::vector<double>({2.0, 3.14}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));

  auto ptc = env.register_data<particle_data_t>("particles", max_ptc_num,
                                                MemType::host_device);
  auto grid = env.register_system<grid_sph_t<Conf>>();
  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_spherical>>(*grid);
  // auto pusher = env.register_system<ptc_updater_cu<Conf>>(grid);

  env.init();

  vector_field<Conf>* B, *J;
  env.get_data("B", &B);
  env.get_data("J", &J);
  (*B)[2].assign_dev(100.0);

  // REQUIRE((*B)[2](20, 34) == Approx(10000.0f));

  ptc->append_dev(vec_t<Scalar, 3>(0.0, 0.0, 0.0),
                  vec_t<Scalar, 3>(0.0, 100.0, 0.0),
                  grid->get_idx(20, 34).linear, 0);

  double dt = 0.001;

  int N = 10;
  std::vector<Scalar> x1(N);
  std::vector<Scalar> x2(N);
  std::vector<uint32_t> cells(N);
  std::vector<Scalar> p1(N);
  std::vector<Scalar> p2(N);

  for (int i = 0; i < N; i++) {
    ptc->copy_to_host(false);

    x1[i] = ptc->x1[0];
    x2[i] = ptc->x2[0];
    p1[i] = ptc->p1[0];
    p2[i] = ptc->p2[0];
    cells[i] = ptc->cell[0];

    pusher->update_particles(dt, i);
  }

  vector_field<Conf> j_ref(*grid);
  j_ref.copy_from(*J);
  j_ref.copy_to_host();

  env.reset();

  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("size", std::vector<double>({2.0, 3.14}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));

  ptc = env.register_data<particle_data_t>("particles", max_ptc_num,
                                           MemType::host_device);
  grid = env.register_system<grid_sph_t<Conf>>();
  auto pusher_old = env.register_system<ptc_updater_old_sph_cu<Conf>>(*grid);

  env.init();

  env.get_data("B", &B);
  env.get_data("J", &J);
  (*B)[2].assign_dev(100.0);

  // REQUIRE((*B)[2](20, 34) == Approx(10000.0f));

  ptc->append_dev(vec_t<Scalar, 3>(0.0, 0.0, 0.0),
                  vec_t<Scalar, 3>(0.0, 100.0, 0.0),
                  grid->get_idx(20, 34).linear, 0);

  for (int i = 0; i < N; i++) {
    ptc->copy_to_host(false);
    CHECK(x1[i] == Approx(ptc->x1[0]));
    CHECK(x2[i] == Approx(ptc->x2[0]));
    CHECK(p1[i] == Approx(ptc->p1[0]));
    CHECK(p2[i] == Approx(ptc->p2[0]));
    CHECK(cells[i] == ptc->cell[0]);

    pusher_old->update_particles(dt, i);
  }

  J->copy_to_host();
  for (auto idx : j_ref[0].indices()) {
    CHECK(j_ref[0][idx] == (*J)[0][idx]);
    CHECK(j_ref[1][idx] == (*J)[1][idx]);
  }
}
