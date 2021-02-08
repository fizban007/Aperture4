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

#include "catch.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/legacy/ptc_updater_old.h"
#include "systems/ptc_updater_base.h"
#include "systems/ptc_updater_simd.h"
#include "utils/timer.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <omp.h>

using namespace Aperture;

TEST_CASE("Comparing two pushers", "[ptc_updater]") {
  typedef Config<2> Conf;

  // omp_set_num_threads(1);

  auto& env = sim_env();
  env.reset();

  int64_t max_ptc_num = 1000000;
  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("size", std::vector<double>({2.0, 3.14}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));

  auto ptc = env.register_data<particle_data_t>("particles", max_ptc_num,
                                                MemType::host_only);
  grid_t<Conf> grid;
  auto pusher = env.register_system<
      // ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian>>(grid);
      ptc_updater_new<Conf, exec_policy_openmp, coord_policy_cartesian>>(grid);
  // auto pusher = env.register_system<ptc_updater_cu<Conf>>(grid);

  env.init();

  vector_field<Conf>*B, *J;
  env.get_data("B", &B);
  env.get_data("J", &J);
  B->set_values(2, [](auto x, auto y, auto z) {
    return 100.0 + x * 20.0 + y * 30.0;
  });

  // REQUIRE((*B)[2](20, 34) == Approx(10000.0f));

  // int n_ptc = 100;
  // for (int i = 0; i < n_ptc; i++) {
  //   ptc->append(vec_t<Scalar, 3>(0.0, 0.0, 0.0),
  //               vec_t<Scalar, 3>(0.0, 100.0, 0.0),
  //               grid.get_idx(20, 34 + i / 10).linear, 0);
  // }
  pusher->fill_multiplicity(10);

  particle_data_t ptc_backup(max_ptc_num);
  ptc_backup.copy_from(*ptc);

  double dt = 0.01;

  int N = 10;
  // std::vector<Scalar> x1(N * n_ptc);
  // std::vector<Scalar> x2(N * n_ptc);
  // std::vector<uint32_t> cells(N * n_ptc);
  // std::vector<Scalar> p1(N * n_ptc);
  // std::vector<Scalar> p2(N * n_ptc);

  std::vector<Scalar> x1(ptc->number());
  std::vector<Scalar> x2(ptc->number());
  std::vector<uint32_t> cell(ptc->number());
  std::vector<Scalar> p1(ptc->number());
  std::vector<Scalar> p2(ptc->number());

  for (int i = 0; i < N; i++) {
    pusher->update_particles(dt, i);
  }
  for (int i = 0; i < ptc->number(); i++) {
    x1[i] = ptc->x1[i];
    x2[i] = ptc->x2[i];
    p1[i] = ptc->p1[i];
    p2[i] = ptc->p2[i];
    cell[i] = ptc->cell[i];
  }

  vector_field<Conf> j_ref(grid);
  j_ref.copy_from(*J);

  env.reset();

  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({2, 2}));
  env.params().add("size", std::vector<double>({2.0, 3.14}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));

  ptc = env.register_data<particle_data_t>("particles", max_ptc_num,
                                           // MemType::host_device);
                                           MemType::host_only);
  auto pusher_old =
      env.register_system<ptc_updater_simd<Conf, coord_policy_cartesian>>(grid);

  env.init();

  env.get_data("B", &B);
  env.get_data("J", &J);
  // (*B)[2].assign(100.0);
  B->set_values(2, [](auto x, auto y, auto z) {
    return 100.0 + x * 20.0 + y * 30.0;
  });

  // REQUIRE((*B)[2](20, 34) == Approx(10000.0f));

  // for (int i = 0; i < n_ptc; i++) {
  //   ptc->append(vec_t<Scalar, 3>(0.0, 0.0, 0.0),
  //               vec_t<Scalar, 3>(0.0, 100.0, 0.0),
  //               grid.get_idx(20, 34 + i / 10).linear, 0);
  // }
  // pusher_old->fill_multiplicity(10);
  ptc->copy_from(ptc_backup);

  for (int i = 0; i < N; i++) {
    pusher_old->update_particles(dt, i);
  }

  for (int i = 0; i < ptc->number(); i++) {
    // Logger::print_info("i {}", i);
    REQUIRE(x1[i] == ptc->x1[i]);
    REQUIRE(x2[i] == ptc->x2[i]);
    REQUIRE(p1[i] == ptc->p1[i]);
    REQUIRE(p2[i] == ptc->p2[i]);
    REQUIRE(cell[i] == ptc->cell[i]);
  }

  for (auto idx : j_ref[0].indices()) {
    REQUIRE(j_ref[0][idx] == (*J)[0][idx]);
    REQUIRE(j_ref[1][idx] == (*J)[1][idx]);
  }
}
