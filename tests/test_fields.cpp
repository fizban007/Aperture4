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

#include "catch2/catch_all.hpp"
#include "core/detail/multi_array_helpers.h"
#include "core/grid.hpp"
#include "data/fields.h"
#include "framework/config.h"

using namespace Aperture;

template <typename Conf>
void set_initial_field(vector_field<Conf> &field) {
  field.set_values(
      0, [](auto x1, auto x2, auto x3) { return x1 * x1 + x2 + x3 * x3 * x3; });
}

TEST_CASE("Memtype is correct for fields", "[fields]") {
  typedef Config<2, float> Conf;
  Grid<2, float> grid;
  grid.dims[0] = 32;
  grid.dims[1] = 32;

  SECTION("host_only") {
    scalar_field<Conf> f(grid, MemType::host_only);
    REQUIRE(f[0].mem_type() == MemType::host_only);
    REQUIRE(f[0].host_allocated() == true);
#ifdef GPU_ENABLED
    REQUIRE(f[0].dev_allocated() == false);
#endif
  }

  SECTION("device_managed") {
    vector_field<Conf> f(grid, MemType::device_managed);
    REQUIRE(f[0].host_allocated() == false);
    REQUIRE(f[0].mem_type() == MemType::device_managed);
#ifdef GPU_ENABLED
    REQUIRE(f[1].dev_allocated() == true);
    REQUIRE(f[2].host_ptr() != nullptr);
#endif
  }

  SECTION("device_host") {
    vector_field<Conf> f(grid, MemType::host_device);
    REQUIRE(f[0].mem_type() == MemType::host_device);
    REQUIRE(f[0].host_allocated() == true);
#ifdef GPU_ENABLED
    REQUIRE(f[1].dev_allocated() == true);
#endif
  }
}

TEST_CASE("Initializing fields", "[fields]") {
  using Conf = Config<2, float>;
  Conf::grid_t g2;
  g2.dims[0] = 32;
  g2.dims[1] = 32;
  vector_field<Conf> vf(g2);

  vf[0].assign_host(3.0f);
  for (auto idx : vf[0].indices()) {
    REQUIRE(vf[0][idx] == 3.0f);
  }
}

TEST_CASE("setting initial value from a function", "[fields]") {
  using Conf = Config<2, float>;
  Conf::grid_t g2;

  for (int i = 0; i < 2; i++) {
    g2.dims[i] = 256;
    g2.N[i] = 252;
    g2.guard[i] = 2;
    g2.sizes[i] = 1.0;
    g2.lower[i] = 0.0;
    g2.delta[i] = g2.sizes[i] / g2.reduced_dim(i);
    g2.inv_delta[i] = 1.0 / g2.delta[i];
  }

  vector_field<Conf> vf(g2);
  vector_field<Conf> vf2(g2);

  // Normal initialization
  set_initial_field(vf);

  for (auto idx : vf[0].indices()) {
    auto pos = idx.get_pos();
    double x1 = g2.coord<0>(pos[0], 0);
    double x2 = g2.coord<1>(pos[1], 0);
    double x3 = 0.0;
    // REQUIRE(vf[0][idx] == Approx(x1 * x1 + x2 + x3 * x3 * x3));
    REQUIRE_THAT(vf[0][idx], Catch::Matchers::WithinRel(x1 * x1 + x2 + x3 * x3 * x3, 1.0e-5));
  }
}

TEST_CASE("Resampling field 1D", "[fields]") {
  using Conf = Config<1, float>;
  Conf::grid_t grid;
  for (int i = 0; i < 1; i++) {
    grid.dims[i] = 256;
    grid.N[i] = 252;
    grid.guard[i] = 2;
    grid.sizes[i] = 1.0;
    grid.lower[i] = 0.0;
    grid.delta[i] = grid.sizes[i] / grid.reduced_dim(i);
    grid.inv_delta[i] = 1.0 / grid.delta[i];
  }
  scalar_field<Conf> f(grid, field_type::vert_centered, MemType::host_only);
  scalar_field<Conf> f2(grid, field_type::cell_centered, MemType::host_only);
  f.set_values(0, [](Scalar x1, Scalar x2, Scalar x3) {
    return x1 + 2.0 * x2 + 3.0 * x3;
  });
  f2.set_values(0, [](Scalar x1, Scalar x2, Scalar x3) {
    return x1 + 2.0 * x2 + 3.0 * x3;
  });
  scalar_field<Conf> f3(grid, field_type::cell_centered, MemType::host_only);
  resample(exec_tags::host{}, f[0], f3[0], grid.guards(), grid.guards(), f.stagger(),
           f3.stagger());
  for (int i = grid.guard[0]; i < grid.dims[0] - grid.guard[0]; i++) {
    // Logger::print_debug("f3 {}, f2 {}", f3[0][i], f2[0][i]);
    // REQUIRE(f3[0][i] == Approx(f2[0][i]));
    REQUIRE_THAT(f3[0][i], Catch::Matchers::WithinULP(f2[0][i], 1));
  }
}

TEST_CASE("Resampling field 2D", "[fields]") {
  using Conf = Config<2, float>;
  Conf::grid_t grid = Conf::make_grid({32, 32}, {2, 2}, {1.0, 1.0}, {0.0, 0.0});
  // for (int i = 0; i < 2; i++) {
  //   grid.dims[i] = 36;
  //   grid.N[i] = 32;
  //   grid.guard[i] = 2;
  //   grid.sizes[i] = 1.0;
  //   grid.lower[i] = 0.0;
  //   grid.delta[i] = grid.sizes[i] / grid.reduced_dim(i);
  //   grid.inv_delta[i] = 1.0 / grid.delta[i];
  // }
  REQUIRE(grid.dims[0] == 36);
  REQUIRE(grid.dims[1] == 36);
  // REQUIRE(grid.sizes[1] == Approx(1.0));
  REQUIRE_THAT(grid.sizes[1], Catch::Matchers::WithinULP(1.0, 1));

  vector_field<Conf> f(grid, field_type::cell_centered, MemType::host_only);
  vector_field<Conf> f2(grid, field_type::vert_centered, MemType::host_only);
  f.set_values(1, [](Scalar x1, Scalar x2, Scalar x3) {
    return x1 + 2.0 * x2 + 3.0 * x3;
  });
  f2.set_values(1, [](Scalar x1, Scalar x2, Scalar x3) {
    return x1 + 2.0 * x2 + 3.0 * x3;
  });
  vector_field<Conf> f3(grid, field_type::vert_centered, MemType::host_only);
  resample(exec_tags::host{}, f[1], f3[1], grid.guards(), grid.guards(), f.stagger(),
           f3.stagger());
  for (auto idx : f3[1].indices()) {
    // auto pos = idx.get_pos();
    auto pos = idx.get_pos();
    if (grid.is_in_bound(pos))
      // CHECK(f3[1][idx] == Approx(f2[1][idx]));
      CHECK_THAT(f3[1][idx], Catch::Matchers::WithinULP(f2[1][idx], 1));
  }
}
