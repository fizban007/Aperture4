#include "catch.hpp"
#include "core/grid.hpp"
#include "data/fields.hpp"
#include "data/field_helpers.h"
#include "framework/config.h"
#include "framework/environment.hpp"

using namespace Aperture;

void
set_initial_field(vector_field<Config<2>>& field) {
  field.set_values(
      0,
      [](auto x1, auto x2, auto x3) {
        return x1 * x1 + x2 + x3 * x3 * x3;
      });
}

TEST_CASE("Memtype is correct for fields", "[fields]") {
  typedef Config<2> Conf;
  Grid<2> grid;
  grid.dims[0] = 32;
  grid.dims[1] = 32;

  SECTION("host_only") {
    scalar_field<Conf> f(grid, MemType::host_only);
    REQUIRE(f[0].mem_type() == MemType::host_only);
    REQUIRE(f[0].host_allocated() == true);
#ifdef CUDA_ENABLED
    REQUIRE(f[0].dev_allocated() == false);
#endif
  }

  SECTION("device_managed") {
    vector_field<Conf> f(grid, MemType::device_managed);
    REQUIRE(f[0].host_allocated() == false);
    REQUIRE(f[0].mem_type() == MemType::device_managed);
#ifdef CUDA_ENABLED
    REQUIRE(f[1].dev_allocated() == true);
    REQUIRE(f[2].host_ptr() != nullptr);
#endif
  }

  SECTION("device_host") {
    vector_field<Conf> f(grid, MemType::host_device);
    REQUIRE(f[0].mem_type() == MemType::host_device);
    REQUIRE(f[0].host_allocated() == true);
#ifdef CUDA_ENABLED
    REQUIRE(f[1].dev_allocated() == true);
#endif
  }

}

TEST_CASE("Initializing fields", "[fields]") {
  Config<2> conf;
  Grid<2> g2;
  g2.dims[0] = 32;
  g2.dims[1] = 32;
  vector_field<Config<2>> vf(g2);

  vf[0].assign_host(3.0f);
  for (auto idx : vf[0].indices()) {
    REQUIRE(vf[0][idx] == 3.0f);
  }
}

TEST_CASE("setting initial value from a function", "[fields]") {
  Grid<2> g2;

  for (int i = 0; i < 2; i++) {
    g2.dims[i] = 256;
    g2.guard[i] = 2;
    g2.skirt[i] = 2;
    g2.sizes[i] = 1.0;
    g2.lower[i] = 0.0;
    g2.delta[i] = g2.sizes[i] / g2.reduced_dim(i);
    g2.inv_delta[i] = 1.0 / g2.delta[i];
  }

  Config<2> conf;
  vector_field<Config<2>> vf(g2);
  vector_field<Config<2>> vf2(g2);

  // Normal initialization
  set_initial_field(vf);

  for (auto idx : vf[0].indices()) {
    auto pos = idx.get_pos();
    double x1 = g2.pos<0>(pos[0], 0);
    double x2 = g2.pos<1>(pos[1], 0);
    double x3 = 0.0;
    REQUIRE(vf[0][idx] == Approx(x1 * x1 + x2 + x3 * x3 * x3));
  }

}

TEST_CASE("Resampling field 1D", "[fields]") {
  Grid<1> grid;
  for (int i = 0; i < 1; i++) {
    grid.dims[i] = 256;
    grid.guard[i] = 2;
    grid.skirt[i] = 2;
    grid.sizes[i] = 1.0;
    grid.lower[i] = 0.0;
    grid.delta[i] = grid.sizes[i] / grid.reduced_dim(i);
    grid.inv_delta[i] = 1.0 / grid.delta[i];
  }
  scalar_field<Config<1>> f(grid, field_type::vert_centered,
                            MemType::host_only);
  scalar_field<Config<1>> f2(grid, field_type::cell_centered,
                             MemType::host_only);
  f.set_values(0, [](Scalar x1, Scalar x2, Scalar x3) {
                    return x1 + 2.0 * x2 + 3.0 * x3;
                  });
  f2.set_values(0, [](Scalar x1, Scalar x2, Scalar x3) {
                     return x1 + 2.0 * x2 + 3.0 * x3;
                   });
  scalar_field<Config<1>> f3(grid, field_type::cell_centered,
                             MemType::host_only);
  resample(f[0], f3[0], index(grid.guard[0]), f.stagger(), f3.stagger());
  for (int i = grid.guard[0]; i < grid.dims[0] - grid.guard[0]; i++) {
    // Logger::print_debug("f3 {}, f2 {}", f3[0][i], f2[0][i]);
    REQUIRE(f3[0][i] == Approx(f2[0][i]));
  }
}


TEST_CASE("Resampling field 2D", "[fields]") {
  Grid<2> grid;
  for (int i = 0; i < 2; i++) {
    grid.dims[i] = 36;
    grid.guard[i] = 2;
    grid.skirt[i] = 2;
    grid.sizes[i] = 1.0;
    grid.lower[i] = 0.0;
    grid.delta[i] = grid.sizes[i] / grid.reduced_dim(i);
    grid.inv_delta[i] = 1.0 / grid.delta[i];
  }
  vector_field<Config<2>> f(grid, field_type::cell_centered,
                            MemType::host_only);
  vector_field<Config<2>> f2(grid, field_type::vert_centered,
                             MemType::host_only);
  f.set_values(1, [](Scalar x1, Scalar x2, Scalar x3) {
                    return x1 + 2.0 * x2 + 3.0 * x3;
                  });
  f2.set_values(1, [](Scalar x1, Scalar x2, Scalar x3) {
                     return x1 + 2.0 * x2 + 3.0 * x3;
                   });
  vector_field<Config<2>> f3(grid, field_type::vert_centered,
                             MemType::host_only);
  resample(f[1], f3[1], index(grid.guard[0], grid.guard[1]),
           f.stagger(), f3.stagger());
  for (auto idx : f3[1].indices()) {
    auto pos = idx.get_pos();
    if (grid.is_in_bound(pos))
      REQUIRE(f3[1][idx] == Approx(f2[1][idx]));
  }
}
