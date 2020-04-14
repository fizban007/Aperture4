#include "catch.hpp"
#include "core/grid.hpp"
#include "data/fields.hpp"
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
    double x3 = g2.pos<2>(pos[2], 0);
    REQUIRE(vf[0][idx] == float(x1 * x1 + x2 + x3 * x3 * x3));
  }

}