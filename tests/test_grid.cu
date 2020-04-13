#include "catch.hpp"
#include "core/grid.hpp"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "framework/parse_params.hpp"
#include "systems/grid.h"
#include "systems/grid_logsph.h"
#include "utils/index.hpp"
#include "utils/kernel_helper.hpp"

using namespace Aperture;

TEST_CASE("Using grid", "[grid]") {
  SECTION("1D grid") {
    Grid<1> g1;

    g1.dims[0] = 12;
    g1.guard[0] = 2;
    g1.skirt[0] = 2;
    g1.sizes[0] = 1.0;
    g1.lower[0] = 0.0;
    g1.delta[0] = g1.sizes[0] / g1.reduced_dim<0>();
    g1.inv_delta[0] = 1.0 / g1.delta[0];

    REQUIRE(g1.reduced_dim<0>() == 8);
    REQUIRE(g1.pos<0>(2, true) == 0.0f);
    REQUIRE(g1.pos<0>(6, 0.7f) == 4.7f * g1.delta[0]);
    REQUIRE(g1.pos<1>(6, 11.6f) == 11.6f);

    REQUIRE(g1.is_in_bound(8) == true);
    REQUIRE(g1.is_in_bound(10) == false);
  }

  SECTION("2D grid") {
    Grid<2> g2;

    for (int i = 0; i < 2; i++) {
      g2.dims[i] = 12;
      g2.guard[i] = 2;
      g2.skirt[i] = 2;
      g2.sizes[i] = 1.0;
      g2.lower[i] = 0.0;
      g2.delta[i] = g2.sizes[i] / g2.reduced_dim(i);
      g2.inv_delta[i] = 1.0 / g2.delta[i];
    }

    REQUIRE(g2.reduced_dim<1>() == 8);
    REQUIRE(g2.pos<1>(2, true) == 0.0f);
    REQUIRE(g2.pos<1>(6, 0.7f) == 4.7f * g2.delta[1]);
    REQUIRE(g2.pos<2>(6, 11.6f) == 11.6f);

    bool b = g2.is_in_bound(index(4, 8));
    REQUIRE(b == true);
    b = g2.is_in_bound(11, 6);
    REQUIRE(b == false);

    int z = g2.find_zone(index(1, 7));
    REQUIRE(z == 3);
    z = g2.find_zone(index(11, 8));
    REQUIRE(z == 5);
    z = g2.find_zone(index(11, 11));
    REQUIRE(z == 8);
  }
}

TEST_CASE("Kernels with grid", "[grid][kernel]") {
  Grid<3> g3;

  for (int i = 0; i < 3; i++) {
    g3.dims[i] = 12;
    g3.guard[i] = 2;
    g3.skirt[i] = g3.guard[i];
    g3.sizes[i] = 1.0;
    g3.lower[i] = 0.0;
    g3.delta[i] = g3.sizes[i] / g3.reduced_dim(i);
    g3.inv_delta[i] = 1.0 / g3.delta[i];
  }

  kernel_launch(
      exec_policy(1, 1),
      [] __device__(Grid<3> g) {
        if (g.is_in_bound(6, 6, 6)) printf("pos is %f\n", g.pos<2>(6, 0.7f));
      },
      g3);
  CudaSafeCall(cudaDeviceSynchronize());
}

TEST_CASE("Grid initialization on constant memory", "[grid][kernel]") {
  Logger::init(0, LogLevel::debug);
  sim_environment env;
  typedef Config<3> Conf;

  // env.params().add("N", std::vector<int64_t>({32, 32, 32}));
  env.params().add("N", std::vector<int64_t>({32, 32, 32}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));

  auto grid = env.register_system<grid_t<Conf>>(env);
  // // env.init();

  kernel_launch({1, 1}, [] __device__() {
    printf("N is %ux%ux%u\n", dev_grid_3d.reduced_dim(0),
           dev_grid_3d.reduced_dim(1), dev_grid_3d.reduced_dim(1));
  });
  CudaSafeCall(cudaDeviceSynchronize());
}

TEST_CASE("Grid with different indexing schemes", "[grid][index]") {
  // All a grid does is simply keeping track of the linear spaces in the 3
  // different dimensions. Linear indexing scheme is a mapping between this grid
  // space to the linear memory space.
}

TEST_CASE("Logsph grid", "[grid][logsph]") {
  sim_environment env;
  typedef Config<2> Conf;

  // env.params().add("N", std::vector<int64_t>({32, 32, 32}));
  env.params().add("N", std::vector<int64_t>({32, 32, 32}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));

  int arr[3];
  get_from_store("guard", arr, env.params());
  auto grid = env.register_system<grid_t<Conf>>(env);
}
