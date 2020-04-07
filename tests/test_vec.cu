#include "catch.hpp"
#include "utils/logger.h"
#include "utils/vec.hpp"

using namespace Aperture;

template <typename T>
__global__ void
knl_print_vec_t(T v) {
  printf("rank is %d, v0 is %d, v1 is %d\n", v.rank(), v[0], v[1]);
}

TEST_CASE("constructing vec_t", "[vec]") {
  auto v = vec<uint32_t>(32, 32);
  REQUIRE(v.rank() == 2);
  REQUIRE(v[0] == 32);
  REQUIRE(v[1] == 32);

  knl_print_vec_t<<<1, 1>>>(v);
  cudaDeviceSynchronize();
}

TEST_CASE("Aritmetic operators", "[vec]") {
  auto u = vec<int>(10, 20);
  auto v = vec<int>(3, 4);

  REQUIRE(u.size() == 200);
  REQUIRE(v.size() == 12);

  auto x = u + v;
  REQUIRE(x[0] == 13);
  REQUIRE(x[1] == 24);

  x = u - v;
  REQUIRE(x[0] == 7);
  REQUIRE(x[1] == 16);

  // Comparison works
  auto y = vec<int>(7, 16);
  REQUIRE(x == y);
}

TEST_CASE("extent_t", "[vec]") {
  extent_t<2> ext(12, 32);

  REQUIRE(ext[0] == 12);
  REQUIRE(ext[1] == 32);
}
