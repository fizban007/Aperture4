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
#include "utils/logger.h"
#include "utils/vec.hpp"

using namespace Aperture;

template <typename T>
__global__ void
knl_print_vec_t(T v) {
  printf("rank is %d, v0 is %d, v1 is %d\n", v.rank(), v[0], v[1]);
}

void test_ext_func(const extent_t<2>& ext) {}

TEST_CASE("constructing vec_t", "[vec]") {
  auto v = vec<uint32_t>(32, 32);
  REQUIRE(v.rank() == 2);
  REQUIRE(v[0] == 32);
  REQUIRE(v[1] == 32);

  knl_print_vec_t<<<1, 1>>>(v);
  cudaDeviceSynchronize();

  // Default construction gives zero
  auto u = vec_t<uint32_t, 3>{};
  REQUIRE(u[0] == 0);
  REQUIRE(u[1] == 0);
  REQUIRE(u[2] == 0);
}

TEST_CASE("Aritmetic operators", "[vec]") {
  auto u = vec<int>(10, 20);
  auto v = vec<int>(3, 4);

  REQUIRE(u.product() == 200);
  REQUIRE(v.product() == 12);

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

TEST_CASE("construction using an array", "[vec]") {
  bool v[4] = {false, true, false, true};
  vec_t<bool, 4> b(v);

  for (int i = 0; i < 4; i++) {
    REQUIRE(b[i] == v[i]);
  }
}

TEST_CASE("construction using initializer lists", "[vec]") {
  test_ext_func({32, 32});
}
