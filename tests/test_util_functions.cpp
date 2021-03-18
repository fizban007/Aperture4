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
#include "utils/logger.h"
#include "utils/util_functions.h"
#include "utils/binary_search.h"
#include <cstdint>

using namespace Aperture;

TEST_CASE("Powers of two", "[util_funcs]") {
  REQUIRE(not_power_of_two(8) == false);
  REQUIRE(not_power_of_two(17) == true);
  REQUIRE(not_power_of_two(UINT64_MAX) == true);

  REQUIRE(is_power_of_two(64) == true);
  REQUIRE(is_power_of_two(INT32_MAX) == false);
  REQUIRE(is_power_of_two(65536) == true);

  REQUIRE(next_power_of_two(54) == 64);
  REQUIRE(next_power_of_two(32167) == 32768);
}

TEST_CASE("Binary search", "[util_funcs]") {
  int N = 100;
  std::vector<float> v(N);

  for (int i = 0; i < N; i++) {
    v[i] = square((float)i / N);
  }

  float u = 0.2f;
  float l, h;
  auto n = upper_bound(u, v.data(), N);
  REQUIRE(v[n - 1] < u);
  REQUIRE(v[n] > u);

  u = 0.5f;
  n = upper_bound(u, v.data(), N);
  REQUIRE(v[n - 1] < u);
  REQUIRE(v[n] > u);

}
