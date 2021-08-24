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
#include "systems/physics/lorentz_transform.hpp"

using namespace Aperture;

TEST_CASE("Lorentz Transform", "[physics]") {
  // 1D velocity addition
  vec_t<double, 3> u_orig(0.9, 0.0, 0.0);
  vec_t<double, 3> v(0.8, 0.0, 0.0);

  vec_t<double, 3> u_prime = lorentz_transform_velocity(u_orig, v);

  REQUIRE(u_prime[0] == Approx((u_orig[0] - v[0]) / (1.0 - u_orig[0] * v[0])));

  u_orig = vec_t<double, 3>(0.0, 0.0, 0.0);
  u_prime = lorentz_transform_velocity(u_orig, v);
  REQUIRE(u_prime[0] == Approx(-v[0]));
}
