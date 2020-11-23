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
#include "systems/grid_ks.h"
#include "utils/gauss_quadrature.h"

using namespace Aperture;

TEST_CASE("Simple test of the Gauss quadrature routine", "[numerical]") {
  double v1 = gauss_quad([] (double x) {
    double a = 0.99;
    double theta = 1.53;
    return Metric_KS::sqrt_gamma(a, x, theta);
  }, 0.9, 0.95);

  REQUIRE(v1 == Approx(0.0761187329413635));

  double v2 = gauss_quad([] (double x) {
    double a = 0.9;
    double r = 0.9;
    return Metric_KS::ag_13(a, r, x);
  }, 1.4, 1.5);

  REQUIRE(v2 == Approx(-0.158252699121388));

  double v3 = gauss_quad([] (double x) {
    double a = 0.9;
    double r = 0.9;
    return Metric_KS::ag_33(a, r, x);
  }, 1.4, 1.5);

  REQUIRE(v3 == Approx(0.18560086426904));
}
