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
#include "utils/interpolation.hpp"
#include "utils/logger.h"
#include "core/multi_array.hpp"

using namespace Aperture;

TEST_CASE("1D linear interpolation", "[interp]") {
  auto v = make_multi_array<double>(extent(4), MemType::host_only);
  auto interp = interpolator<bspline<1>, 1>{};

  v.assign(1.0);
  auto idx = v.get_idx(1);
  // auto pos = idx.get_pos();
  REQUIRE(interp(v, vec_t<float, 3>(0.1, 0.1, 0.1),
                 idx) == Approx(1.0));
}

TEST_CASE("2D cubic interpolation", "[interp]") {
  auto v = make_multi_array<double>(extent(4, 4), MemType::host_only);
  auto interp = interpolator<bspline<3>, 2>{};

  v.assign(1.0);
  auto idx = v.get_idx(1, 1);
  double a = interp(v, vec_t<float, 3>(0.3, 0.4, 0.5),
                    idx);
  REQUIRE(a == Approx(1.0));

  Logger::print_info("a is {}", a);

  v.emplace(0, {1.0, 2.0, 3.0, 4.0,
                2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 5.0, 6.0,
                4.0, 5.0, 6.0, 7.0});
  a = interp(v, vec_t<double, 3>(0.0, 1.0, 0.5), idx);
  REQUIRE(a == Approx(4.0));
}
