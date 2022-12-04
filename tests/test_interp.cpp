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
#include "core/multi_array.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"

using namespace Aperture;

TEST_CASE("1D linear interpolation", "[interp]") {
  auto v = make_multi_array<double>(extent(4), MemType::host_only);
  auto interp = interpolator<bspline<1>, 1>{};

  v[0] = 3.0;
  v[1] = 4.0;
  v[2] = 5.0;
  v[3] = 6.0;
  auto idx = v.get_idx(1);
  // auto pos = idx.get_pos();
  REQUIRE(interp(v, vec_t<float, 3>(0.1, 0.2, 0.3), idx) ==
          Catch::Approx(0.1 * 5.0 + 0.9 * 4.0));
}

TEST_CASE("2D cubic interpolation", "[interp]") {
  auto v = make_multi_array<double>(extent(4, 4), MemType::host_only);
  auto interp = interpolator<bspline<3>, 2>{};

  v.assign(1.0);
  auto idx = v.get_idx(1, 1);
  double a = interp(v, vec_t<float, 3>(0.3, 0.4, 0.5), idx);
  REQUIRE(a == Catch::Approx(1.0));

  v.emplace(0, {1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0,
                5.0, 6.0, 7.0});
  a = interp(v, vec_t<double, 3>(0.0, 1.0, 0.5), idx);
  REQUIRE(a == Catch::Approx(4.0));
}

TEST_CASE("3D linear interpolation", "[interp]") {
  auto ext = extent(4, 4, 4);
  auto v = make_multi_array<double>(ext, MemType::host_only);
  auto interp = interp_t<1, 3>{};

  for (auto idx : v.indices()) {
    auto pos = idx.get_pos();
    v[idx] = pos[0] + 0.1 * pos[1] + 0.2 * pos[2];
  }

  auto idx = default_idx_t<3>(index(2, 2, 2), ext);
  auto x = vec_t<double, 3>(0.3, 0.45, 0.65);
  auto a = interp(x, v, idx, ext);

  REQUIRE(a == Catch::Approx(2.6 * 0.7 * 0.55 * 0.35 + 3.6 * 0.3 * 0.55 * 0.35 +
                      2.7 * 0.7 * 0.45 * 0.35 + 3.7 * 0.3 * 0.45 * 0.35 +
                      2.8 * 0.7 * 0.55 * 0.65 + 3.8 * 0.3 * 0.55 * 0.65 +
                      2.9 * 0.7 * 0.45 * 0.65 + 3.9 * 0.3 * 0.45 * 0.65));

  auto interp_old = interpolator<bspline<1>, 3>{};
  auto b = interp_old(v, x, idx);

  REQUIRE(a == Catch::Approx(b));

  auto v2 = make_multi_array<double, idx_zorder_t>(ext);
  for (auto idx : v2.indices()) {
    auto pos = idx.get_pos();
    v2[idx] = pos[0] + 0.1 * pos[1] + 0.2 * pos[2];
  }
  auto idx2 = idx_zorder_t<3>(index(2, 2, 2), ext);
  b = interp(x, v2, idx2, ext);
  REQUIRE(a == Catch::Approx(b));
}
