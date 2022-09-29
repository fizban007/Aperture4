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
#include "core/math.hpp"
#include "core/buffer.hpp"
#include "utils/kernel_helper.hpp"

using namespace Aperture;

TEST_CASE("Checking double precision math functions", "[math]") {
  buffer<double> x(2, MemType::device_managed);
  x[0] = 3.0;
  kernel_launch({1, 1}, [] __device__(double* x) {
      x[1] = math::sin(x[0]);
    }, x.dev_ptr());
  GpuSafeCall(gpuDeviceSynchronize());

  REQUIRE(x[1] == std::sin(x[0]));
  REQUIRE(math::abs(x[0]) == std::abs(x[0]));
  REQUIRE(math::sqrt(x[0]) == std::sqrt(x[0]));
  REQUIRE(math::cos(3.0f) == std::cos(3.0f));
}
