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

#ifndef __SCATTERING_H_
#define __SCATTERING_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "utils/util_functions.h"

namespace Aperture {

constexpr double re_square = 7.91402e-26;

template <typename value_t>
HOST_DEVICE value_t
sigma_ic(value_t x) {
  if (x < 1.0e-3) {
    return 1.0f - 2.0f * x + 26.0f * x * x / 5.0f;
  } else {
    double l = math::log(1.0 + 2.0 * x);
    return 0.75 *
           ((1.0 + x) * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - l) / cube(x) +
            0.5 * l / x - (1.0 + 3.0 * x) / square(1.0 + 2.0 * x));
  }
}

template <typename value_t>
HOST_DEVICE value_t
sigma_gg(value_t beta) {
  return (1.0 - square(beta)) *
         ((3.0 - beta * beta * beta * beta) * log((1.0 + beta) / (1.0 - beta)) -
          2.0 * beta * (2.0 - beta * beta));
}

}  // namespace Aperture

#endif  // __SCATTERING_H_
