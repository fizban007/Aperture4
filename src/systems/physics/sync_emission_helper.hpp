/*
 * Copyright (c) 2022 Alex Chen.
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

#pragma once

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/multi_array.hpp"
#include "core/random.h"
#include "core/typedefs_and_constants.h"
#include "utils/binary_search.h"
#include "utils/util_functions.h"

namespace Aperture {

struct sync_emission_helper_t {
  using value_t = Scalar;

  int nx;
  value_t logx_min, logx_max, dlogx;
  value_t* ptr_lookup;

  // Function to generate synchro-curvature photon energy. The resulting energy
  // is normalized such that cyclotron frequency at BQ is at electron rest mass,
  // or hbar e BQ / m_e c = m_e c^2. We assume the photon energy is much less
  // than gamma. Otherwise we need the quantum prescription of synchrotron
  // emission template <typename Rng>
  HOST_DEVICE value_t gen_curv_photon(value_t gamma, value_t Rc, value_t BQ,
                                      rand_state& state) const {
    value_t e_c = 3.0f * cube(gamma) / (2.0f * BQ * Rc);
    value_t l, h, b;
    value_t u = rng_uniform<value_t>(state);

    auto m = upper_bound(u, ptr_lookup, nx);

    if (m > 0) {
      l = ptr_lookup[m - 1];
      h = ptr_lookup[m];
      b = (u - l) / (h - l) + m - 1;
    } else {
      b = 0;
    }
    value_t x = math::exp(logx_min + b * dlogx);
    return std::min(x * e_c, gamma - 1.001f);
  }

  // template <typename Rng>
  HOST_DEVICE value_t gen_curv_photon(value_t e_c, value_t gamma,
                                      rand_state& state) const {
    value_t l, h, b;
    value_t u = rng_uniform<value_t>(state);

    auto m = upper_bound(u, ptr_lookup, nx);

    if (m > 0) {
      l = ptr_lookup[m - 1];
      h = ptr_lookup[m];
      b = (u - l) / (h - l) + m - 1;
    } else {
      b = 0;
    }
    value_t x = math::exp(logx_min + b * dlogx);
    return std::min(x * e_c, gamma - 1.001f);
  }

  // template <typename Rng>
  HOST_DEVICE value_t gen_sync_photon(value_t gamma, value_t accel_perp,
                                      value_t BQ, rand_state& state) const {
    return gen_curv_photon(gamma, gamma / accel_perp, BQ, state);
  }
};

}  // namespace Aperture
