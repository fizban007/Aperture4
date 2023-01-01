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

#include "core/gpu_translation_layer.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"

namespace Aperture {

HD_INLINE Scalar
dipole_curv_radius(Scalar r, Scalar th) {
  Scalar sinth = std::max(
      math::sin(th), (Scalar)1.0e-5);  // Avoid the fringe case of sinth = 0
  Scalar costh2 = 1.0f - sinth * sinth;
  Scalar tmp = 1.0f + 3.0f * costh2;
  Scalar Rc = r * tmp * math::sqrt(tmp) / (3.0f * sinth * (1.0f + costh2));
  return Rc;
}

HD_INLINE Scalar
dipole_curv_radius_above_polar_cap(Scalar x, Scalar y, Scalar z) {
  Scalar r_cyl = math::sqrt(x * x + y * y);
  Scalar z_r = z + 1.0f;  // R* is our unit
  Scalar th = atan2(r_cyl, z_r);
  Scalar r = math::sqrt(z_r * z_r + r_cyl * r_cyl);
  return dipole_curv_radius(r, th);
}

HD_INLINE Scalar
magnetic_pair_production_rate(Scalar b, Scalar eph, Scalar sinth,
                              Scalar Rpc_over_Rstar) {
  // The coefficient is 0.23 * \alpha_f * R_pc / \labmdabar_c, seems no reason
  // to rescale return 4.35e13 * b * sinth * math::exp(-4.0f / 3.0f / (0.5f *
  // eph * b * sinth));
  return 4.35e13 * Rpc_over_Rstar * b *
         math::exp(-4.0f / 3.0f / (0.5f * eph * b * sinth));
}

}  // namespace Aperture
