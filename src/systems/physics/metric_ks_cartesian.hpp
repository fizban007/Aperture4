/*
 * Copyright (c) 2026 Alex Chen.
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

// Kerr-Schild metric in Cartesian coordinates (x, y, z).
//
// The full 4D metric is  g_ab = eta_ab + f l_a l_b  where
//   f  = 2 M r / rho^2,   rho^2 = r^2 + a^2 z^2 / r^2,
//   l_a = (1, l_x, l_y, l_z)  is an ingoing principal null covector.
//
// 3+1 quantities derived from this:
//   alpha     = 1 / sqrt(1 + f)
//   beta^i    = f / (1 + f) * l_i          (shift vector, upper index)
//   gamma_ij  = delta_ij + f l_i l_j       (spatial 3-metric)
//   gamma^ij  = delta^ij - f/(1+f) l_i l_j (inverse spatial 3-metric)
//   sqrt(gamma) = sqrt(1 + f)
//
// Units: M = 1 throughout.  The spin parameter a is in [0, 1).

#pragma once

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

namespace Aperture {

namespace Metric_KS_Cart {

// ---------------------------------------------------------------------------
// Boyer-Lindquist radius from Cartesian coordinates.
//
// Solves  r^4 - (x^2+y^2+z^2 - a^2) r^2 - a^2 z^2 = 0  for r^2 > 0.
// ---------------------------------------------------------------------------
HD_INLINE Scalar
radius(Scalar x, Scalar y, Scalar z, Scalar a) {
  Scalar w = x * x + y * y + z * z;
  Scalar a2 = a * a;
  Scalar d = w - a2;
  // r^2 = (d + sqrt(d^2 + 4 a^2 z^2)) / 2
  Scalar r2 = 0.5f * (d + math::sqrt(d * d + 4.0f * a2 * z * z));
  return math::sqrt(r2);
}

// ---------------------------------------------------------------------------
// Null covector spatial components  l_i  in Cartesian coordinates.
//   l_x = (r x + a y) / (r^2 + a^2)
//   l_y = (r y - a x) / (r^2 + a^2)
//   l_z = z / r
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 3>
null_covector(Scalar x, Scalar y, Scalar z, Scalar a, Scalar r) {
  Scalar r2a2_inv = 1.0f / (r * r + a * a);
  vec_t<Scalar, 3> l;
  l[0] = (r * x + a * y) * r2a2_inv;
  l[1] = (r * y - a * x) * r2a2_inv;
  l[2] = z / r;
  return l;
}

// ---------------------------------------------------------------------------
// rho^2 = r^2 + a^2 cos^2(theta) = r^2 + a^2 z^2 / r^2
// ---------------------------------------------------------------------------
HD_INLINE Scalar
rho2(Scalar r, Scalar z, Scalar a) {
  return r * r + a * a * z * z / (r * r);
}

// ---------------------------------------------------------------------------
// Scalar function  f = 2 r / rho^2   (with M = 1)
// ---------------------------------------------------------------------------
HD_INLINE Scalar
f_ks(Scalar r, Scalar z, Scalar a) {
  return 2.0f * r / rho2(r, z, a);
}

// ---------------------------------------------------------------------------
// Lapse:  alpha = 1 / sqrt(1 + f)
// ---------------------------------------------------------------------------
HD_INLINE Scalar
alpha(Scalar f) {
  return 1.0f / math::sqrt(1.0f + f);
}

HD_INLINE Scalar
alpha(Scalar x, Scalar y, Scalar z, Scalar a) {
  Scalar r = radius(x, y, z, a);
  return alpha(f_ks(r, z, a));
}

// ---------------------------------------------------------------------------
// Shift vector (upper index):  beta^i = f / (1 + f) * l_i
//
// Note: l_i = l^i since the spatial part of l has unit Euclidean norm,
// so raising with delta^ij is trivial.
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 3>
beta_upper(Scalar f, const vec_t<Scalar, 3>& l) {
  Scalar coeff = f / (1.0f + f);
  vec_t<Scalar, 3> b;
  b[0] = coeff * l[0];
  b[1] = coeff * l[1];
  b[2] = coeff * l[2];
  return b;
}

HD_INLINE vec_t<Scalar, 3>
beta_upper(Scalar x, Scalar y, Scalar z, Scalar a) {
  Scalar r = radius(x, y, z, a);
  Scalar fv = f_ks(r, z, a);
  auto l = null_covector(x, y, z, a, r);
  return beta_upper(fv, l);
}

// ---------------------------------------------------------------------------
// Spatial 3-metric:  gamma_ij = delta_ij + f l_i l_j
//
// Stored as a symmetric 3x3 in a vec_t<Scalar, 6>:
//   [g_xx, g_xy, g_xz, g_yy, g_yz, g_zz]
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 6>
gamma_lower(Scalar f, const vec_t<Scalar, 3>& l) {
  vec_t<Scalar, 6> g;
  g[0] = 1.0f + f * l[0] * l[0];  // g_xx
  g[1] = f * l[0] * l[1];          // g_xy
  g[2] = f * l[0] * l[2];          // g_xz
  g[3] = 1.0f + f * l[1] * l[1];  // g_yy
  g[4] = f * l[1] * l[2];          // g_yz
  g[5] = 1.0f + f * l[2] * l[2];  // g_zz
  return g;
}

// ---------------------------------------------------------------------------
// Inverse spatial 3-metric:  gamma^ij = delta^ij - f/(1+f) l_i l_j
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 6>
gamma_upper(Scalar f, const vec_t<Scalar, 3>& l) {
  Scalar c = -f / (1.0f + f);
  vec_t<Scalar, 6> g;
  g[0] = 1.0f + c * l[0] * l[0];  // g^xx
  g[1] = c * l[0] * l[1];          // g^xy
  g[2] = c * l[0] * l[2];          // g^xz
  g[3] = 1.0f + c * l[1] * l[1];  // g^yy
  g[4] = c * l[1] * l[2];          // g^yz
  g[5] = 1.0f + c * l[2] * l[2];  // g^zz
  return g;
}

// ---------------------------------------------------------------------------
// sqrt(det(gamma)) = sqrt(1 + f)
// ---------------------------------------------------------------------------
HD_INLINE Scalar
sqrt_gamma(Scalar f) {
  return math::sqrt(1.0f + f);
}

HD_INLINE Scalar
sqrt_gamma(Scalar x, Scalar y, Scalar z, Scalar a) {
  Scalar r = radius(x, y, z, a);
  return sqrt_gamma(f_ks(r, z, a));
}

// ---------------------------------------------------------------------------
// Combined quantity  sqrt(gamma) * beta^i = f / sqrt(1+f) * l_i
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 3>
sq_gamma_beta(Scalar f, const vec_t<Scalar, 3>& l) {
  Scalar coeff = f / math::sqrt(1.0f + f);
  vec_t<Scalar, 3> sb;
  sb[0] = coeff * l[0];
  sb[1] = coeff * l[1];
  sb[2] = coeff * l[2];
  return sb;
}

// ---------------------------------------------------------------------------
// Lower an upper-index vector:  v_i = gamma_ij v^j
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 3>
lower(const vec_t<Scalar, 3>& v, Scalar f, const vec_t<Scalar, 3>& l) {
  // gamma_ij v^j = v_i + f l_i (l_j v^j)
  Scalar ldotv = l[0] * v[0] + l[1] * v[1] + l[2] * v[2];
  vec_t<Scalar, 3> result;
  result[0] = v[0] + f * l[0] * ldotv;
  result[1] = v[1] + f * l[1] * ldotv;
  result[2] = v[2] + f * l[2] * ldotv;
  return result;
}

// ---------------------------------------------------------------------------
// Raise a lower-index vector:  v^i = gamma^ij v_j
// ---------------------------------------------------------------------------
HD_INLINE vec_t<Scalar, 3>
raise(const vec_t<Scalar, 3>& v, Scalar f, const vec_t<Scalar, 3>& l) {
  // gamma^ij v_j = v^i - f/(1+f) l^i (l_j v_j)
  Scalar ldotv = l[0] * v[0] + l[1] * v[1] + l[2] * v[2];
  Scalar c = f / (1.0f + f);
  vec_t<Scalar, 3> result;
  result[0] = v[0] - c * l[0] * ldotv;
  result[1] = v[1] - c * l[1] * ldotv;
  result[2] = v[2] - c * l[2] * ldotv;
  return result;
}

// ---------------------------------------------------------------------------
// Dot product of two upper-index vectors:  gamma_ij u^i v^j
// ---------------------------------------------------------------------------
HD_INLINE Scalar
dot_product_u(const vec_t<Scalar, 3>& u, const vec_t<Scalar, 3>& v,
              Scalar f, const vec_t<Scalar, 3>& l) {
  Scalar flat = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  Scalar lu = l[0] * u[0] + l[1] * u[1] + l[2] * u[2];
  Scalar lv = l[0] * v[0] + l[1] * v[1] + l[2] * v[2];
  return flat + f * lu * lv;
}

// ---------------------------------------------------------------------------
// Dot product of two lower-index vectors:  gamma^ij u_i v_j
// ---------------------------------------------------------------------------
HD_INLINE Scalar
dot_product_l(const vec_t<Scalar, 3>& u, const vec_t<Scalar, 3>& v,
              Scalar f, const vec_t<Scalar, 3>& l) {
  Scalar flat = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  Scalar lu = l[0] * u[0] + l[1] * u[1] + l[2] * u[2];
  Scalar lv = l[0] * v[0] + l[1] * v[1] + l[2] * v[2];
  return flat - f / (1.0f + f) * lu * lv;
}

// ---------------------------------------------------------------------------
// Outer horizon radius:  r_+ = 1 + sqrt(1 - a^2)   (M = 1)
// ---------------------------------------------------------------------------
HD_INLINE Scalar
rH(Scalar a) {
  return 1.0f + math::sqrt(1.0f - a * a);
}

// ---------------------------------------------------------------------------
// Convenience: compute all 3+1 quantities at a point in one pass.
//
// Outputs (via reference):
//   r      - Boyer-Lindquist radius
//   l      - spatial null covector
//   fv     - scalar f = 2r/rho^2
//   alp    - lapse alpha
//   bu     - shift vector beta^i  (upper index)
//   sgb    - sqrt(gamma) * beta^i
//   sg     - sqrt(det(gamma))
// ---------------------------------------------------------------------------
HD_INLINE void
compute_all(Scalar x, Scalar y, Scalar z, Scalar a,
            Scalar& r, vec_t<Scalar, 3>& l, Scalar& fv,
            Scalar& alp, vec_t<Scalar, 3>& bu,
            vec_t<Scalar, 3>& sgb, Scalar& sg) {
  r = radius(x, y, z, a);
  l = null_covector(x, y, z, a, r);
  fv = f_ks(r, z, a);
  Scalar one_plus_f = 1.0f + fv;
  Scalar sqrt_opf = math::sqrt(one_plus_f);
  alp = 1.0f / sqrt_opf;
  sg = sqrt_opf;
  Scalar beta_coeff = fv / one_plus_f;
  Scalar sgbeta_coeff = fv / sqrt_opf;
  bu[0] = beta_coeff * l[0];
  bu[1] = beta_coeff * l[1];
  bu[2] = beta_coeff * l[2];
  sgb[0] = sgbeta_coeff * l[0];
  sgb[1] = sgbeta_coeff * l[1];
  sgb[2] = sgbeta_coeff * l[2];
}

}  // namespace Metric_KS_Cart

}  // namespace Aperture
