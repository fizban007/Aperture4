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

#pragma once

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

namespace Aperture {

namespace Metric_KS {

HD_INLINE Scalar
rho2(Scalar a, Scalar r, Scalar th) {
  Scalar cth = math::cos(th);
  return r * r + a * a * cth * cth;
}

HD_INLINE Scalar
rho2(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return r * r + a * a * cth * cth;
}

HD_INLINE Scalar
Z(Scalar a, Scalar r, Scalar th) {
  return 2.0f * r / rho2(a, r, th);
}

HD_INLINE Scalar
Z(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 2.0f * r / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
Delta(Scalar a, Scalar r) {
  return r * r + a * a - 2.0f * r;
}

HD_INLINE Scalar
Sigma(Scalar a, Scalar r, Scalar th) {
  Scalar r2a2 = r * r + a * a;
  Scalar sth = math::sin(th);
  return square(r2a2) - square(a * sth) * (r2a2 - 2.0f * r);
}

HD_INLINE Scalar
Sigma(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar r2a2 = r * r + a * a;
  return square(r2a2) - square(a * sth) * (r2a2 - 2.0f * r);
}

HD_INLINE Scalar
alpha(Scalar a, Scalar r, Scalar th) {
  return 1.0f / math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
alpha(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f / math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
beta1(Scalar a, Scalar r, Scalar th) {
  return 2.0f * r / (r * (2.0f + r) + square(a * math::cos(th)));
}

HD_INLINE Scalar
beta1(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 2.0f * r / (r * (2.0f + r) + square(a * cth));
}

HD_INLINE Scalar
g_11(Scalar a, Scalar r, Scalar th) {
  return 1.0f + Z(a, r, th);
}

HD_INLINE Scalar
g_11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f + Z(a, r, sth, cth);
}

HD_INLINE Scalar
g_22(Scalar a, Scalar r, Scalar th) {
  return rho2(a, r, th);
}

HD_INLINE Scalar
g_22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return rho2(a, r, sth, cth);
}

HD_INLINE Scalar
g_33(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return Sigma(a, r, th) * sth * sth / rho2(a, r, th);
}

HD_INLINE Scalar
g_33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return Sigma(a, r, sth, cth) * sth * sth / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
g_13(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return -a * sth * sth * (1.0f + Z(a, r, th));
}

HD_INLINE Scalar
g_13(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return -a * sth * sth * (1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
ag_11(Scalar a, Scalar r, Scalar th) {
  return math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
ag_22(Scalar a, Scalar r, Scalar th) {
  return rho2(a, r, th) / math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return rho2(a, r, sth, cth) / math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
ag_33(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return Sigma(a, r, th) * sth * sth /
         (rho2(a, r, th) * math::sqrt(1.0f + Z(a, r, th)));
}

HD_INLINE Scalar
ag_33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return Sigma(a, r, sth, cth) * sth * sth /
         (rho2(a, r, sth, cth) * math::sqrt(1.0f + Z(a, r, sth, cth)));
}

HD_INLINE Scalar
ag_13(Scalar a, Scalar r, Scalar th) {
  Scalar sth = math::sin(th);
  return -a * sth * sth * math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
ag_13(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return -a * sth * sth * math::sqrt(1.0f + Z(a, r, sth, cth));
}

HD_INLINE Scalar
sqrt_gamma(Scalar a, Scalar r, Scalar th) {
  return rho2(a, r, th) * math::sin(th) * math::sqrt(1.0f + Z(a, r, th));
}

HD_INLINE Scalar
sqrt_gamma(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return rho2(a, r, sth, cth) * sth * math::sqrt(1.0f + Z(a, r, sth, cth));
}

// This returns the composite value of sqrt(gamma) * beta1
HD_INLINE Scalar
sq_gamma_beta(Scalar a, Scalar r, Scalar th) {
  Scalar z = Z(a, r, th);
  return 2.0f * r * math::sin(th) / math::sqrt(1.0f + z);
}

HD_INLINE Scalar
sq_gamma_beta(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar z = Z(a, r, sth, cth);
  return 2.0f * r * sth / math::sqrt(1.0f + z);
}

// These are the upper index gamma matrix elements
HD_INLINE Scalar
gu11(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  Scalar rho = rho2(a, r, sth, cth);
  return (a * a + r * r) / rho - 2.0f * r / (rho + 2.0f * r);
}

HD_INLINE Scalar
gu13(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return a / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
gu22(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f / rho2(a, r, sth, cth);
}

HD_INLINE Scalar
gu33(Scalar a, Scalar r, Scalar sth, Scalar cth) {
  return 1.0f / rho2(a, r, sth, cth) / square(sth);
}

// Dot product between two upper-index vectors
HD_INLINE Scalar
dot_product_u(const vec_t<Scalar, 3>& v1, const vec_t<Scalar, 3>& v2, Scalar a,
              Scalar r, Scalar sth, Scalar cth) {
  return g_11(a, r, sth, cth) * v1[0] * v2[0] +
         g_13(a, r, sth, cth) * (v1[0] * v2[2] + v1[2] * v2[0]) +
         g_22(a, r, sth, cth) * v1[1] * v2[1] +
         g_33(a, r, sth, cth) * v1[2] * v2[2];
}

// Dot product between two lower-index vectors
HD_INLINE Scalar
dot_product_l(const vec_t<Scalar, 3>& v1, const vec_t<Scalar, 3>& v2, Scalar a,
              Scalar r, Scalar sth, Scalar cth) {
  Scalar rho = rho2(a, r, sth, cth);
  return gu11(a, r, sth, cth) * v1[0] * v2[0] +
         (v1[1] * v2[1] + v1[2] * v2[2] / square(sth) +
          (v1[0] * v2[2] + v1[2] * v2[0]) * a) /
             rho;
}

// function to compute u^0 from lower index components
HD_INLINE Scalar
u0(Scalar a, Scalar r, Scalar sth, Scalar cth, const vec_t<Scalar, 3>& u,
   bool is_photon = false) {
  Scalar ep = (is_photon ? 0.0f : 1.0f);
  return math::sqrt(dot_product_l(u, u, a, r, sth, cth) + ep) /
         alpha(a, r, sth, cth);
}

// function to compute u^0 from lower index components
HD_INLINE Scalar
u0(Scalar a, Scalar r, Scalar th, const vec_t<Scalar, 3>& u,
   bool is_photon = false) {
  Scalar sth = math::sin(th);
  Scalar cth = math::cos(th);
  return u0(a, r, sth, cth, u, is_photon);
}

// Compute u_0 from lower index u_i
HD_INLINE Scalar
u_0(Scalar a, Scalar r, Scalar sth, Scalar cth, const vec_t<Scalar, 3>& u,
    bool is_photon = false) {
  Scalar ep = (is_photon ? 0.0f : 1.0f);
  return u[0] * beta1(a, r, sth, cth) -
         alpha(a, r, sth, cth) *
             math::sqrt(dot_product_l(u, u, a, r, sth, cth) + ep);
}

// Compute u_0 from lower index u_i
HD_INLINE Scalar
u_0(Scalar a, Scalar r, Scalar th, const vec_t<Scalar, 3>& u,
    bool is_photon = false) {
  Scalar sth = math::sin(th);
  Scalar cth = math::cos(th);
  return u_0(a, r, sth, cth, u, is_photon);
}

HD_INLINE Scalar
rH(Scalar a) {
  return 1.0f + math::sqrt(1.0f - a * a);
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_to_ZAMO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float th) {
  vec_t<Float, 4> result;

  auto sth = math::sin(th);
  auto cth = math::cos(th);
  auto A = Sigma(a, r, sth, cth);
  auto D = Delta(a, r);
  auto S = rho2(a, r, sth, cth);

  result[0] = u[0] * math::sqrt(A / S / D) + u[3] * 2.0 * a * r / math::sqrt(A * D * S);
  result[1] = (2.0 * r * u[0] + D * u[1] + a * u[3]) / math::sqrt(D * S);
  result[2] = u[2] / math::sqrt(S);
  result[3] = math::sqrt(S / A) * u[3] / sth;
  return result;
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_from_ZAMO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float th) {
  vec_t<Float, 4> result;

  auto sth = math::sin(th);
  auto cth = math::cos(th);
  auto A = Sigma(a, r, sth, cth);
  auto D = Delta(a, r);
  auto S = rho2(a, r, sth, cth);

  result[0] = (u[0] * S * math::sqrt(D) - u[3] * 2.0 * a * r * sth) / math::sqrt(A * S);
  result[1] = -2.0 * r * math::sqrt(S / D / A) * u[0] + math::sqrt(S / D) * u[1] +
      (4.0 * r * r - A) * a * sth * u[3] / math::sqrt(A * S) / D;
  result[2] = u[2] * math::sqrt(S);
  result[3] = math::sqrt(A / S) * u[3] * sth;
  return result;
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_to_FIDO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float sth, Float cth) {
  vec_t<Float, 4> result;

  auto sqrtS = math::sqrt(rho2(a, r, sth, cth));
  auto alf = alpha(a, r, sth, cth);

  result[0] = u[0] / alf - u[1] * alf * Z(a, r, sth, cth);
  result[1] = alf * u[1];
  result[2] = u[2] / sqrtS;
  result[3] = (a * sth * u[1] + u[3] / sth) / sqrtS;
  return result;
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_to_FIDO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float th) {
  auto sth = math::sin(th);
  auto cth = math::cos(th);
  return convert_to_FIDO_lower(u, a, r, sth, cth);
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_from_FIDO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float sth, Float cth) {
  vec_t<Float, 4> result;

  auto sqrtS = math::sqrt(rho2(a, r, sth, cth));
  auto alf = alpha(a, r, sth, cth);

  result[0] = alf * (u[0] + u[1] * Z(a, r, sth, cth));
  result[1] = u[1] / alf;
  result[2] = u[2] * sqrtS;
  result[3] = sqrtS * sth * u[3] - a * sth * sth * u[1] / alf;
  return result;
}

template <typename Float>
HOST_DEVICE vec_t<Float, 4>
convert_from_FIDO_lower(const vec_t<Float, 4>& u, Float a, Float r, Float th) {
  auto sth = math::sin(th);
  auto cth = math::cos(th);
  return convert_from_FIDO_lower(u, a, r, sth, cth);
}

}  // namespace Metric_KS

}  // namespace Aperture
