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

// Metric interface for a spatial 3-metric in spherical coordinates (r, θ, φ).
//
// Axisymmetry is assumed: γ_{rθ} = γ_{θφ} = 0, β^θ = β^φ = 0.
//
// The two concrete metrics provided here, flat_spherical_metric and
// ks_spherical_metric, are plain-old-data structs whose evaluation
// methods are all HD_INLINE.  They can be captured by value into GPU
// lambdas.  The `prismatic_mesh_metric::compute_metric` entry point is
// templated on the concrete metric type so no virtual dispatch is
// required.

#pragma once

#include "core/cuda_control.h"
#include "core/typedefs_and_constants.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include <cmath>

namespace Aperture {

// =========================================================================
// Flat space in spherical coordinates:  ds² = dr² + r²dθ² + r²sin²θ dφ²
// =========================================================================
struct flat_spherical_metric {
  HD_INLINE Scalar g_rr(Scalar r, Scalar sth, Scalar cth) const { return 1; }
  HD_INLINE Scalar g_thth(Scalar r, Scalar sth, Scalar cth) const {
    return r * r;
  }
  HD_INLINE Scalar g_phph(Scalar r, Scalar sth, Scalar cth) const {
    return r * r * sth * sth;
  }
  HD_INLINE Scalar g_rth(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  HD_INLINE Scalar g_rph(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  HD_INLINE Scalar g_thph(Scalar r, Scalar sth, Scalar cth) const { return 0; }

  HD_INLINE Scalar sqrt_gamma(Scalar r, Scalar sth, Scalar cth) const {
    return r * r * sth;
  }
  HD_INLINE Scalar alpha(Scalar r, Scalar sth, Scalar cth) const { return 1; }
  HD_INLINE Scalar beta_r(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  HD_INLINE Scalar sq_gamma_beta_r(Scalar r, Scalar sth, Scalar cth) const {
    return 0;
  }
};

// =========================================================================
// Kerr-Schild in spherical KS coordinates.
// Wraps the existing Metric_KS functions from metric_kerr_schild.hpp
// (those are already HD_INLINE), so this struct is device-safe.
// =========================================================================
struct ks_spherical_metric {
  Scalar a;  // spin parameter

  HD_INLINE explicit ks_spherical_metric(Scalar a_) : a(a_) {}
  HD_INLINE ks_spherical_metric() : a(0) {}

  HD_INLINE Scalar g_rr(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::g_11(a, r, sth, cth);
  }
  HD_INLINE Scalar g_thth(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::g_22(a, r, sth, cth);
  }
  HD_INLINE Scalar g_phph(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::g_33(a, r, sth, cth);
  }
  HD_INLINE Scalar g_rth(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  HD_INLINE Scalar g_rph(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::g_13(a, r, sth, cth);
  }
  HD_INLINE Scalar g_thph(Scalar r, Scalar sth, Scalar cth) const { return 0; }

  HD_INLINE Scalar sqrt_gamma(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::sqrt_gamma(a, r, sth, cth);
  }
  HD_INLINE Scalar alpha(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::alpha(a, r, sth, cth);
  }
  HD_INLINE Scalar beta_r(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::beta1(a, r, sth, cth);
  }
  HD_INLINE Scalar sq_gamma_beta_r(Scalar r, Scalar sth, Scalar cth) const {
    return Metric_KS::sq_gamma_beta(a, r, sth, cth);
  }
};

}  // namespace Aperture
