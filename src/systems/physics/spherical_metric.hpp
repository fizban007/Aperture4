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
// The user provides γ_{ij}(r, θ) plus the 3+1 lapse α and shift β^r.
// All functions take (r, sinθ, cosθ) following the convention in
// metric_kerr_schild.hpp.
//
// Axisymmetry is assumed: γ_{rθ} = γ_{θφ} = 0, β^θ = β^φ = 0.
// Override g_rth(), g_thph() for non-axisymmetric metrics.

#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/physics/metric_kerr_schild.hpp"
#include <cmath>

namespace Aperture {

struct spherical_metric_t {
  virtual ~spherical_metric_t() = default;

  // --- Spatial 3-metric diagonal components ---
  virtual Scalar g_rr(Scalar r, Scalar sth, Scalar cth) const = 0;
  virtual Scalar g_thth(Scalar r, Scalar sth, Scalar cth) const = 0;
  virtual Scalar g_phph(Scalar r, Scalar sth, Scalar cth) const = 0;

  // --- Off-diagonal (default zero for axisymmetric metrics) ---
  virtual Scalar g_rth(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  virtual Scalar g_rph(Scalar r, Scalar sth, Scalar cth) const { return 0; }
  virtual Scalar g_thph(Scalar r, Scalar sth, Scalar cth) const { return 0; }

  // --- Determinant ---
  virtual Scalar sqrt_gamma(Scalar r, Scalar sth, Scalar cth) const = 0;

  // --- 3+1 lapse (default 1 = no gravity) ---
  virtual Scalar alpha(Scalar r, Scalar sth, Scalar cth) const { return 1; }

  // --- 3+1 shift r-component (default 0) ---
  virtual Scalar beta_r(Scalar r, Scalar sth, Scalar cth) const { return 0; }

  // --- Composite: sqrt(gamma) * beta^r (used in shift cross terms) ---
  virtual Scalar sq_gamma_beta_r(Scalar r, Scalar sth, Scalar cth) const {
    return sqrt_gamma(r, sth, cth) * beta_r(r, sth, cth);
  }
};

// =========================================================================
// Flat space in spherical coordinates:  ds² = dr² + r²dθ² + r²sin²θ dφ²
// =========================================================================
struct flat_spherical_metric : spherical_metric_t {
  Scalar g_rr(Scalar r, Scalar sth, Scalar cth) const override {
    return 1;
  }
  Scalar g_thth(Scalar r, Scalar sth, Scalar cth) const override {
    return r * r;
  }
  Scalar g_phph(Scalar r, Scalar sth, Scalar cth) const override {
    return r * r * sth * sth;
  }
  Scalar sqrt_gamma(Scalar r, Scalar sth, Scalar cth) const override {
    return r * r * sth;
  }
};

// =========================================================================
// Kerr-Schild in spherical KS coordinates.
// Wraps the existing Metric_KS functions from metric_kerr_schild.hpp.
// =========================================================================
struct ks_spherical_metric : spherical_metric_t {
  Scalar a;  // spin parameter

  explicit ks_spherical_metric(Scalar a_) : a(a_) {}

  Scalar g_rr(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::g_11(a, r, sth, cth);
  }
  Scalar g_thth(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::g_22(a, r, sth, cth);
  }
  Scalar g_phph(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::g_33(a, r, sth, cth);
  }
  Scalar g_rph(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::g_13(a, r, sth, cth);
  }
  Scalar sqrt_gamma(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::sqrt_gamma(a, r, sth, cth);
  }
  Scalar alpha(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::alpha(a, r, sth, cth);
  }
  Scalar beta_r(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::beta1(a, r, sth, cth);
  }
  Scalar sq_gamma_beta_r(Scalar r, Scalar sth, Scalar cth) const override {
    return Metric_KS::sq_gamma_beta(a, r, sth, cth);
  }
};

}  // namespace Aperture
