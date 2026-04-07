#pragma once
//
// Spherical cavity modes for vacuum Maxwell convergence testing.
//
// Provides:
//   - Spherical Bessel functions j_l(x), y_l(x) and their derivatives
//   - Real spherical harmonics Y_lm(theta, phi) and partial derivatives
//   - Eigenvalue solver for TE/TM modes between two PEC spheres at r=a, r=b
//   - Analytical TE/TM field evaluators (host-only, double precision)
//
// All routines are host-only and double precision: they are used inside
// initial-condition setters and per-step boundary applicators where speed
// is unimportant compared to accuracy.

#include "core/gpu_translation_layer.h"
#include <cmath>
#include <stdexcept>

namespace Aperture {
namespace cavity_modes {

// =========================================================================
// Spherical Bessel functions and derivatives
// =========================================================================
//
// j_l(x) and y_l(x) computed by upward recursion in l, starting from the
// closed forms for l = 0, 1. Stable for the small l (<= ~6) we need.

HD_INLINE double sph_jl(int l, double x) {
  if (l == 0) {
    if (std::abs(x) < 1e-12) return 1.0;
    return std::sin(x) / x;
  }
  if (l == 1) {
    if (std::abs(x) < 1e-12) return x / 3.0;
    return std::sin(x) / (x * x) - std::cos(x) / x;
  }
  // Upward recursion: j_{l+1}(x) = ((2l+1)/x) j_l(x) - j_{l-1}(x)
  double j_prev = std::sin(x) / x;                          // j_0
  double j_curr = std::sin(x) / (x * x) - std::cos(x) / x;  // j_1
  for (int n = 1; n < l; n++) {
    double j_next = (double(2 * n + 1) / x) * j_curr - j_prev;
    j_prev = j_curr;
    j_curr = j_next;
  }
  return j_curr;
}

HD_INLINE double sph_yl(int l, double x) {
  if (l == 0) return -std::cos(x) / x;
  if (l == 1) return -std::cos(x) / (x * x) - std::sin(x) / x;
  double y_prev = -std::cos(x) / x;
  double y_curr = -std::cos(x) / (x * x) - std::sin(x) / x;
  for (int n = 1; n < l; n++) {
    double y_next = (double(2 * n + 1) / x) * y_curr - y_prev;
    y_prev = y_curr;
    y_curr = y_next;
  }
  return y_curr;
}

// Derivative dj_l/dx using the identity
//   f_l'(x) = f_{l-1}(x) - ((l+1)/x) f_l(x)
// valid for both spherical Bessel functions.
HD_INLINE double sph_jl_prime(int l, double x) {
  if (l == 0) return -sph_jl(1, x);
  return sph_jl(l - 1, x) - (double(l + 1) / x) * sph_jl(l, x);
}

HD_INLINE double sph_yl_prime(int l, double x) {
  if (l == 0) return -sph_yl(1, x);
  return sph_yl(l - 1, x) - (double(l + 1) / x) * sph_yl(l, x);
}

// =========================================================================
// Eigenvalue solver for TE/TM modes between two PEC spheres
// =========================================================================
//
// TE modes: tangential E ∝ f(r) where f(r) = j_l(kr) + α y_l(kr).
//   PEC: f(a) = f(b) = 0.
//   Pin α = -j_l(ka)/y_l(ka) so f(a) = 0; the eigenvalue equation is
//     D_TE(k) ≡ j_l(ka) y_l(kb) - j_l(kb) y_l(ka) = 0
//
// TM modes: tangential E ∝ (rg)'(r) where g(r) = j_l(kr) + α y_l(kr).
//   PEC: (rg)'(a) = (rg)'(b) = 0.
//   Define u_l(kr) ≡ d/dr [r j_l(kr)] = j_l(kr) + kr j_l'(kr)
//          v_l(kr) ≡ d/dr [r y_l(kr)] = y_l(kr) + kr y_l'(kr)
//   Eigenvalue equation:
//     D_TM(k) ≡ u_l(ka) v_l(kb) - u_l(kb) v_l(ka) = 0

HD_INLINE double te_determinant(int l, double k, double a, double b) {
  return sph_jl(l, k * a) * sph_yl(l, k * b)
       - sph_jl(l, k * b) * sph_yl(l, k * a);
}

HD_INLINE double tm_radial_u(int l, double k, double r) {
  // u_l = j_l(kr) + kr j_l'(kr)
  double x = k * r;
  return sph_jl(l, x) + x * sph_jl_prime(l, x);
}

HD_INLINE double tm_radial_v(int l, double k, double r) {
  double x = k * r;
  return sph_yl(l, x) + x * sph_yl_prime(l, x);
}

HD_INLINE double tm_determinant(int l, double k, double a, double b) {
  return tm_radial_u(l, k, a) * tm_radial_v(l, k, b)
       - tm_radial_u(l, k, b) * tm_radial_v(l, k, a);
}

// Find the n-th positive root (n = 1 for the lowest) of D(k) = 0 over
// k ∈ (k_min, k_max). Brackets the sign change on a fine grid, then
// refines each bracket via bisection until the requested root index.
template <typename DetFunc>
inline double find_nth_root(DetFunc D, int n_root, double k_min,
                            double k_max, int n_grid = 4096,
                            double tol = 1e-12, int max_iter = 200) {
  if (n_root < 1) {
    throw std::runtime_error("cavity_modes::find_nth_root: n_root must be >= 1");
  }
  double dk = (k_max - k_min) / double(n_grid);
  double k_prev = k_min;
  double f_prev = D(k_prev);
  int found = 0;
  for (int i = 1; i <= n_grid; i++) {
    double k_curr = k_min + i * dk;
    double f_curr = D(k_curr);
    // Skip intervals containing singularities (huge value flips)
    bool sign_change = (f_prev * f_curr < 0.0)
                     && (std::abs(f_prev) < 1e10)
                     && (std::abs(f_curr) < 1e10);
    if (sign_change) {
      found++;
      if (found == n_root) {
        // Bisection on [k_prev, k_curr]
        double a = k_prev, b = k_curr;
        double fa = f_prev, fb = f_curr;
        for (int iter = 0; iter < max_iter; iter++) {
          double mid = 0.5 * (a + b);
          double fm = D(mid);
          if (std::abs(fm) < tol || (b - a) < tol * mid) return mid;
          if (fa * fm < 0.0) { b = mid; fb = fm; }
          else                { a = mid; fa = fm; }
        }
        return 0.5 * (a + b);
      }
    }
    k_prev = k_curr;
    f_prev = f_curr;
  }
  throw std::runtime_error(
      "cavity_modes::find_nth_root: requested root not found in (k_min, k_max)");
}

// Convenience wrappers — search range estimated from cavity geometry.
inline double te_eigenvalue(int l, double a, double b, int n_root) {
  // First positive root is roughly π/(b-a). Search up to ~10×.
  double dr = b - a;
  double k_min = 1e-6;
  double k_max = (n_root + 5) * M_PI / dr;
  return find_nth_root(
      [l, a, b](double k) { return te_determinant(l, k, a, b); },
      n_root, k_min, k_max);
}

inline double tm_eigenvalue(int l, double a, double b, int n_root) {
  double dr = b - a;
  double k_min = 1e-6;
  double k_max = (n_root + 5) * M_PI / dr;
  return find_nth_root(
      [l, a, b](double k) { return tm_determinant(l, k, a, b); },
      n_root, k_min, k_max);
}

// =========================================================================
// Real spherical harmonics
// =========================================================================
//
// Convention (matches scipy/SHTOOLS "real" form, Condon-Shortley phase):
//   m  > 0:  Y_lm(θ,φ) = sqrt(2) * N_l|m| * P_l^|m|(cos θ) * cos(|m| φ)
//   m  = 0:  Y_l0(θ,φ) =          N_l0  * P_l(cos θ)
//   m  < 0:  Y_lm(θ,φ) = sqrt(2) * N_l|m| * P_l^|m|(cos θ) * sin(|m| φ)
// with N_lm = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!).
//
// Returns the harmonic and its θ, φ partial derivatives in (theta, phi)
// coordinates. Pass the precomputed (cos θ, sin θ) to avoid recomputing.

// Computes P_l^m(x) and dP_l^m/dx for m ≥ 0, l ≥ m, using upward recursion
// in l from the diagonal P_m^m. Also fills out P_{l-1}^m for the derivative.
HD_INLINE void assoc_legendre(int l, int m, double x, double& Plm, double& Plm_lo) {
  // Returns Plm = P_l^m(x), and Plm_lo = P_{l-1}^m(x) (or 0 if l == m).
  // m must satisfy 0 <= m <= l.
  // Standard upward recursion (Numerical Recipes 6.7):
  //   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
  //   P_{m+1}^m(x) = x (2m+1) P_m^m(x)
  //   (l-m) P_l^m(x) = x (2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
  double Pmm = 1.0;
  if (m > 0) {
    double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
    double fact = 1.0;
    for (int i = 1; i <= m; i++) {
      Pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  if (l == m) {
    Plm = Pmm;
    Plm_lo = 0.0;
    return;
  }
  double Pmmp1 = x * double(2 * m + 1) * Pmm;
  if (l == m + 1) {
    Plm = Pmmp1;
    Plm_lo = Pmm;
    return;
  }
  double P_lm2 = Pmm;
  double P_lm1 = Pmmp1;
  double P_l = 0.0;
  for (int ll = m + 2; ll <= l; ll++) {
    P_l = (x * double(2 * ll - 1) * P_lm1 - double(ll + m - 1) * P_lm2)
        / double(ll - m);
    P_lm2 = P_lm1;
    P_lm1 = P_l;
  }
  Plm = P_l;
  Plm_lo = P_lm2;  // = P_{l-1}^m
}

// Evaluates real Y_lm and its θ, φ partial derivatives.
HD_INLINE void real_sph_harm(int l, int m, double theta, double phi,
                             double& Y, double& dY_dtheta, double& dY_dphi) {
  int am = (m >= 0) ? m : -m;
  double cos_t = std::cos(theta);
  double sin_t = std::sin(theta);

  // Normalization constant (without the sqrt(2) factor for m != 0)
  double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
  for (int k = l - am + 1; k <= l + am; k++) {
    norm /= std::sqrt(double(k));
  }

  // Associated Legendre P_l^|m| and P_{l-1}^|m| (in cos θ)
  double Plm, Plm_lo;
  assoc_legendre(l, am, cos_t, Plm, Plm_lo);

  // dP_l^m / dθ = (1/sin θ) * [l cos θ P_l^m - (l+m) P_{l-1}^m]
  // (this is a standard identity; safer than differentiating in x)
  double dPlm_dtheta;
  if (std::abs(sin_t) < 1e-14) {
    // Polar singularity: derivative is finite for l ≥ 1 but its evaluation
    // here would be 0/0. Handle by using a small ε offset, which is good
    // enough since polar quadrature points are interior to mesh edges/faces.
    double eps = 1e-10;
    double t_eps = (theta < M_PI / 2.0) ? eps : (M_PI - eps);
    double cos_e = std::cos(t_eps), sin_e = std::sin(t_eps);
    double Pe, Pe_lo;
    assoc_legendre(l, am, cos_e, Pe, Pe_lo);
    dPlm_dtheta = (double(l) * cos_e * Pe - double(l + am) * Pe_lo) / sin_e;
  } else {
    dPlm_dtheta = (double(l) * cos_t * Plm - double(l + am) * Plm_lo) / sin_t;
  }

  if (m == 0) {
    Y = norm * Plm;
    dY_dtheta = norm * dPlm_dtheta;
    dY_dphi = 0.0;
  } else if (m > 0) {
    double sqrt2 = std::sqrt(2.0);
    double cos_mp = std::cos(m * phi);
    double sin_mp = std::sin(m * phi);
    Y = sqrt2 * norm * Plm * cos_mp;
    dY_dtheta = sqrt2 * norm * dPlm_dtheta * cos_mp;
    dY_dphi = -sqrt2 * norm * Plm * double(m) * sin_mp;
  } else {  // m < 0
    double sqrt2 = std::sqrt(2.0);
    double cos_mp = std::cos(am * phi);
    double sin_mp = std::sin(am * phi);
    Y = sqrt2 * norm * Plm * sin_mp;
    dY_dtheta = sqrt2 * norm * dPlm_dtheta * sin_mp;
    dY_dphi = sqrt2 * norm * Plm * double(am) * cos_mp;
  }
}

// =========================================================================
// TE/TM mode field evaluators
// =========================================================================
//
// All field evaluators take Cartesian (x, y, z), the mode parameters
// (l, m, k, alpha), and an overall amplitude. They internally convert to
// spherical coordinates and apply the analytic formulas.
//
// Spatial patterns are evaluated. Time dependence is supplied externally
// via a (cos_phase, sin_phase) pair so the same routine handles different
// initial-phase choices and snapshots at arbitrary times.
//
// Both modes are derived from a Debye potential A = r·f(r)·Y_lm·r̂.
// Computing ∇×A directly in spherical coordinates gives the formulas below.
//
// TE mode (transverse electric, E_r = 0):
//   f(r)         = j_l(kr) + α y_l(kr)
//   E_θ          = (f / sin θ) ∂Y/∂φ
//   E_φ          = -f ∂Y/∂θ
//   B_r          = -(l(l+1)/(ω r)) f Y
//   B_θ          = -((rf)'/(ω r)) ∂Y/∂θ
//   B_φ          = -((rf)'/(ω r sin θ)) ∂Y/∂φ
//
// TM mode (B_r = 0) — same shapes with E ↔ B and an overall sign swap:
//   g(r)         = j_l(kr) + α y_l(kr)
//   B_θ          = (g / sin θ) ∂Y/∂φ
//   B_φ          = -g ∂Y/∂θ
//   E_r          = -(l(l+1)/(ω r)) g Y
//   E_θ          = -((rg)'/(ω r)) ∂Y/∂θ
//   E_φ          = -((rg)'/(ω r sin θ)) ∂Y/∂φ

struct mode_params {
  int l;
  int m;
  double k;       // wavenumber
  double alpha;   // y_l mixing coefficient (to satisfy inner BC)
  double amp;    // overall amplitude
  bool is_te;    // true: TE, false: TM
};

// Compute α so that f(a) = 0 (TE) or (rf)'(a) = 0 (TM).
inline double inner_bc_alpha(int l, double k, double a, bool is_te) {
  double x = k * a;
  if (is_te) {
    double yla = sph_yl(l, x);
    if (std::abs(yla) < 1e-30) {
      throw std::runtime_error("inner_bc_alpha: y_l(ka) ~ 0; reseed search");
    }
    return -sph_jl(l, x) / yla;
  } else {
    double u = sph_jl(l, x) + x * sph_jl_prime(l, x);
    double v = sph_yl(l, x) + x * sph_yl_prime(l, x);
    if (std::abs(v) < 1e-30) {
      throw std::runtime_error("inner_bc_alpha: v_l(ka) ~ 0; reseed search");
    }
    return -u / v;
  }
}

// Convert (Bx, By, Bz) Cartesian → (Br, Btheta, Bphi) orthonormal spherical
HD_INLINE void cartesian_to_spherical_basis(double x, double y, double z,
                                             double& r, double& theta, double& phi,
                                             double& er_x, double& er_y, double& er_z,
                                             double& et_x, double& et_y, double& et_z,
                                             double& ep_x, double& ep_y, double& ep_z) {
  r = std::sqrt(x * x + y * y + z * z);
  theta = std::acos(z / r);
  phi = std::atan2(y, x);
  double sin_t = std::sin(theta), cos_t = std::cos(theta);
  double sin_p = std::sin(phi), cos_p = std::cos(phi);
  er_x = sin_t * cos_p;  er_y = sin_t * sin_p;  er_z = cos_t;
  et_x = cos_t * cos_p;  et_y = cos_t * sin_p;  et_z = -sin_t;
  ep_x = -sin_p;          ep_y = cos_p;           ep_z = 0.0;
}

// Evaluate the (E_pat, B_pat) spatial patterns at point (x, y, z) in Cartesian.
// pattern is the spatial part; multiply by the chosen time function externally.
HD_INLINE void evaluate_mode_patterns(const mode_params& mp, double x, double y, double z,
                                       double& Ex_pat, double& Ey_pat, double& Ez_pat,
                                       double& Bx_pat, double& By_pat, double& Bz_pat) {
  double r, theta, phi;
  double er_x, er_y, er_z, et_x, et_y, et_z, ep_x, ep_y, ep_z;
  cartesian_to_spherical_basis(x, y, z, r, theta, phi,
                                er_x, er_y, er_z,
                                et_x, et_y, et_z,
                                ep_x, ep_y, ep_z);

  double Y, dY_dtheta, dY_dphi;
  real_sph_harm(mp.l, mp.m, theta, phi, Y, dY_dtheta, dY_dphi);

  double kr = mp.k * r;
  double f = sph_jl(mp.l, kr) + mp.alpha * sph_yl(mp.l, kr);
  // d/dr (r f) = f + r f', and f'(r) = k f_l'(kr)
  double fp_x = sph_jl_prime(mp.l, kr) + mp.alpha * sph_yl_prime(mp.l, kr);
  double rf_prime = f + r * mp.k * fp_x;

  double sin_t = std::sin(theta);
  double safe_inv_sin = (std::abs(sin_t) > 1e-14) ? 1.0 / sin_t : 0.0;

  // ω = c k = k (c = 1)
  double omega = mp.k;
  double l_lp1 = double(mp.l) * double(mp.l + 1);

  double Er = 0, Et = 0, Ep = 0;  // orthonormal-basis components
  double Br = 0, Bt = 0, Bp_ = 0;

  if (mp.is_te) {
    // TE: E has no radial component
    Et = f * dY_dphi * safe_inv_sin;
    Ep = -f * dY_dtheta;
    Br = -(l_lp1 / (omega * r)) * f * Y;
    Bt = -(rf_prime / (omega * r)) * dY_dtheta;
    Bp_ = -(rf_prime / (omega * r)) * dY_dphi * safe_inv_sin;
  } else {
    // TM: B has no radial component
    Bt = f * dY_dphi * safe_inv_sin;
    Bp_ = -f * dY_dtheta;
    Er = -(l_lp1 / (omega * r)) * f * Y;
    Et = -(rf_prime / (omega * r)) * dY_dtheta;
    Ep = -(rf_prime / (omega * r)) * dY_dphi * safe_inv_sin;
  }

  // Apply overall amplitude
  Er *= mp.amp;  Et *= mp.amp;  Ep *= mp.amp;
  Br *= mp.amp;  Bt *= mp.amp;  Bp_ *= mp.amp;

  // Convert from orthonormal spherical basis to Cartesian
  Ex_pat = Er * er_x + Et * et_x + Ep * ep_x;
  Ey_pat = Er * er_y + Et * et_y + Ep * ep_y;
  Ez_pat = Er * er_z + Et * et_z + Ep * ep_z;
  Bx_pat = Br * er_x + Bt * et_x + Bp_ * ep_x;
  By_pat = Br * er_y + Bt * et_y + Bp_ * ep_y;
  Bz_pat = Br * er_z + Bt * et_z + Bp_ * ep_z;
}

// Time-dependent field at time t. start_with_e selects the initial phase:
//   start_with_e == true:  E(t) = +E_pat cos(ωt),  B(t) = +B_pat sin(ωt)
//   start_with_e == false: E(t) = -E_pat sin(ωt),  B(t) = +B_pat cos(ωt)
// In both cases, E_pat and B_pat are linked by Faraday's law:
//   B_pat = -(1/ω) ∇ × E_pat (for TE)
//   E_pat = -(1/ω) ∇ × B_pat (for TM)
// matching the formulas in evaluate_mode_patterns above.
inline void evaluate_mode_at_time(const mode_params& mp, double x, double y, double z,
                                   double t, bool start_with_e,
                                   double& Ex, double& Ey, double& Ez,
                                   double& Bx, double& By, double& Bz) {
  double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
  evaluate_mode_patterns(mp, x, y, z, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
  double omega = mp.k;
  double cphase = std::cos(omega * t);
  double sphase = std::sin(omega * t);
  if (start_with_e) {
    Ex = Ex_p * cphase;  Ey = Ey_p * cphase;  Ez = Ez_p * cphase;
    Bx = Bx_p * sphase;  By = By_p * sphase;  Bz = Bz_p * sphase;
  } else {
    Ex = -Ex_p * sphase; Ey = -Ey_p * sphase; Ez = -Ez_p * sphase;
    Bx = Bx_p * cphase;  By = By_p * cphase;  Bz = Bz_p * cphase;
  }
}

}  // namespace cavity_modes
}  // namespace Aperture
