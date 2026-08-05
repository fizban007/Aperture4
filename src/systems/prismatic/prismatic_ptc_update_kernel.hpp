#pragma once

#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
#include "systems/physics/radiation_reaction.hpp"
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/prismatic_deposit.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include "systems/prismatic/prismatic_particles.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include "utils/util_functions.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Coordinate helpers
// =========================================================================

template <typename MP>
HD_INLINE void local_to_cartesian_impl(
    const MP& mp, int tri_idx, int layer_idx,
    Scalar l1, Scalar l2, Scalar zeta,
    Scalar& x, Scalar& y, Scalar& z) {
  Scalar l3 = Scalar(1) - l1 - l2;
  int v0 = mp.tri_verts[tri_idx*3], v1 = mp.tri_verts[tri_idx*3+1],
      v2 = mp.tri_verts[tri_idx*3+2];
  Scalar sx = l1*mp.sphere_vx[v0] + l2*mp.sphere_vx[v1] + l3*mp.sphere_vx[v2];
  Scalar sy = l1*mp.sphere_vy[v0] + l2*mp.sphere_vy[v1] + l3*mp.sphere_vy[v2];
  Scalar sz = l1*mp.sphere_vz[v0] + l2*mp.sphere_vz[v1] + l3*mp.sphere_vz[v2];
  Scalar s_inv = Scalar(1) / std::sqrt(sx*sx + sy*sy + sz*sz);
  sx *= s_inv; sy *= s_inv; sz *= s_inv;
  Scalar r = mp.radii[layer_idx] +
             zeta * (mp.radii[layer_idx + 1] - mp.radii[layer_idx]);
  x = r*sx; y = r*sy; z = r*sz;
}

template <typename MP>
HD_INLINE bool cartesian_to_local_impl(
    const MP& mp, Scalar x, Scalar y, Scalar z,
    int& tri_idx, int& layer_idx, Scalar& l1, Scalar& l2, Scalar& zeta,
    int tri_hint = -1) {
  Scalar r = std::sqrt(x*x + y*y + z*z);
  layer_idx = mp.find_radial_layer(r);
  if (layer_idx < 0) return false;
  zeta = mp.compute_zeta(layer_idx, r);
  Scalar ri = Scalar(1) / r;
  tri_idx = mp.find_triangle(x*ri, y*ri, z*ri, tri_hint);
  Scalar l3;
  mp.compute_barycentric(tri_idx, x*ri, y*ri, z*ri, l1, l2, l3);
  // Containment check: on the closed global sphere the walk always
  // terminates inside a triangle (λ ≥ −1e-10); a clearly-outside result
  // means the walk hit a LOCAL T_halo boundary (−1 neighbor) — i.e. the
  // particle left the halo, which the CFL contract forbids.  Treat as
  // absorption (the acceptance tests' cross-rank LIVE-count comparison
  // detects any systematic occurrence loudly).
  if (l1 < Scalar(-1e-4) || l2 < Scalar(-1e-4) || l3 < Scalar(-1e-4)) {
    return false;
  }
  return true;
}

// =========================================================================
// Boris push: full equations of motion
//
// Updates (px,py,pz) in place using E,B at particle position.
// Returns gamma after push.
// =========================================================================
HD_INLINE Scalar boris_push(
    Scalar& px, Scalar& py, Scalar& pz,
    Scalar Ex, Scalar Ey, Scalar Ez,
    Scalar Bx, Scalar By, Scalar Bz,
    Scalar q, Scalar m, Scalar dt) {
  Scalar qdt_2m = q * dt / (Scalar(2) * m);

  // Half E-kick
  px += qdt_2m * Ex; py += qdt_2m * Ey; pz += qdt_2m * Ez;

  // B rotation
  Scalar gamma_mid = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  Scalar tx = qdt_2m*Bx/gamma_mid;
  Scalar ty = qdt_2m*By/gamma_mid;
  Scalar tz = qdt_2m*Bz/gamma_mid;
  Scalar t2 = tx*tx + ty*ty + tz*tz;
  Scalar sx = Scalar(2)*tx/(Scalar(1)+t2);
  Scalar sy = Scalar(2)*ty/(Scalar(1)+t2);
  Scalar sz = Scalar(2)*tz/(Scalar(1)+t2);

  Scalar ppx = px + (py*tz - pz*ty);
  Scalar ppy = py + (pz*tx - px*tz);
  Scalar ppz = pz + (px*ty - py*tx);
  px += (ppy*sz - ppz*sy);
  py += (ppz*sx - ppx*sz);
  pz += (ppx*sy - ppy*sx);

  // Half E-kick
  px += qdt_2m * Ex; py += qdt_2m * Ey; pz += qdt_2m * Ez;

  return std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
}

// =========================================================================
// Guiding center push: zero-curvature case (Eqs 12-13 in Bacchini+2020)
//
// State: particle stores u_par in p1, mu in p2 (p3 unused for GCA).
// The flag PtcFlag::tracked is repurposed to indicate GCA mode.
//
// Position update via fixed-point iteration (Eq 18):
//   R^{n+1} = R^n + dt/2 * u_par^{n+1/2} * (b^n/Gamma^n + b^{n+1}/Gamma^{n+1})
//           + dt/2 * (v_E^n + v_E^{n+1})
//
// Parallel momentum update (Eq 17):
//   u_par^{n+1/2} = u_par^{n-1/2} + (q/m) * dt * E_par^n
// =========================================================================

// First-order GCA drifts beyond ExB, from the recovery gradient:
//   v_drift = ( m u_par^2 (b x kappa) + mu (b x grad|B|) ) / (Gamma q B)
// with kappa = (b.grad)b.  G is piecewise constant per prism (the hat
// gradient of the recovery vertex field), so both terms are first-order
// accurate.  The magnitude is clamped to 0.5c — near B nulls the drift
// expansion diverges and the hybrid switch hands the particle to Boris
// anyway.
template <typename MP>
HD_INLINE void gca_drift_velocity(
    const MP& mp, const Scalar* Bv, int tri, int layer,
    const Scalar l[3], Scalar zeta, Scalar u_par, Scalar mu,
    Scalar q, Scalar m, Scalar Gamma, Scalar v_dr[3]) {
  Scalar B[3], G[3][3];
  interpolate_B_recovery_grad(mp, Bv, tri, layer, l, zeta, B, G);
  Scalar Bmag = math::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  v_dr[0] = v_dr[1] = v_dr[2] = 0;
  if (Bmag < Scalar(1e-15)) return;
  Scalar b[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
  // (b.grad)B, then kappa = ((b.grad)B - b (b.(b.grad)B)) / |B|
  Scalar dB[3];
  for (int i = 0; i < 3; i++)
    dB[i] = G[i][0]*b[0] + G[i][1]*b[1] + G[i][2]*b[2];
  Scalar dBpar = dB[0]*b[0] + dB[1]*b[1] + dB[2]*b[2];
  Scalar kap[3];
  for (int i = 0; i < 3; i++) kap[i] = (dB[i] - b[i]*dBpar) / Bmag;
  // grad|B|_j = b_i G[i][j]
  Scalar gB[3];
  for (int j = 0; j < 3; j++)
    gB[j] = b[0]*G[0][j] + b[1]*G[1][j] + b[2]*G[2][j];
  Scalar w[3];
  for (int j = 0; j < 3; j++)
    w[j] = m * u_par * u_par * kap[j] + mu * gB[j];
  Scalar pref = Scalar(1) / (Gamma * q * Bmag);
  v_dr[0] = pref * (b[1]*w[2] - b[2]*w[1]);
  v_dr[1] = pref * (b[2]*w[0] - b[0]*w[2]);
  v_dr[2] = pref * (b[0]*w[1] - b[1]*w[0]);
  Scalar v2 = v_dr[0]*v_dr[0] + v_dr[1]*v_dr[1] + v_dr[2]*v_dr[2];
  if (v2 > Scalar(0.25)) {
    Scalar s = Scalar(0.5) / math::sqrt(v2);
    v_dr[0] *= s; v_dr[1] *= s; v_dr[2] *= s;
  }
}

// Parallel gradient of |B| from the recovery gradient: b·grad|B| with
// grad|B|_j = b_i G[i][j].  Consumed by the mirror force in the u_par
// update; same piecewise-constant first-order accuracy as the drifts.
template <typename MP>
HD_INLINE Scalar gca_grad_B_par(const MP& mp, const Scalar* Bv, int tri,
                                int layer, const Scalar l[3], Scalar zeta) {
  Scalar B[3], G[3][3];
  interpolate_B_recovery_grad(mp, Bv, tri, layer, l, zeta, B, G);
  Scalar Bmag = math::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag < Scalar(1e-15)) return Scalar(0);
  Scalar b[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
  Scalar gBpar = Scalar(0);
  for (int j = 0; j < 3; j++)
    gBpar += b[j] * (b[0]*G[0][j] + b[1]*G[1][j] + b[2]*G[2][j]);
  return gBpar;
}

// =========================================================================
// 3+1 metric terms on the particle side ("fake GR").
//
// PCTS15 (arXiv:1510.01734) Eq. 7:
//     dp/dt = alpha q (E + v x B) + alpha m gamma g + alpha H.p
//     dx/dt = alpha v - beta
// We keep the first term of each and drop the gravitational acceleration
// and the gravitomagnetic tensor, which is the paper's own stated
// approximation for strong pulsar fields.
//
// With beta = -v_LT (dec_solver_geometry.hpp) the position update is
// dx/dt = alpha v + v_LT.  Note the shift is a PURE TRANSPORT term: it
// advects every particle identically regardless of momentum, never enters
// dp/dt, and so does not touch the gyration, mu, or the adiabaticity that
// the GCA dispatch tests.  That is why it drops into the guiding-centre
// velocity without disturbing anything else.
//
// alpha and v_LT are functions of POSITION ONLY (static metric), so there
// is no new particle state, no field interpolation, and no checkpoint
// schema change.
// =========================================================================
// gr_metric_params / gr_metric_at live in dec_solver_geometry.hpp, next to
// frame_drag_shift and gr_lapse, so the field solver and the pusher read
// the sign convention from one place.

struct GCAPushResult {
  Scalar new_x, new_y, new_z;
  Scalar u_par;   // updated parallel 4-velocity
  Scalar mu;      // magnetic moment (conserved)
  Scalar gamma;
  bool valid;
};

template <typename MP>
HD_INLINE GCAPushResult gca_push(
    Scalar old_x, Scalar old_y, Scalar old_z,
    Scalar u_par_half,  // u_par at n-1/2
    Scalar mu,           // magnetic moment
    Scalar Ex, Scalar Ey, Scalar Ez,
    Scalar Bx, Scalar By, Scalar Bz,
    Scalar q, Scalar m, Scalar dt,
    const MP& mp,
    const Scalar* E_e, const Scalar* B_f,
    int tri_hint,
    bool include_curvature,
    const Scalar* Bv_rec = nullptr,
    int tri0 = -1, int layer0 = -1,
    const Scalar* l0 = nullptr, Scalar zeta0 = Scalar(0),
    gr_metric_params grp = gr_metric_params{}) {
  GCAPushResult result;
  result.mu = mu;
  result.valid = true;

  Scalar B = std::sqrt(Bx*Bx + By*By + Bz*Bz);
  if (B < Scalar(1e-15)) {
    // No B field — can't do GCA
    result.valid = false;
    return result;
  }

  // b = B/|B|
  Scalar bx = Bx/B, by = By/B, bz = Bz/B;

  // E_par = E · b
  Scalar E_par = Ex*bx + Ey*by + Ez*bz;

  // v_E = E×B drift velocity (Eq 5)
  // w_E = c E×B / (E² + B²), but in c=1 units:
  // w_E = E×B / (E² + B²)
  Scalar E2 = Ex*Ex + Ey*Ey + Ez*Ez;
  Scalar B2 = Bx*Bx + By*By + Bz*Bz;
  Scalar denom = E2 + B2;
  Scalar wEx = (Ey*Bz - Ez*By) / denom;
  Scalar wEy = (Ez*Bx - Ex*Bz) / denom;
  Scalar wEz = (Ex*By - Ey*Bx) / denom;

  // v_E = w_E/(2w_E²) * (1 - sqrt(1 - 4w_E²))
  // For w_E² << 1 (usual case), v_E ≈ w_E
  // Floors on the discriminant and on 1 - v_E²: 4w² -> 1 as E -> B and
  // in float 1 - 4w² rounds NEGATIVE once E/B is within ~1e-4 of 1 —
  // sqrt(NaN) here is exactly how t11 died when the current sheet
  // formed.  The dispatch margin (E < 0.9 B) keeps particles out of
  // this regime, but the fixed-point iteration below re-evaluates
  // fields at PREDICTED positions which can still land in E ~ B
  // territory, so both sites are floored.
  Scalar w2 = wEx*wEx + wEy*wEy + wEz*wEz;
  Scalar vE_factor;
  if (Scalar(4)*w2 < Scalar(0.01)) {
    vE_factor = Scalar(1) + w2;  // Taylor expansion
  } else {
    Scalar disc = Scalar(1) - Scalar(4)*w2;
    if (disc < Scalar(1e-6)) disc = Scalar(1e-6);
    vE_factor = (Scalar(1) - std::sqrt(disc)) / (Scalar(2)*w2);
  }
  Scalar vEx = wEx * vE_factor;
  Scalar vEy = wEy * vE_factor;
  Scalar vEz = wEz * vE_factor;

  // Lorentz factor kappa for drift frame
  Scalar vE2 = vEx*vEx + vEy*vEy + vEz*vEz;
  if (vE2 > Scalar(1) - Scalar(1e-6)) vE2 = Scalar(1) - Scalar(1e-6);
  Scalar kappa = Scalar(1) / std::sqrt(Scalar(1) - vE2);

  // Step 1: update u_par (Eq 17): parallel electric force + mirror force.
  // The mirror term follows from the Hamiltonian Gamma(u_par, mu, B):
  //   du_par/dt = -m dGamma/ds|_{u,mu} = -(mu kappa^3 / (m Gamma)) d_s|B|,
  // with Gamma evaluated at time n (u_par^{n-1/2}).  It vanishes
  // identically for the mu = 0 locked-injection population, which is why
  // its absence went unnoticed until the mu != 0 trajectory test; it is
  // what mirrors trapped particles.  d_s|B| comes from the same recovery
  // gradient as the curvature/grad-B drifts, but unlike those this term
  // is NOT optional physics, so it is gated only on gradient
  // availability, not on include_curvature.
  // GR: the whole parallel force carries the lapse (dp/dt = alpha q E_par
  // + ...), so scaling dt in the MOMENTUM update alone is exactly
  // equivalent -- the force is linear in dt.  The position update below
  // keeps the unscaled dt and applies alpha to the velocity instead,
  // which is the dx/dt = alpha v - beta half of the pair.
  // BITWISE NO-OP CONTRACT.  Every GR site below BRANCHES on gr_on and
  // keeps the flat expression verbatim in the else, rather than relying on
  // alpha = 1 / v_LT = 0 to collapse the GR form algebraically.
  //
  // The algebraic route would NOT be safe: x + 0.0 != x when x is -0.0 (it
  // returns +0.0), and -0.0 is exactly what `-w*y` yields at zero drag.  A
  // sign-of-zero reaching cartesian_to_local_impl can pick a different
  // triangle in the walk, which is a discrete flip, not a rounding
  // difference.  Branching makes the guarantee structural instead of an
  // argument about IEEE corner cases; the branch is kernel-uniform, so it
  // costs no divergence.  Pinned by "Zero-strength GR terms are a bitwise
  // no-op in the full push".
  Scalar alpha0, vlt0x, vlt0y, vlt0z;
  gr_metric_at(grp, old_x, old_y, old_z, alpha0, vlt0x, vlt0y, vlt0z);
  const bool gr_on = !gr_is_identity(grp);
  const Scalar dt_f = gr_on ? alpha0 * dt : dt;

  Scalar u_perp_sq = Scalar(2) * mu * B * kappa / m;
  Scalar u_par_new = u_par_half + (q/m) * dt_f * E_par;
  if (Bv_rec != nullptr && tri0 >= 0 &&
      l0 != nullptr && mu != Scalar(0)) {
    Scalar Gamma_n = kappa * std::sqrt(Scalar(1) + u_par_half*u_par_half +
                                       u_perp_sq);
    Scalar gBpar = gca_grad_B_par(mp, Bv_rec, tri0, layer0, l0, zeta0);
    u_par_new -= dt_f * (mu * kappa * kappa * kappa / (m * Gamma_n)) * gBpar;
  }
  result.u_par = u_par_new;

  // Lorentz factor: Gamma = kappa * sqrt(1 + (u_par² + 2*mu*B*kappa)/m)
  // In our units with m=1: Gamma = kappa * sqrt(1 + u_par² + 2*mu*B*kappa)
  Scalar Gamma = kappa * std::sqrt(Scalar(1) + u_par_new*u_par_new + u_perp_sq);
  result.gamma = Gamma;

  // Curvature + grad-B drift at the current position (recovery
  // gradient; zero when the feature is off or state unavailable).
  Scalar vdr0[3] = {0, 0, 0};
  if (include_curvature && Bv_rec != nullptr && tri0 >= 0 &&
      l0 != nullptr) {
    gca_drift_velocity(mp, Bv_rec, tri0, layer0, l0, zeta0,
                       u_par_new, mu, q, m, Gamma, vdr0);
  }

  // Step 2: position update with fixed-point iteration (Eq 18)
  // R^{n+1} = R^n + dt * (u_par/Gamma * b + v_E)
  // Start with explicit Euler predict
  // GR: dx/dt = alpha * v_gc + v_LT.  v_LT is added AFTER the guiding-centre
  // velocity is assembled and must never be folded into v_E -- kappa above
  // is the Lorentz factor of the LOCALLY MEASURED drift, and boosting it by
  // a coordinate transport term would corrupt Gamma, u_perp and the
  // cooling rate.
  Scalar Rx, Ry, Rz;
  if (gr_on) {
    Rx = old_x + dt * (alpha0 * (u_par_new / Gamma * bx + vEx + vdr0[0])
                       + vlt0x);
    Ry = old_y + dt * (alpha0 * (u_par_new / Gamma * by + vEy + vdr0[1])
                       + vlt0y);
    Rz = old_z + dt * (alpha0 * (u_par_new / Gamma * bz + vEz + vdr0[2])
                       + vlt0z);
  } else {
    Rx = old_x + dt * (u_par_new / Gamma * bx + vEx + vdr0[0]);
    Ry = old_y + dt * (u_par_new / Gamma * by + vEy + vdr0[1]);
    Rz = old_z + dt * (u_par_new / Gamma * bz + vEz + vdr0[2]);
  }

  // Fixed-point iterations: evaluate b and v_E at the new position
  for (int iter = 0; iter < 3; iter++) {
    // Find prism at predicted position and interpolate fields
    int new_tri, new_layer;
    Scalar nl1, nl2, nzeta;
    if (!cartesian_to_local_impl(mp, Rx, Ry, Rz,
                                 new_tri, new_layer, nl1, nl2, nzeta,
                                 tri_hint)) {
      result.valid = false;
      return result;
    }

    Scalar nl[3] = {nl1, nl2, Scalar(1)-nl1-nl2};
    Scalar nEx, nEy, nEz, nBx, nBy, nBz;
    interpolate_fields(mp, new_tri, new_layer, nl, nzeta,
                       E_e, B_f, nEx, nEy, nEz, nBx, nBy, nBz);
    if (Bv_rec != nullptr) {
      // Second-order C0 recovery gather for B at the predicted
      // position — the primal Whitney gather's O(h) face jumps
      // random-walk guiding centers exactly like they scatter Boris
      // particles (the A1 result); GCA is not exempt.
      interpolate_B_recovery(mp, Bv_rec, new_tri, new_layer, nl, nzeta,
                             nBx, nBy, nBz);
    }

    Scalar nB = std::sqrt(nBx*nBx + nBy*nBy + nBz*nBz);
    if (nB < Scalar(1e-15)) { result.valid = false; return result; }

    Scalar nbx = nBx/nB, nby = nBy/nB, nbz = nBz/nB;

    // Recompute v_E at new position
    Scalar nE2 = nEx*nEx + nEy*nEy + nEz*nEz;
    Scalar nB2 = nBx*nBx + nBy*nBy + nBz*nBz;
    Scalar nd = nE2 + nB2;
    Scalar nwEx = (nEy*nBz - nEz*nBy) / nd;
    Scalar nwEy = (nEz*nBx - nEx*nBz) / nd;
    Scalar nwEz = (nEx*nBy - nEy*nBx) / nd;
    Scalar nw2 = nwEx*nwEx + nwEy*nwEy + nwEz*nwEz;
    Scalar ndisc = Scalar(1) - Scalar(4)*nw2;
    if (ndisc < Scalar(1e-6)) ndisc = Scalar(1e-6);
    Scalar nfac = (Scalar(4)*nw2 < Scalar(0.01))
        ? Scalar(1) + nw2
        : (Scalar(1) - std::sqrt(ndisc)) / (Scalar(2)*nw2);
    Scalar nvEx = nwEx*nfac, nvEy = nwEy*nfac, nvEz = nwEz*nfac;

    Scalar nvE2 = nvEx*nvEx + nvEy*nvEy + nvEz*nvEz;
    if (nvE2 > Scalar(1) - Scalar(1e-6)) nvE2 = Scalar(1) - Scalar(1e-6);
    Scalar nkappa = Scalar(1) / std::sqrt(Scalar(1) - nvE2);
    Scalar nGamma = nkappa * std::sqrt(Scalar(1) + u_par_new*u_par_new +
                                       Scalar(2)*mu*nB*nkappa/m);

    Scalar nvdr[3] = {0, 0, 0};
    if (include_curvature && Bv_rec != nullptr) {
      gca_drift_velocity(mp, Bv_rec, new_tri, new_layer, nl, nzeta,
                         u_par_new, mu, q, m, nGamma, nvdr);
    }

    // Average b/Gamma, v_E, and the drifts between old/new (Eq 14).  The
    // metric terms ride the same trapezoid: alpha weights each endpoint's
    // guiding-centre velocity, and v_LT (a velocity in its own right) is
    // averaged directly.
    //
    // Branch, do not merge: see the bitwise no-op contract above.
    if (gr_on) {
      Scalar alpha1, vlt1x, vlt1y, vlt1z;
      gr_metric_at(grp, Rx, Ry, Rz, alpha1, vlt1x, vlt1y, vlt1z);
      Rx = old_x + dt * Scalar(0.5) * (
          u_par_new * (alpha0*bx/Gamma + alpha1*nbx/nGamma) +
          alpha0*vEx + alpha1*nvEx + alpha0*vdr0[0] + alpha1*nvdr[0] +
          vlt0x + vlt1x);
      Ry = old_y + dt * Scalar(0.5) * (
          u_par_new * (alpha0*by/Gamma + alpha1*nby/nGamma) +
          alpha0*vEy + alpha1*nvEy + alpha0*vdr0[1] + alpha1*nvdr[1] +
          vlt0y + vlt1y);
      Rz = old_z + dt * Scalar(0.5) * (
          u_par_new * (alpha0*bz/Gamma + alpha1*nbz/nGamma) +
          alpha0*vEz + alpha1*nvEz + alpha0*vdr0[2] + alpha1*nvdr[2] +
          vlt0z + vlt1z);
    } else {
      Rx = old_x + dt * Scalar(0.5) * (
          u_par_new * (bx/Gamma + nbx/nGamma) + vEx + nvEx + vdr0[0] + nvdr[0]);
      Ry = old_y + dt * Scalar(0.5) * (
          u_par_new * (by/Gamma + nby/nGamma) + vEy + nvEy + vdr0[1] + nvdr[1]);
      Rz = old_z + dt * Scalar(0.5) * (
          u_par_new * (bz/Gamma + nbz/nGamma) + vEz + nvEz + vdr0[2] + nvdr[2]);
    }
  }

  result.new_x = Rx; result.new_y = Ry; result.new_z = Rz;
  return result;
}

// =========================================================================
// Single-particle update: dispatches to Boris or GCA
//
// Particle storage convention:
//   For Boris: p1,p2,p3 = Cartesian momentum (px,py,pz)
//   For GCA:   p1 = u_par (parallel 4-velocity), p2 = mu (magnetic moment)
//              p3 = u_perp_g (perpendicular gyration speed, for reconstruction)
//
// The particle flag bit PtcFlag::tracked indicates GCA mode.
// =========================================================================

template <typename MP>
HOST_DEVICE inline void update_single_particle(
    const MP& mp, int N_tri,
    prism_ptc_ptrs& ptrs, size_t n,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar q, Scalar m, Scalar dt,
    bool use_gca = false, bool include_curvature = false,
    const Scalar* Bv_rec = nullptr, Scalar absorb_r = Scalar(0),
    Scalar* rho_abs = nullptr, Scalar* gamma_wsum = nullptr,
    Scalar gca_switch_omegac = Scalar(20), bool zero_mu_on_capture = false,
    Scalar sync_cool_coef = Scalar(0),
    gr_metric_params grp = gr_metric_params{}) {
  int tri_idx, layer_idx;
  prism_cell_decode(ptrs.cell[n], N_tri, tri_idx, layer_idx);
  Scalar l1 = ptrs.x1[n], l2 = ptrs.x2[n];
  Scalar l3 = Scalar(1) - l1 - l2;
  Scalar zeta = ptrs.x3[n];

  // Interpolate E, B at particle position.  E always uses the primal
  // Whitney 1-forms (the adjoint partner of the charge-conserving
  // deposit); B optionally uses the C0 second-order vertex recovery
  // (prismatic_vertex_recovery.h) when Bv_rec is provided.
  Scalar l[3] = {l1, l2, l3};
  Scalar Ex, Ey, Ez, Bx, By, Bz;
  interpolate_fields(mp, tri_idx, layer_idx, l, zeta,
                     E_e, B_f, Ex, Ey, Ez, Bx, By, Bz);
  if (Bv_rec != nullptr) {
    interpolate_B_recovery(mp, Bv_rec, tri_idx, layer_idx, l, zeta,
                           Bx, By, Bz);
  }

  Scalar old_x, old_y, old_z;
  local_to_cartesian_impl(mp, tri_idx, layer_idx, l1, l2, zeta,
                          old_x, old_y, old_z);

  // Per-particle hybrid dispatch (master switch use_gca): a particle is
  // pushed by GCA while its gyration is under-resolved and the drift
  // frame exists, and handed to Boris "at the last minute" — near B
  // nulls (current sheet) where the gyro-frequency drops below
  // gca_switch_omegac or E exceeds B.  The momentum slots are converted
  // at each transition; the gca_state flag records the representation.
  // With mu = 0 (synchrotron-locked injection) both conversions are
  // exact: the momentum is u_par b in both representations.
  //
  // The criterion is the RATE omega_c/gamma against a rate threshold --
  // dt appears nowhere.  It used to: the test was omega_c dt / gamma >
  // 0.1, which made the PHYSICAL switching surface a function of the
  // step.  "0.1 at every level" silently meant a different surface at
  // every level (L5 10.2, L6 20.4, L7 40.7 in these units), and the
  // measured L7 consequences -- Y-point jitter 21.3 deg vs L6's 12.0,
  // +22% open flux, a spurious P/2 shedding cycle -- were largely that
  // artifact.  Holding the RATE fixed is what the L7 gca005 A/B branch
  // did by hand (0.05 at L7 dt == 0.1 at L6 dt == 20.37), and it
  // restored the L6 surface.  Keep dt out of here: the dispatch must be
  // bit-identical at dt and dt/2 (tests/test_prismatic_pusher.cpp).
  bool do_gca = false;
  if (use_gca) {
    Scalar B2l = Bx*Bx + By*By + Bz*Bz;
    Scalar E2l = Ex*Ex + Ey*Ey + Ez*Ez;
    Scalar Bmag = math::sqrt(B2l);
    Scalar gam_prev = ptrs.E[n] > Scalar(1) ? ptrs.E[n] : Scalar(1);
    Scalar wc = math::abs(q / m) * Bmag / gam_prev;
    // Margin below E = B: the GCA drift frame degenerates (kappa ->
    // inf; 1 - 4w² rounds float-negative -> NaN) as E -> B.  t11
    // (threshold 0.05) crashed exactly this way when the current sheet
    // formed.  Requiring E < 0.9 B hands near-degenerate particles to
    // Boris, which is the physically correct pusher there anyway.
    do_gca = (wc > gca_switch_omegac) && (B2l * Scalar(0.81) > E2l) &&
             (Bmag > Scalar(1e-15));
    bool was_gca = check_flag(ptrs.flag[n], PtcFlagEx::gca_state);
    if (was_gca && !do_gca) {
      // GCA -> Boris: reconstruct momentum.  The gyrophase is not carried
      // by the GCA representation, so it has to be INVENTED here -- and it
      // must be invented ISOTROPICALLY.  This used to place all of u_perp
      // along e1 = b x a_hat with a_hat a fixed LAB axis, which gives every
      // converting particle the same perpendicular direction regardless of
      // charge: a net bulk momentum injection rather than a current.  In the
      // equatorial plane beyond the light cylinder b lies in-plane, so that
      // e1 is vertical and the kick pushes plasma alternately out of and
      // into the current sheet -- the m=1 pile-up / one-sided sheet seen in
      // the aligned-rotator runs (2026-08-03).  mu = 0 at injection and the
      // GCA push conserves it, so the defect only bites after a
      // Boris -> GCA recapture regenerates mu, i.e. hardest in the sheet.
      //
      // The phase is hashed from the particle id rather than drawn from the
      // rng state pool, for two reasons: the pool is indexed by thread, so a
      // draw would depend on thread scheduling and break the bitwise
      // partition-invariance the solver relies on; and an id hash is
      // dt-invariant, which the dispatch test requires.  The cost is that a
      // given particle always converts at the same phase -- harmless, since
      // what matters is that the phase is uncorrelated ACROSS particles.
      Scalar b0[3] = {Bx / Bmag, By / Bmag, Bz / Bmag};
      Scalar u_par = ptrs.p1[n];
      Scalar up2 = Scalar(2) * ptrs.p2[n] * Bmag / m;
      Scalar u_perp = math::sqrt(up2 > Scalar(0) ? up2 : Scalar(0));
      // Orthonormal triad (e1, e2, b0).  With an isotropic phase the choice
      // of reference axis only fixes where psi = 0, so it no longer matters;
      // the branch is kept solely to stay clear of b0 || a_hat.
      Scalar ax = (math::abs(b0[0]) < Scalar(0.9)) ? Scalar(1) : Scalar(0);
      Scalar ay = Scalar(1) - ax;
      Scalar e1[3] = {-b0[2]*ay, b0[2]*ax, b0[0]*ay - b0[1]*ax};
      Scalar en = math::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
      if (en > Scalar(1e-15)) {
        e1[0] /= en; e1[1] /= en; e1[2] /= en;
      } else {
        e1[0] = Scalar(1); e1[1] = Scalar(0); e1[2] = Scalar(0);
      }
      Scalar e2[3] = {b0[1]*e1[2] - b0[2]*e1[1],
                      b0[2]*e1[0] - b0[0]*e1[2],
                      b0[0]*e1[1] - b0[1]*e1[0]};
      // splitmix64 on the particle id -> uniform phase in [0, 2pi).
      uint64_t h = ptrs.id[n] + 0x9e3779b97f4a7c15ull;
      h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
      h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
      h = h ^ (h >> 31);
      Scalar psi = Scalar(6.283185307179586) *
                   (Scalar)(h >> 11) * Scalar(1.0 / 9007199254740992.0);
      Scalar w1 = u_perp * math::cos(psi), w2 = u_perp * math::sin(psi);
      ptrs.p1[n] = u_par * b0[0] + w1 * e1[0] + w2 * e2[0];
      ptrs.p2[n] = u_par * b0[1] + w1 * e1[1] + w2 * e2[1];
      ptrs.p3[n] = u_par * b0[2] + w1 * e1[2] + w2 * e2[2];
      clear_flag(ptrs.flag[n], PtcFlagEx::gca_state);
    } else if (!was_gca && do_gca) {
      // Boris -> GCA: project onto b; remainder becomes mu (or is
      // radiated instantly under the synchrotron-locking option).
      Scalar b0[3] = {Bx / Bmag, By / Bmag, Bz / Bmag};
      Scalar px = ptrs.p1[n], py = ptrs.p2[n], pz = ptrs.p3[n];
      Scalar u_par = px*b0[0] + py*b0[1] + pz*b0[2];
      Scalar u_perp_sq = px*px + py*py + pz*pz - u_par*u_par;
      if (u_perp_sq < Scalar(0)) u_perp_sq = Scalar(0);
      ptrs.p1[n] = u_par;
      ptrs.p2[n] = zero_mu_on_capture
                       ? Scalar(0)
                       : m * u_perp_sq / (Scalar(2) * Bmag);
      ptrs.p3[n] = zero_mu_on_capture ? Scalar(0)
                                      : math::sqrt(u_perp_sq);
      set_flag(ptrs.flag[n], PtcFlagEx::gca_state);
    }
  }

  Scalar new_x, new_y, new_z;
  Scalar gamma;

  if (do_gca) {
    // GCA push
    Scalar u_par = ptrs.p1[n];
    Scalar mu = ptrs.p2[n];

    Scalar l_old[3] = {l1, l2, l3};
    auto res = gca_push(old_x, old_y, old_z, u_par, mu,
                        Ex, Ey, Ez, Bx, By, Bz,
                        q, m, dt, mp, E_e, B_f, tri_idx,
                        include_curvature, Bv_rec,
                        tri_idx, layer_idx, l_old, zeta, grp);

    if (!res.valid) {
      ptrs.cell[n] = empty_cell;
      return;
    }

    new_x = res.new_x; new_y = res.new_y; new_z = res.new_z;
    gamma = res.gamma;
    ptrs.p1[n] = res.u_par;
    ptrs.p2[n] = res.mu;
    ptrs.E[n] = gamma;
  } else {
    // Boris push.  GR: dp/dt = alpha q (E + v x B), and the force is
    // linear in dt, so integrating over alpha*dt is exactly equivalent and
    // leaves the Boris rotation itself untouched.  The synchrotron drag is
    // a force too and carries the same factor.
    Scalar alpha_b, vltx, vlty, vltz;
    gr_metric_at(grp, old_x, old_y, old_z, alpha_b, vltx, vlty, vltz);
    const bool gr_on = !gr_is_identity(grp);
    const Scalar dt_f = gr_on ? alpha_b * dt : dt;

    Scalar px = ptrs.p1[n], py = ptrs.p2[n], pz = ptrs.p3[n];
    gamma = boris_push(px, py, pz, Ex, Ey, Ez, Bx, By, Bz, q, m, dt_f);

    // Synchrotron drag, operator-split onto the Lorentz force above.
    // This is what makes the hybrid switch physical: the GCA side holds
    // mu = 0 (synchrotron-locked) while the Boris side had NO radiative
    // drag at all, so the two branches ran different physics and the
    // switching surface -- which moves with dt -- left a footprint on the
    // solution.  With the drag on, cooled-Boris relaxes onto the mu = 0
    // GCA state wherever both are valid, and the surface stops mattering.
    //
    // The Landau-Lifshitz force (systems/physics/radiation_reaction.hpp,
    // shared with pusher_synchrotron) preserves the pitch angle at high
    // gamma rather than driving it to zero, and needs no E >= B or
    // B -> 0 guard: it falls smoothly to zero with the fields, so the
    // hot kinetic core inside the sheet stays uncooled on its own.
    // The GCA path is deliberately untouched (mu = 0 there already; a
    // parallel-only particle radiates by CURVATURE, out of scope here).
    if (sync_cool_coef > Scalar(0) &&
        !check_flag(ptrs.flag[n], PtcFlag::ignore_radiation)) {
      sync_drag_substep(px, py, pz, gamma, Ex, Ey, Ez, Bx, By, Bz,
                        sync_cool_coef, dt_f);
    }

    ptrs.p1[n] = px; ptrs.p2[n] = py; ptrs.p3[n] = pz;
    ptrs.E[n] = gamma;

    // dx/dt = alpha v + v_LT  (= alpha v - beta).  Branch, do not merge:
    // see the bitwise no-op contract in gca_push.
    Scalar vx = px/gamma, vy = py/gamma, vz = pz/gamma;
    if (gr_on) {
      new_x = old_x + (alpha_b*vx + vltx)*dt;
      new_y = old_y + (alpha_b*vy + vlty)*dt;
      new_z = old_z + (alpha_b*vz + vltz)*dt;
    } else {
      new_x = old_x + vx*dt;
      new_y = old_y + vy*dt;
      new_z = old_z + vz*dt;
    }
  }

  // Boundary absorption.  Particles leaving the radial domain are
  // absorbed implicitly: find_radial_layer returns -1 outside
  // [r_min, r_max] and the cartesian_to_local conversion below fails.
  // absorb_r > 0 additionally absorbs at a configurable radius (the
  // damping-layer entrance in magnetosphere runs — particles must not
  // stream through the absorber where the fields are unphysical).
  // Note the final partial-step current is NOT deposited for absorbed
  // particles; the resulting Gauss-law residual sits on boundary cells
  // that the inner BC / damping layer own anyway.
  if (absorb_r > Scalar(0) &&
      new_x*new_x + new_y*new_y + new_z*new_z > absorb_r*absorb_r) {
    ptrs.cell[n] = empty_cell;
    return;
  }

  // Convert new position to local coordinates
  int new_tri, new_layer;
  Scalar new_l1, new_l2, new_zeta;
  if (!cartesian_to_local_impl(mp, new_x, new_y, new_z,
                               new_tri, new_layer, new_l1, new_l2,
                               new_zeta, tri_idx)) {
    ptrs.cell[n] = empty_cell;
    return;
  }

  // Current deposition
  if (!check_flag(ptrs.flag[n], PtcFlag::ignore_current) && J_e != nullptr) {
    Scalar l_old[3] = {l1, l2, l3};
    Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
    Scalar zeta_new_in_old;
    if (new_tri == tri_idx && new_layer == layer_idx) {
      zeta_new_in_old = new_zeta;
    } else {
      Scalar r_new = std::sqrt(new_x*new_x + new_y*new_y + new_z*new_z);
      Scalar dr_old = mp.radii[layer_idx+1] - mp.radii[layer_idx];
      zeta_new_in_old = (r_new - mp.radii[layer_idx]) / dr_old;
      Scalar ri = Scalar(1) / r_new;
      mp.compute_barycentric(tri_idx, new_x*ri, new_y*ri, new_z*ri,
                             l_new[0], l_new[1], l_new[2]);
    }
    int dep_tri, dep_layer;
    deposit_current(mp, tri_idx, layer_idx, l_old, zeta,
                    l_new, zeta_new_in_old,
                    q * ptrs.weight[n] / dt, J_e, dep_tri, dep_layer);
  }

  // Charge density; optionally |charge| density and gamma-weighted
  // |charge| density (multiplicity + mean-Lorentz-factor diagnostics).
  {
    Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
    if (rho != nullptr) {
      deposit_rho(mp, new_tri, new_layer, l_new, new_zeta,
                  q * ptrs.weight[n], rho);
    }
    Scalar aqw = math::abs(q) * ptrs.weight[n];
    if (rho_abs != nullptr) {
      deposit_rho(mp, new_tri, new_layer, l_new, new_zeta, aqw, rho_abs);
    }
    if (gamma_wsum != nullptr) {
      deposit_rho(mp, new_tri, new_layer, l_new, new_zeta, gamma * aqw,
                  gamma_wsum);
    }
  }

  // Store new position
  ptrs.x1[n] = new_l1; ptrs.x2[n] = new_l2; ptrs.x3[n] = new_zeta;
  ptrs.cell[n] = prism_cell_encode(new_tri, new_layer, N_tri);
}

// Update all particles in a loop (CPU version).
template <typename MP>
inline void update_particles_loop(
    const MP& mp, int N_tri,
    prism_ptc_ptrs& ptrs, size_t num,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar charge_e, Scalar mass_e, Scalar dt,
    bool use_gca = false, bool include_curvature = false,
    Scalar absorb_r = Scalar(0), Scalar sync_cool_coef = Scalar(0)) {
  for (size_t n = 0; n < num; n++) {
    if (ptrs.cell[n] == empty_cell) continue;
    int sp = get_ptc_type(ptrs.flag[n]);
    Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
    update_single_particle(mp, N_tri, ptrs, n, E_e, B_f, J_e, rho,
                           q, mass_e, dt, use_gca, include_curvature,
                           nullptr, absorb_r, nullptr, nullptr,
                           Scalar(20), false, sync_cool_coef);
  }
}

}  // namespace Aperture
