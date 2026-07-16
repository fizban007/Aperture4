#pragma once

#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
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

HD_INLINE void local_to_cartesian_impl(
    const prismatic_mesh_ptrs& mp, int tri_idx, int layer_idx,
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

HD_INLINE bool cartesian_to_local_impl(
    const prismatic_mesh_ptrs& mp, Scalar x, Scalar y, Scalar z,
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

struct GCAPushResult {
  Scalar new_x, new_y, new_z;
  Scalar u_par;   // updated parallel 4-velocity
  Scalar mu;      // magnetic moment (conserved)
  Scalar gamma;
  bool valid;
};

HD_INLINE GCAPushResult gca_push(
    Scalar old_x, Scalar old_y, Scalar old_z,
    Scalar u_par_half,  // u_par at n-1/2
    Scalar mu,           // magnetic moment
    Scalar Ex, Scalar Ey, Scalar Ez,
    Scalar Bx, Scalar By, Scalar Bz,
    Scalar q, Scalar m, Scalar dt,
    const prismatic_mesh_ptrs& mp,
    const Scalar* E_e, const Scalar* B_f,
    int tri_hint,
    bool include_curvature) {
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

  // Step 1: update u_par (Eq 17)
  Scalar u_par_new = u_par_half + (q/m) * dt * E_par;
  result.u_par = u_par_new;

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
  Scalar w2 = wEx*wEx + wEy*wEy + wEz*wEz;
  Scalar vE_factor;
  if (Scalar(4)*w2 < Scalar(0.01)) {
    vE_factor = Scalar(1) + w2;  // Taylor expansion
  } else {
    vE_factor = (Scalar(1) - std::sqrt(Scalar(1) - Scalar(4)*w2)) / (Scalar(2)*w2);
  }
  Scalar vEx = wEx * vE_factor;
  Scalar vEy = wEy * vE_factor;
  Scalar vEz = wEz * vE_factor;

  // Lorentz factor kappa for drift frame
  Scalar vE2 = vEx*vEx + vEy*vEy + vEz*vEz;
  Scalar kappa = Scalar(1) / std::sqrt(Scalar(1) - vE2);

  // Lorentz factor: Gamma = kappa * sqrt(1 + (u_par² + 2*mu*B*kappa)/m)
  // In our units with m=1: Gamma = kappa * sqrt(1 + u_par² + 2*mu*B*kappa)
  Scalar u_perp_sq = Scalar(2) * mu * B * kappa / m;
  Scalar Gamma = kappa * std::sqrt(Scalar(1) + u_par_new*u_par_new + u_perp_sq);
  result.gamma = Gamma;

  // Step 2: position update with fixed-point iteration (Eq 18)
  // R^{n+1} = R^n + dt * (u_par/Gamma * b + v_E)
  // Start with explicit Euler predict
  Scalar Rx = old_x + dt * (u_par_new / Gamma * bx + vEx);
  Scalar Ry = old_y + dt * (u_par_new / Gamma * by + vEy);
  Scalar Rz = old_z + dt * (u_par_new / Gamma * bz + vEz);

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
    Scalar nfac = (Scalar(4)*nw2 < Scalar(0.01))
        ? Scalar(1) + nw2
        : (Scalar(1) - std::sqrt(Scalar(1) - Scalar(4)*nw2)) / (Scalar(2)*nw2);
    Scalar nvEx = nwEx*nfac, nvEy = nwEy*nfac, nvEz = nwEz*nfac;

    Scalar nkappa = Scalar(1) / std::sqrt(Scalar(1) - nvEx*nvEx - nvEy*nvEy - nvEz*nvEz);
    Scalar nGamma = nkappa * std::sqrt(Scalar(1) + u_par_new*u_par_new +
                                       Scalar(2)*mu*nB*nkappa/m);

    // Average b/Gamma and v_E between old and new positions (Eq 14)
    Rx = old_x + dt * Scalar(0.5) * (
        u_par_new * (bx/Gamma + nbx/nGamma) + vEx + nvEx);
    Ry = old_y + dt * Scalar(0.5) * (
        u_par_new * (by/Gamma + nby/nGamma) + vEy + nvEy);
    Rz = old_z + dt * Scalar(0.5) * (
        u_par_new * (bz/Gamma + nbz/nGamma) + vEz + nvEz);
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

HOST_DEVICE inline void update_single_particle(
    const prismatic_mesh_ptrs& mp, int N_tri,
    prism_ptc_ptrs& ptrs, size_t n,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar q, Scalar m, Scalar dt,
    bool use_gca = false, bool include_curvature = false,
    const Scalar* Bv_rec = nullptr) {
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
    // NOTE: the GCA path below re-interpolates B internally at predicted
    // positions and still uses the primal gather there; recovery
    // currently upgrades the Boris path only.
    interpolate_B_recovery(mp, Bv_rec, tri_idx, layer_idx, l, zeta,
                           Bx, By, Bz);
  }

  Scalar old_x, old_y, old_z;
  local_to_cartesian_impl(mp, tri_idx, layer_idx, l1, l2, zeta,
                          old_x, old_y, old_z);

  Scalar new_x, new_y, new_z;
  Scalar gamma;

  if (use_gca) {
    // GCA push
    Scalar u_par = ptrs.p1[n];
    Scalar mu = ptrs.p2[n];

    auto res = gca_push(old_x, old_y, old_z, u_par, mu,
                        Ex, Ey, Ez, Bx, By, Bz,
                        q, m, dt, mp, E_e, B_f, tri_idx,
                        include_curvature);

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
    // Boris push
    Scalar px = ptrs.p1[n], py = ptrs.p2[n], pz = ptrs.p3[n];
    gamma = boris_push(px, py, pz, Ex, Ey, Ez, Bx, By, Bz, q, m, dt);
    ptrs.p1[n] = px; ptrs.p2[n] = py; ptrs.p3[n] = pz;
    ptrs.E[n] = gamma;

    Scalar vx = px/gamma, vy = py/gamma, vz = pz/gamma;
    new_x = old_x + vx*dt;
    new_y = old_y + vy*dt;
    new_z = old_z + vz*dt;
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

  // Charge density
  if (rho != nullptr) {
    Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
    deposit_rho(mp, new_tri, new_layer, l_new, new_zeta,
                q * ptrs.weight[n], rho);
  }

  // Store new position
  ptrs.x1[n] = new_l1; ptrs.x2[n] = new_l2; ptrs.x3[n] = new_zeta;
  ptrs.cell[n] = prism_cell_encode(new_tri, new_layer, N_tri);
}

// Update all particles in a loop (CPU version).
inline void update_particles_loop(
    const prismatic_mesh_ptrs& mp, int N_tri,
    prism_ptc_ptrs& ptrs, size_t num,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar charge_e, Scalar mass_e, Scalar dt,
    bool use_gca = false, bool include_curvature = false) {
  for (size_t n = 0; n < num; n++) {
    if (ptrs.cell[n] == empty_cell) continue;
    int sp = get_ptc_type(ptrs.flag[n]);
    Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
    update_single_particle(mp, N_tri, ptrs, n, E_e, B_f, J_e, rho,
                           q, mass_e, dt, use_gca, include_curvature);
  }
}

}  // namespace Aperture
