#pragma once

#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include "systems/prismatic/prismatic_particles.h"
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
// Single-particle update kernel: Boris push + move + deposit
//
// Pure computation on raw arrays — no framework dependency.
// Can be called from tests or from the system_t::update().
// =========================================================================

HOST_DEVICE inline void update_single_particle(
    const prismatic_mesh_ptrs& mp, int N_r,
    prism_ptc_ptrs& ptrs, size_t n,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar q, Scalar m, Scalar dt) {
  int tri_idx, layer_idx;
  prism_cell_decode(ptrs.cell[n], N_r, tri_idx, layer_idx);
  Scalar l1 = ptrs.x1[n], l2 = ptrs.x2[n];
  Scalar l3 = Scalar(1) - l1 - l2;
  Scalar zeta = ptrs.x3[n];

  // 1. Interpolate E, B
  Scalar l[3] = {l1, l2, l3};
  Scalar Ex, Ey, Ez, Bx, By, Bz;
  interpolate_fields(mp, tri_idx, layer_idx, l, zeta,
                     E_e, B_f, Ex, Ey, Ez, Bx, By, Bz);

  // 2. Boris push
  Scalar qdt_2m = q * dt / (Scalar(2) * m);
  Scalar px = ptrs.p1[n] + qdt_2m*Ex;
  Scalar py = ptrs.p2[n] + qdt_2m*Ey;
  Scalar pz = ptrs.p3[n] + qdt_2m*Ez;

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

  px += qdt_2m*Ex; py += qdt_2m*Ey; pz += qdt_2m*Ez;

  ptrs.p1[n] = px; ptrs.p2[n] = py; ptrs.p3[n] = pz;
  Scalar gamma = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.E[n] = gamma;

  // 3. Position update
  Scalar old_x, old_y, old_z;
  local_to_cartesian_impl(mp, tri_idx, layer_idx, l1, l2, zeta,
                          old_x, old_y, old_z);
  Scalar vx = px/gamma, vy = py/gamma, vz = pz/gamma;
  Scalar new_x = old_x + vx*dt;
  Scalar new_y = old_y + vy*dt;
  Scalar new_z = old_z + vz*dt;

  int new_tri, new_layer;
  Scalar new_l1, new_l2, new_zeta;
  if (!cartesian_to_local_impl(mp, new_x, new_y, new_z,
                               new_tri, new_layer, new_l1, new_l2,
                               new_zeta, tri_idx)) {
    ptrs.cell[n] = empty_cell;
    return;
  }

  // 4. Current deposition
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

  // 5. Charge density
  if (rho != nullptr) {
    Scalar l_new[3] = {new_l1, new_l2, Scalar(1) - new_l1 - new_l2};
    deposit_rho(mp, new_tri, new_layer, l_new, new_zeta,
                q * ptrs.weight[n], rho);
  }

  // 6. Store new position
  ptrs.x1[n] = new_l1; ptrs.x2[n] = new_l2; ptrs.x3[n] = new_zeta;
  ptrs.cell[n] = prism_cell_encode(new_tri, new_layer, N_r);
}

// Update all particles in a loop (CPU version).
inline void update_particles_loop(
    const prismatic_mesh_ptrs& mp, int N_r,
    prism_ptc_ptrs& ptrs, size_t num,
    const Scalar* E_e, const Scalar* B_f,
    Scalar* J_e, Scalar* rho,
    Scalar charge_e, Scalar mass_e, Scalar dt) {
  for (size_t n = 0; n < num; n++) {
    if (ptrs.cell[n] == empty_cell) continue;
    int sp = get_ptc_type(ptrs.flag[n]);
    Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
    update_single_particle(mp, N_r, ptrs, n, E_e, B_f, J_e, rho,
                           q, mass_e, dt);
  }
}

}  // namespace Aperture
