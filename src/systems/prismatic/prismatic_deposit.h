#pragma once

#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include <cmath>

namespace Aperture {

// Accumulate value into J buffer.  Uses atomicAdd on GPU to handle
// concurrent writes from multiple threads.
template <typename FloatT>
HD_INLINE void atomic_add_scalar(FloatT* addr, FloatT val) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(addr, val);
#else
  *addr += val;
#endif
}

// =========================================================================
// Charge density deposit (Whitney 0-form)
//
// rho_v = q * weight * W^0_v(x) = q * weight * lambda_i * phi_k(zeta)
// Deposits to the 6 vertices of the containing prism.
// =========================================================================
template <typename FloatT>
HD_INLINE void deposit_rho(
    const prismatic_mesh_ptrs& mesh,
    int tri_idx, int layer_idx,
    const FloatT l[3], FloatT zeta,
    FloatT q_weight,
    FloatT* rho) {
  FloatT phi_bot = FloatT(1) - zeta;  // bottom shell
  FloatT phi_top = zeta;               // top shell

  for (int i = 0; i < 3; i++) {
    int sv = mesh.tri_verts[tri_idx * 3 + i];
    // Bottom vertex (shell = layer_idx)
    int v_bot = layer_idx * mesh.N_vert_s + sv;
    atomic_add_scalar(&rho[v_bot], q_weight * l[i] * phi_bot);
    // Top vertex (shell = layer_idx + 1)
    int v_top = (layer_idx + 1) * mesh.N_vert_s + sv;
    atomic_add_scalar(&rho[v_top], q_weight * l[i] * phi_top);
  }
}

// =========================================================================
// Single-prism current deposition (Whitney 1-form path integrals)
// =========================================================================
template <typename FloatT>
HOST_DEVICE void deposit_current_single_prism(
    const prismatic_mesh_ptrs& mesh,
    int tri_idx, int layer_idx,
    const FloatT l_old[3], FloatT zeta_old,
    const FloatT l_new[3], FloatT zeta_new,
    FloatT q_over_dt,
    FloatT* J) {
  FloatT dl[3] = {l_new[0] - l_old[0],
                  l_new[1] - l_old[1],
                  l_new[2] - l_old[2]};
  FloatT dz = zeta_new - zeta_old;

  FloatT phi_bar_0 = FloatT(0.5) * ((FloatT(1.0) - zeta_old) +
                                    (FloatT(1.0) - zeta_new));
  FloatT phi_bar_1 = FloatT(0.5) * (zeta_old + zeta_new);

  const int circuit_from[3] = {0, 1, 2};
  const int circuit_to[3]   = {1, 2, 0};

  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j];
    int i_to = circuit_to[j];
    FloatT A = l_old[i_from] * dl[i_to] - l_old[i_to] * dl[i_from];
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];
    int sphere_e = mesh.tri_edges_s[tri_idx * 3 + j];

    atomic_add_scalar(&J[mesh.h_edge_idx(layer_idx, sphere_e)],
                      q_over_dt * FloatT(sign) * A * phi_bar_0);
    atomic_add_scalar(&J[mesh.h_edge_idx(layer_idx + 1, sphere_e)],
                      q_over_dt * FloatT(sign) * A * phi_bar_1);
  }

  for (int i = 0; i < 3; i++) {
    int sphere_v = mesh.tri_verts[tri_idx * 3 + i];
    FloatT avg_lambda = FloatT(0.5) * (l_old[i] + l_new[i]);
    atomic_add_scalar(&J[mesh.v_edge_idx(layer_idx, sphere_v)],
                      q_over_dt * dz * avg_lambda);
  }
}

// =========================================================================
// Crossing detection
// =========================================================================
template <typename FloatT>
HD_INLINE int detect_crossing(
    const FloatT l_old[3], FloatT zeta_old,
    const FloatT l_new[3], FloatT zeta_new,
    FloatT& crossing_s, int& cross_idx) {
  crossing_s = FloatT(1.0);
  cross_idx = 0;
  int cross_type = 0;

  FloatT dz = zeta_new - zeta_old;
  if (zeta_new < FloatT(0.0) && dz != FloatT(0.0)) {
    FloatT s = -zeta_old / dz;
    if (s >= FloatT(0.0) && s < crossing_s) {
      crossing_s = s;
      cross_idx = -1;
      cross_type = 1;
    }
  }
  if (zeta_new > FloatT(1.0) && dz != FloatT(0.0)) {
    FloatT s = (FloatT(1.0) - zeta_old) / dz;
    if (s >= FloatT(0.0) && s < crossing_s) {
      crossing_s = s;
      cross_idx = +1;
      cross_type = 1;
    }
  }

  for (int i = 0; i < 3; i++) {
    FloatT dl = l_new[i] - l_old[i];
    if (l_new[i] < FloatT(0.0) && dl != FloatT(0.0)) {
      FloatT s = -l_old[i] / dl;
      if (s >= FloatT(0.0) && s < crossing_s) {
        crossing_s = s;
        cross_idx = i;
        cross_type = 2;
      }
    }
  }

  return cross_type;
}

// =========================================================================
// Multi-prism current deposition with trajectory splitting
// =========================================================================
template <typename FloatT>
HOST_DEVICE void deposit_current(
    const prismatic_mesh_ptrs& mesh,
    int tri_idx, int layer_idx,
    const FloatT l_old[3], FloatT zeta_old,
    const FloatT l_new[3], FloatT zeta_new,
    FloatT q_over_dt,
    FloatT* J,
    int& new_tri, int& new_layer) {
  const int opposite_edge[3] = {1, 2, 0};
  const int max_crossings = 4;

  // Compute physical 3D target position once
  FloatT tgt_sx = FloatT(0.0), tgt_sy = FloatT(0.0), tgt_sz = FloatT(0.0);
  for (int i = 0; i < 3; i++) {
    int sv = mesh.tri_verts[tri_idx * 3 + i];
    tgt_sx += l_new[i] * mesh.sphere_vx[sv];
    tgt_sy += l_new[i] * mesh.sphere_vy[sv];
    tgt_sz += l_new[i] * mesh.sphere_vz[sv];
  }
  FloatT dr0 = mesh.radii[layer_idx + 1] - mesh.radii[layer_idx];
  FloatT tgt_r = mesh.radii[layer_idx] + zeta_new * dr0;

  FloatT cur_l[3] = {l_old[0], l_old[1], l_old[2]};
  FloatT cur_z = zeta_old;
  int cur_tri = tri_idx;
  int cur_layer = layer_idx;

  for (int cross = 0; cross < max_crossings; cross++) {
    FloatT tgt_l[3];
    mesh.compute_barycentric(cur_tri, tgt_sx, tgt_sy, tgt_sz,
                             tgt_l[0], tgt_l[1], tgt_l[2]);
    FloatT tgt_z = mesh.compute_zeta(cur_layer, tgt_r);

    FloatT s;
    int cross_idx;
    int cross_type = detect_crossing(cur_l, cur_z, tgt_l, tgt_z, s, cross_idx);

    if (cross_type == 0) {
      deposit_current_single_prism(mesh, cur_tri, cur_layer,
                                   cur_l, cur_z, tgt_l, tgt_z,
                                   q_over_dt, J);
      new_tri = cur_tri;
      new_layer = cur_layer;
      return;
    }

    FloatT cross_l[3];
    for (int i = 0; i < 3; i++) {
      cross_l[i] = cur_l[i] + s * (tgt_l[i] - cur_l[i]);
    }
    FloatT cross_z = cur_z + s * (tgt_z - cur_z);

    if (cross_type == 1) {
      cross_z = (cross_idx < 0) ? FloatT(0.0) : FloatT(1.0);
    } else {
      cross_l[cross_idx] = FloatT(0.0);
    }

    deposit_current_single_prism(mesh, cur_tri, cur_layer,
                                 cur_l, cur_z, cross_l, cross_z,
                                 q_over_dt, J);

    if (cross_type == 1) {
      if (cross_idx < 0) {
        if (cur_layer <= 0) { new_tri = cur_tri; new_layer = 0; return; }
        cur_layer--;
        cur_z = FloatT(1.0);
      } else {
        if (cur_layer >= mesh.N_r - 1) { new_tri = cur_tri; new_layer = mesh.N_r - 1; return; }
        cur_layer++;
        cur_z = FloatT(0.0);
      }
      for (int i = 0; i < 3; i++) cur_l[i] = cross_l[i];
    } else {
      int edge_local = opposite_edge[cross_idx];
      int next_tri = mesh.tri_neighbor[cur_tri * 3 + edge_local];
      if (next_tri < 0) { new_tri = cur_tri; new_layer = cur_layer; return; }

      FloatT sx = FloatT(0.0), sy = FloatT(0.0), sz = FloatT(0.0);
      for (int i = 0; i < 3; i++) {
        int sv = mesh.tri_verts[cur_tri * 3 + i];
        sx += cross_l[i] * mesh.sphere_vx[sv];
        sy += cross_l[i] * mesh.sphere_vy[sv];
        sz += cross_l[i] * mesh.sphere_vz[sv];
      }
      mesh.compute_barycentric(next_tri, sx, sy, sz,
                               cur_l[0], cur_l[1], cur_l[2]);
      cur_z = cross_z;
      cur_tri = next_tri;
    }
  }

  // Exhausted crossing budget
  FloatT tgt_l[3];
  mesh.compute_barycentric(cur_tri, tgt_sx, tgt_sy, tgt_sz,
                           tgt_l[0], tgt_l[1], tgt_l[2]);
  FloatT tgt_z = mesh.compute_zeta(cur_layer, tgt_r);
  deposit_current_single_prism(mesh, cur_tri, cur_layer,
                               cur_l, cur_z, tgt_l, tgt_z,
                               q_over_dt, J);
  new_tri = cur_tri;
  new_layer = cur_layer;
}

// =========================================================================
// Whitney-form field interpolation
// =========================================================================
template <typename FloatT>
HOST_DEVICE void interpolate_fields(
    const prismatic_mesh_ptrs& mesh,
    int tri_idx, int layer_idx,
    const FloatT l[3], FloatT zeta,
    const FloatT* E_e, const FloatT* B_f,
    FloatT& Ex, FloatT& Ey, FloatT& Ez,
    FloatT& Bx, FloatT& By, FloatT& Bz) {
  int edges[9];
  mesh.prism_edge_indices(tri_idx, layer_idx, edges);

  int sv[3];
  for (int i = 0; i < 3; i++) sv[i] = mesh.tri_verts[tri_idx * 3 + i];

  // Vertex positions at radial midpoint
  FloatT px[3], py[3], pz[3];
  FloatT r_mid = FloatT(0.5) * (mesh.radii[layer_idx] + mesh.radii[layer_idx + 1]);
  for (int i = 0; i < 3; i++) {
    px[i] = r_mid * mesh.sphere_vx[sv[i]];
    py[i] = r_mid * mesh.sphere_vy[sv[i]];
    pz[i] = r_mid * mesh.sphere_vz[sv[i]];
  }

  // Face normal (unnormalized = 2A * n_hat)
  FloatT e1x = px[1]-px[0], e1y = py[1]-py[0], e1z = pz[1]-pz[0];
  FloatT e2x = px[2]-px[0], e2y = py[2]-py[0], e2z = pz[2]-pz[0];
  FloatT nx = e1y*e2z - e1z*e2y;
  FloatT ny = e1z*e2x - e1x*e2z;
  FloatT nz = e1x*e2y - e1y*e2x;
  FloatT two_A_sq = nx*nx + ny*ny + nz*nz;

  // Barycentric gradients
  FloatT gl[3][3];
  for (int i = 0; i < 3; i++) {
    int j = (i + 1) % 3, k = (i + 2) % 3;
    FloatT dx = px[k]-px[j], dy = py[k]-py[j], dz = pz[k]-pz[j];
    gl[i][0] = (ny*dz - nz*dy) / two_A_sq;
    gl[i][1] = (nz*dx - nx*dz) / two_A_sq;
    gl[i][2] = (nx*dy - ny*dx) / two_A_sq;
  }

  // Radial unit vector at particle position
  FloatT r_hat_x = 0, r_hat_y = 0, r_hat_z = 0;
  for (int i = 0; i < 3; i++) {
    r_hat_x += l[i] * mesh.sphere_vx[sv[i]];
    r_hat_y += l[i] * mesh.sphere_vy[sv[i]];
    r_hat_z += l[i] * mesh.sphere_vz[sv[i]];
  }
  FloatT r_norm = std::sqrt(r_hat_x*r_hat_x + r_hat_y*r_hat_y + r_hat_z*r_hat_z);
  if (r_norm > 0) { r_hat_x /= r_norm; r_hat_y /= r_norm; r_hat_z /= r_norm; }

  FloatT dr = mesh.radii[layer_idx + 1] - mesh.radii[layer_idx];
  FloatT dzeta_dx = r_hat_x / dr;
  FloatT dzeta_dy = r_hat_y / dr;
  FloatT dzeta_dz = r_hat_z / dr;

  FloatT phi[2] = {FloatT(1.0) - zeta, zeta};

  // E field: Whitney 1-form expansion
  const int circuit_from[3] = {0, 1, 2};
  const int circuit_to[3]   = {1, 2, 0};

  Ex = Ey = Ez = FloatT(0.0);
  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j], i_to = circuit_to[j];
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];
    FloatT wx = l[i_from]*gl[i_to][0] - l[i_to]*gl[i_from][0];
    FloatT wy = l[i_from]*gl[i_to][1] - l[i_to]*gl[i_from][1];
    FloatT wz = l[i_from]*gl[i_to][2] - l[i_to]*gl[i_from][2];
    for (int k = 0; k < 2; k++) {
      FloatT coeff = FloatT(sign) * E_e[edges[j + k*3]] * phi[k];
      Ex += coeff * wx; Ey += coeff * wy; Ez += coeff * wz;
    }
  }
  for (int i = 0; i < 3; i++) {
    FloatT coeff = E_e[edges[6+i]] * l[i];
    Ex += coeff*dzeta_dx; Ey += coeff*dzeta_dy; Ez += coeff*dzeta_dz;
  }

  // B field: Whitney 2-form expansion
  Bx = By = Bz = FloatT(0.0);
  FloatT dl12_x = gl[0][1]*gl[1][2] - gl[0][2]*gl[1][1];
  FloatT dl12_y = gl[0][2]*gl[1][0] - gl[0][0]*gl[1][2];
  FloatT dl12_z = gl[0][0]*gl[1][1] - gl[0][1]*gl[1][0];
  for (int k = 0; k < 2; k++) {
    FloatT coeff = FloatT(2.0) * B_f[mesh.tri_face_idx(layer_idx+k, tri_idx)] * phi[k];
    Bx += coeff*dl12_x; By += coeff*dl12_y; Bz += coeff*dl12_z;
  }
  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j], i_to = circuit_to[j];
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];
    int sphere_e = mesh.tri_edges_s[tri_idx * 3 + j];
    FloatT wx = l[i_from]*gl[i_to][0] - l[i_to]*gl[i_from][0];
    FloatT wy = l[i_from]*gl[i_to][1] - l[i_to]*gl[i_from][1];
    FloatT wz = l[i_from]*gl[i_to][2] - l[i_to]*gl[i_from][2];
    FloatT bx = wy*dzeta_dz - wz*dzeta_dy;
    FloatT by = wz*dzeta_dx - wx*dzeta_dz;
    FloatT bz = wx*dzeta_dy - wy*dzeta_dx;
    FloatT coeff = FloatT(sign) * B_f[mesh.rect_face_idx(layer_idx, sphere_e)];
    Bx += coeff*bx; By += coeff*by; Bz += coeff*bz;
  }
}

}  // namespace Aperture
