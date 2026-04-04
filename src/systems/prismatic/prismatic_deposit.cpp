#include "systems/prismatic/prismatic_deposit.h"
#include <cmath>

namespace Aperture {

void deposit_current_single_prism(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l_old[3], Scalar zeta_old,
    const Scalar l_new[3], Scalar zeta_new,
    Scalar q_over_dt,
    Scalar* J) {
  // Deltas
  Scalar dl[3] = {l_new[0] - l_old[0],
                  l_new[1] - l_old[1],
                  l_new[2] - l_old[2]};
  Scalar dz = zeta_new - zeta_old;

  // Radial hat function averages
  Scalar phi_bar_0 = 0.5 * ((1.0 - zeta_old) + (1.0 - zeta_new));  // bottom
  Scalar phi_bar_1 = 0.5 * (zeta_old + zeta_new);                    // top

  // Triangle boundary circuit: edge j goes from circuit_from[j] to circuit_to[j]
  //   edge 0: local 0 -> 1
  //   edge 1: local 1 -> 2
  //   edge 2: local 2 -> 0
  static const int circuit_from[3] = {0, 1, 2};
  static const int circuit_to[3]   = {1, 2, 0};

  // Deposit horizontal edge currents (6 edges: 3 bottom + 3 top)
  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j];
    int i_to = circuit_to[j];

    // Angular factor A_ij = l_old[i]*dl[j] - l_old[j]*dl[i]
    Scalar A = l_old[i_from] * dl[i_to] - l_old[i_to] * dl[i_from];

    // Sign: converts from circuit direction to canonical edge direction
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];
    int sphere_e = mesh.tri_edges_s[tri_idx * 3 + j];

    // Bottom horizontal edge (shell layer_idx)
    J[mesh.h_edge_idx(layer_idx, sphere_e)] +=
        q_over_dt * sign * A * phi_bar_0;

    // Top horizontal edge (shell layer_idx + 1)
    J[mesh.h_edge_idx(layer_idx + 1, sphere_e)] +=
        q_over_dt * sign * A * phi_bar_1;
  }

  // Deposit vertical edge currents (3 edges)
  for (int i = 0; i < 3; i++) {
    int sphere_v = mesh.tri_verts[tri_idx * 3 + i];
    Scalar avg_lambda = 0.5 * (l_old[i] + l_new[i]);
    J[mesh.v_edge_idx(layer_idx, sphere_v)] += q_over_dt * dz * avg_lambda;
  }
}

int detect_crossing(const Scalar l_old[3], Scalar zeta_old,
                    const Scalar l_new[3], Scalar zeta_new,
                    Scalar& crossing_s, int& cross_idx) {
  crossing_s = 1.0;
  cross_idx = 0;
  int cross_type = 0;

  // Check radial crossings: zeta < 0 (bottom) or zeta > 1 (top)
  Scalar dz = zeta_new - zeta_old;
  if (zeta_new < 0.0 && dz != 0.0) {
    Scalar s = -zeta_old / dz;
    if (s >= 0.0 && s < crossing_s) {
      crossing_s = s;
      cross_idx = -1;  // bottom
      cross_type = 1;
    }
  }
  if (zeta_new > 1.0 && dz != 0.0) {
    Scalar s = (1.0 - zeta_old) / dz;
    if (s >= 0.0 && s < crossing_s) {
      crossing_s = s;
      cross_idx = +1;  // top
      cross_type = 1;
    }
  }

  // Check angular crossings: any lambda_i < 0
  for (int i = 0; i < 3; i++) {
    Scalar dl = l_new[i] - l_old[i];
    if (l_new[i] < 0.0 && dl != 0.0) {
      Scalar s = -l_old[i] / dl;
      if (s >= 0.0 && s < crossing_s) {
        crossing_s = s;
        cross_idx = i;
        cross_type = 2;
      }
    }
  }

  return cross_type;
}

void deposit_current(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l_old[3], Scalar zeta_old,
    const Scalar l_new[3], Scalar zeta_new,
    Scalar q_over_dt,
    Scalar* J,
    int& new_tri, int& new_layer) {
  // Opposite edge mapping: lambda_i < 0 means cross edge opposite_edge[i]
  //   l0 < 0 -> edge 1 (v1-v2), l1 < 0 -> edge 2 (v0-v2), l2 < 0 -> edge 0 (v0-v1)
  static const int opposite_edge[3] = {1, 2, 0};

  static const int max_crossings = 4;

  // Compute the physical 3D target position once. After each crossing we
  // recompute the local (lambda, zeta) coordinates in the new prism from this.
  Scalar tgt_sx = 0.0, tgt_sy = 0.0, tgt_sz = 0.0;
  for (int i = 0; i < 3; i++) {
    int sv = mesh.tri_verts[tri_idx * 3 + i];
    tgt_sx += l_new[i] * mesh.sphere_vx[sv];
    tgt_sy += l_new[i] * mesh.sphere_vy[sv];
    tgt_sz += l_new[i] * mesh.sphere_vz[sv];
  }
  Scalar dr0 = mesh.radii[layer_idx + 1] - mesh.radii[layer_idx];
  Scalar tgt_r = mesh.radii[layer_idx] + zeta_new * dr0;

  Scalar cur_l[3] = {l_old[0], l_old[1], l_old[2]};
  Scalar cur_z = zeta_old;
  int cur_tri = tri_idx;
  int cur_layer = layer_idx;

  for (int cross = 0; cross < max_crossings; cross++) {
    // Compute target in current prism coordinates
    Scalar tgt_l[3];
    mesh.compute_barycentric(cur_tri, tgt_sx, tgt_sy, tgt_sz,
                             tgt_l[0], tgt_l[1], tgt_l[2]);
    Scalar tgt_z = mesh.compute_zeta(cur_layer, tgt_r);

    Scalar s;
    int cross_idx;
    int cross_type = detect_crossing(cur_l, cur_z, tgt_l, tgt_z, s, cross_idx);

    if (cross_type == 0) {
      // No crossing — deposit in current prism and done
      deposit_current_single_prism(mesh, cur_tri, cur_layer,
                                   cur_l, cur_z, tgt_l, tgt_z,
                                   q_over_dt, J);
      new_tri = cur_tri;
      new_layer = cur_layer;
      return;
    }

    // Compute crossing point
    Scalar cross_l[3];
    for (int i = 0; i < 3; i++) {
      cross_l[i] = cur_l[i] + s * (tgt_l[i] - cur_l[i]);
    }
    Scalar cross_z = cur_z + s * (tgt_z - cur_z);

    // Clamp crossing coordinates to boundary
    if (cross_type == 1) {
      cross_z = (cross_idx < 0) ? Scalar(0.0) : Scalar(1.0);
    } else {
      cross_l[cross_idx] = 0.0;
    }

    // Deposit from cur to crossing point in current prism
    deposit_current_single_prism(mesh, cur_tri, cur_layer,
                                 cur_l, cur_z, cross_l, cross_z,
                                 q_over_dt, J);

    // Move to the next prism
    if (cross_type == 1) {
      // Radial crossing
      if (cross_idx < 0) {
        if (cur_layer <= 0) {
          new_tri = cur_tri;
          new_layer = 0;
          return;  // Hit inner boundary
        }
        cur_layer--;
        cur_z = 1.0;  // Top of the prism below
      } else {
        if (cur_layer >= mesh.m_N_r - 1) {
          new_tri = cur_tri;
          new_layer = mesh.m_N_r - 1;
          return;  // Hit outer boundary
        }
        cur_layer++;
        cur_z = 0.0;  // Bottom of the prism above
      }
      // Barycentric coordinates unchanged for radial crossing
      for (int i = 0; i < 3; i++) cur_l[i] = cross_l[i];
    } else {
      // Angular crossing: lambda[cross_idx] went to 0
      int edge_local = opposite_edge[cross_idx];
      int next_tri = mesh.tri_neighbor[cur_tri * 3 + edge_local];

      if (next_tri < 0) {
        new_tri = cur_tri;
        new_layer = cur_layer;
        return;  // Shouldn't happen on closed sphere
      }

      // Recompute crossing-point barycentric coords in the neighbor triangle
      Scalar sx = 0.0, sy = 0.0, sz = 0.0;
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

  // Exhausted crossing budget — deposit remainder in current prism
  Scalar tgt_l[3];
  mesh.compute_barycentric(cur_tri, tgt_sx, tgt_sy, tgt_sz,
                           tgt_l[0], tgt_l[1], tgt_l[2]);
  Scalar tgt_z = mesh.compute_zeta(cur_layer, tgt_r);
  deposit_current_single_prism(mesh, cur_tri, cur_layer,
                               cur_l, cur_z, tgt_l, tgt_z,
                               q_over_dt, J);
  new_tri = cur_tri;
  new_layer = cur_layer;
}

void interpolate_fields(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l[3], Scalar zeta,
    const Scalar* E_e, const Scalar* B_f,
    Scalar& Ex, Scalar& Ey, Scalar& Ez,
    Scalar& Bx, Scalar& By, Scalar& Bz) {
  // Get prism edge/face indices
  int edges[9];
  mesh.prism_edge_indices(tri_idx, layer_idx, edges);

  // Get triangle vertex positions on unit sphere
  int sv[3];
  for (int i = 0; i < 3; i++) {
    sv[i] = mesh.tri_verts[tri_idx * 3 + i];
  }

  // Compute gradients of barycentric coordinates in 3D.
  // For triangle (p0, p1, p2), grad(lambda_i) = n x (p_{i+2} - p_{i+1}) / (2A)
  // where n is the unit normal and A is the triangle area.
  Scalar px[3], py[3], pz[3];
  Scalar r_mid = 0.5 * (mesh.radii[layer_idx] + mesh.radii[layer_idx + 1]);
  for (int i = 0; i < 3; i++) {
    px[i] = r_mid * mesh.sphere_vx[sv[i]];
    py[i] = r_mid * mesh.sphere_vy[sv[i]];
    pz[i] = r_mid * mesh.sphere_vz[sv[i]];
  }

  // Face normal (unnormalized = 2A * n_hat)
  Scalar e1x = px[1] - px[0], e1y = py[1] - py[0], e1z = pz[1] - pz[0];
  Scalar e2x = px[2] - px[0], e2y = py[2] - py[0], e2z = pz[2] - pz[0];
  Scalar nx = e1y * e2z - e1z * e2y;
  Scalar ny = e1z * e2x - e1x * e2z;
  Scalar nz = e1x * e2y - e1y * e2x;
  Scalar two_A_sq = nx * nx + ny * ny + nz * nz;

  // grad(lambda_i) = n x (p_{next_next} - p_{next}) / |n|^2
  // Cyclic: for i, next = (i+1)%3, next_next = (i+2)%3
  Scalar gl[3][3];  // gl[i][xyz] = gradient of lambda_i
  for (int i = 0; i < 3; i++) {
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    Scalar dx = px[k] - px[j], dy = py[k] - py[j], dz = pz[k] - pz[j];
    // n x d
    gl[i][0] = (ny * dz - nz * dy) / two_A_sq;
    gl[i][1] = (nz * dx - nx * dz) / two_A_sq;
    gl[i][2] = (nx * dy - ny * dx) / two_A_sq;
  }

  // Radial unit vector at particle position (for dζ direction)
  Scalar r_hat_x = 0, r_hat_y = 0, r_hat_z = 0;
  for (int i = 0; i < 3; i++) {
    r_hat_x += l[i] * mesh.sphere_vx[sv[i]];
    r_hat_y += l[i] * mesh.sphere_vy[sv[i]];
    r_hat_z += l[i] * mesh.sphere_vz[sv[i]];
  }
  Scalar r_norm = std::sqrt(r_hat_x * r_hat_x + r_hat_y * r_hat_y +
                            r_hat_z * r_hat_z);
  if (r_norm > 0) { r_hat_x /= r_norm; r_hat_y /= r_norm; r_hat_z /= r_norm; }

  // Physical dζ direction: dζ maps [r_k, r_{k+1}] -> [0,1], so the
  // gradient of ζ in 3D is r_hat / dr
  Scalar dr = mesh.radii[layer_idx + 1] - mesh.radii[layer_idx];
  Scalar dzeta_dx = r_hat_x / dr;
  Scalar dzeta_dy = r_hat_y / dr;
  Scalar dzeta_dz = r_hat_z / dr;

  // Hat functions
  Scalar phi[2] = {Scalar(1.0) - zeta, zeta};

  // E field: sum over 9 edges of E_e * W^1_e
  // W^1_{ij,k} = (lambda_i * d(lambda_j) - lambda_j * d(lambda_i)) * phi_k(zeta)
  // W^1_i = lambda_i * d(zeta)
  Ex = Ey = Ez = 0.0;

  // Horizontal edges: 6 edges (3 sphere edges x 2 levels)
  static const int circuit_from[3] = {0, 1, 2};
  static const int circuit_to[3]   = {1, 2, 0};

  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j];
    int i_to = circuit_to[j];
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];

    // Whitney 1-form: (l_i * dl_j - l_j * dl_i)
    // = l_i * grad(l_j) - l_j * grad(l_i)
    Scalar wx = l[i_from] * gl[i_to][0] - l[i_to] * gl[i_from][0];
    Scalar wy = l[i_from] * gl[i_to][1] - l[i_to] * gl[i_from][1];
    Scalar wz = l[i_from] * gl[i_to][2] - l[i_to] * gl[i_from][2];

    for (int k = 0; k < 2; k++) {
      int edge_idx = edges[j + k * 3];  // bottom (k=0) or top (k=1)
      Scalar coeff = sign * E_e[edge_idx] * phi[k];
      Ex += coeff * wx;
      Ey += coeff * wy;
      Ez += coeff * wz;
    }
  }

  // Vertical edges: 3 edges
  for (int i = 0; i < 3; i++) {
    int edge_idx = edges[6 + i];
    // W^1_i = lambda_i * d(zeta)
    Scalar coeff = E_e[edge_idx] * l[i];
    Ex += coeff * dzeta_dx;
    Ey += coeff * dzeta_dy;
    Ez += coeff * dzeta_dz;
  }

  // B field: sum over 5 faces of B_f * W^2_f
  // W^2_k = 2 * dl1 ^ dl2 * phi_k(zeta)   (triangular faces)
  // W^2_{ij} = (l_i * dl_j - l_j * dl_i) ^ d(zeta)  (rectangular faces)
  Bx = By = Bz = 0.0;

  // Triangular faces (2): W^2_k = 2 * (dl1 ^ dl2) * phi_k
  // dl1 ^ dl2 as a vector (Hodge dual in 3D) = grad(l1) x grad(l2)
  Scalar dl12_x = gl[0][1] * gl[1][2] - gl[0][2] * gl[1][1];
  Scalar dl12_y = gl[0][2] * gl[1][0] - gl[0][0] * gl[1][2];
  Scalar dl12_z = gl[0][0] * gl[1][1] - gl[0][1] * gl[1][0];

  for (int k = 0; k < 2; k++) {
    int face_idx = mesh.tri_face_idx(layer_idx + k, tri_idx);
    Scalar coeff = 2.0 * B_f[face_idx] * phi[k];
    Bx += coeff * dl12_x;
    By += coeff * dl12_y;
    Bz += coeff * dl12_z;
  }

  // Rectangular faces (3): W^2_{ij} = (l_i * dl_j - l_j * dl_i) ^ d(zeta)
  for (int j = 0; j < 3; j++) {
    int i_from = circuit_from[j];
    int i_to = circuit_to[j];
    int sign = mesh.tri_edge_signs[tri_idx * 3 + j];

    int sphere_e = mesh.tri_edges_s[tri_idx * 3 + j];
    int face_idx = mesh.rect_face_idx(layer_idx, sphere_e);

    // (l_i * gl_j - l_j * gl_i) x d(zeta)
    Scalar wx = l[i_from] * gl[i_to][0] - l[i_to] * gl[i_from][0];
    Scalar wy = l[i_from] * gl[i_to][1] - l[i_to] * gl[i_from][1];
    Scalar wz = l[i_from] * gl[i_to][2] - l[i_to] * gl[i_from][2];

    // cross product with d(zeta)
    Scalar bx = wy * dzeta_dz - wz * dzeta_dy;
    Scalar by = wz * dzeta_dx - wx * dzeta_dz;
    Scalar bz = wx * dzeta_dy - wy * dzeta_dx;

    Scalar coeff = sign * B_f[face_idx];
    Bx += coeff * bx;
    By += coeff * by;
    Bz += coeff * bz;
  }
}

}  // namespace Aperture
