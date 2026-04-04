#pragma once

#include "core/gpu_translation_layer.h"
#include "core/typedefs_and_constants.h"
#include <cmath>

namespace Aperture {

// Lightweight, trivially-copyable struct holding raw pointers into the
// prismatic mesh data.  Can be passed by value to GPU kernels.
//
// Populate via prismatic_mesh::host_ptrs() or prismatic_mesh::dev_ptrs().
// All HD_INLINE methods mirror the corresponding prismatic_mesh methods
// but operate on the raw pointers in this struct.
struct prismatic_mesh_ptrs {
  // --- Mesh parameters (scalar copies) ---
  int N_r;
  int N_tri;
  int N_vert_s;
  int N_edge_s;
  int N_verts;
  int N_edges;
  int N_faces;

  // --- Radii ---
  const Scalar* radii;          // [N_r + 1]

  // --- Incidence d1 in CSR (face -> edges) ---
  const int* d1_row_ptr;        // [N_faces + 1]
  const int* d1_col_idx;
  const Scalar* d1_val;

  // --- Transpose d1^T in CSR (edge -> faces) ---
  const int* d1t_row_ptr;       // [N_edges + 1]
  const int* d1t_col_idx;
  const Scalar* d1t_val;

  // --- Hodge star ---
  const Scalar* hodge1_inv;     // [N_edges]
  const Scalar* hodge2;         // [N_faces]

  // --- Boundary tags ---
  const int* edge_boundary;     // [N_edges]
  const int* face_boundary;     // [N_faces]
  const int* edge_radial_layer; // [N_edges]
  const int* face_radial_layer; // [N_faces]

  // --- Sphere mesh data (for particle operations) ---
  const Scalar* sphere_vx;      // [N_vert_s]
  const Scalar* sphere_vy;
  const Scalar* sphere_vz;
  const int* tri_verts;         // [N_tri * 3]
  const int* tri_edges_s;       // [N_tri * 3]
  const int* tri_edge_signs;    // [N_tri * 3]
  const int* tri_neighbor;      // [N_tri * 3]

  // =======================================================================
  // Indexing helpers
  // =======================================================================

  HD_INLINE int h_edge_idx(int k, int e) const {
    return k * N_edge_s + e;
  }
  HD_INLINE int v_edge_idx(int k, int s) const {
    return (N_r + 1) * N_edge_s + k * N_vert_s + s;
  }
  HD_INLINE int tri_face_idx(int k, int t) const {
    return k * N_tri + t;
  }
  HD_INLINE int rect_face_idx(int k, int e) const {
    return (N_r + 1) * N_tri + k * N_edge_s + e;
  }

  // =======================================================================
  // Get the 9 global edge indices for prism (tri_idx, layer_idx).
  // edges[0..2] = bottom horizontal (shell k)
  // edges[3..5] = top horizontal (shell k+1)
  // edges[6..8] = vertical
  // =======================================================================
  HD_INLINE void prism_edge_indices(int tri_idx, int layer_idx,
                                    int edges[9]) const {
    edges[0] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 0]);
    edges[1] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 1]);
    edges[2] = h_edge_idx(layer_idx, tri_edges_s[tri_idx * 3 + 2]);
    edges[3] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 0]);
    edges[4] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 1]);
    edges[5] = h_edge_idx(layer_idx + 1, tri_edges_s[tri_idx * 3 + 2]);
    edges[6] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 0]);
    edges[7] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 1]);
    edges[8] = v_edge_idx(layer_idx, tri_verts[tri_idx * 3 + 2]);
  }

  // =======================================================================
  // Barycentric coordinates of point (sx,sy,sz) in triangle tri_idx
  // on the unit sphere.
  // =======================================================================
  HOST_DEVICE void compute_barycentric(int tri_idx, Scalar sx, Scalar sy,
                                     Scalar sz, Scalar& l1, Scalar& l2,
                                     Scalar& l3) const {
    int v0 = tri_verts[tri_idx * 3 + 0];
    int v1 = tri_verts[tri_idx * 3 + 1];
    int v2 = tri_verts[tri_idx * 3 + 2];

    Scalar p0x = sphere_vx[v0], p0y = sphere_vy[v0], p0z = sphere_vz[v0];
    Scalar p1x = sphere_vx[v1], p1y = sphere_vy[v1], p1z = sphere_vz[v1];
    Scalar p2x = sphere_vx[v2], p2y = sphere_vy[v2], p2z = sphere_vz[v2];

    Scalar e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
    Scalar e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;
    Scalar nx = e1y * e2z - e1z * e2y;
    Scalar ny = e1z * e2x - e1x * e2z;
    Scalar nz = e1x * e2y - e1y * e2x;
    Scalar n_dot_n = nx * nx + ny * ny + nz * nz;

    Scalar d1x = p1x - sx, d1y = p1y - sy, d1z = p1z - sz;
    Scalar d2x = p2x - sx, d2y = p2y - sy, d2z = p2z - sz;
    l1 = (d1y*d2z - d1z*d2y) * nx + (d1z*d2x - d1x*d2z) * ny +
         (d1x*d2y - d1y*d2x) * nz;
    l1 /= n_dot_n;

    Scalar d0x = p0x - sx, d0y = p0y - sy, d0z = p0z - sz;
    l2 = (d2y*d0z - d2z*d0y) * nx + (d2z*d0x - d2x*d0z) * ny +
         (d2x*d0y - d2y*d0x) * nz;
    l2 /= n_dot_n;

    l3 = Scalar(1.0) - l1 - l2;
  }

  // =======================================================================
  // Find radial layer k such that radii[k] <= r < radii[k+1].
  // Returns -1 if out of range.
  // =======================================================================
  HD_INLINE int find_radial_layer(Scalar r) const {
    if (r < radii[0] || r > radii[N_r]) return -1;
    int lo = 0, hi = N_r - 1;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (r < radii[mid + 1]) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }

  // =======================================================================
  // Normalized radial coordinate within layer k.
  // =======================================================================
  HD_INLINE Scalar compute_zeta(int k, Scalar r) const {
    return (r - radii[k]) / (radii[k + 1] - radii[k]);
  }

  // =======================================================================
  // Find which sphere triangle contains (sx, sy, sz) on the unit sphere.
  // Walk algorithm starting from tri_hint.
  // =======================================================================
  HOST_DEVICE int find_triangle(Scalar sx, Scalar sy, Scalar sz,
                              int tri_hint = -1) const {
    int t = tri_hint;
    if (t < 0 || t >= N_tri) t = 0;

    // opposite_edge[i] = edge index across which to walk when lambda_i < 0
    // l0 < 0 -> edge 1, l1 < 0 -> edge 2, l2 < 0 -> edge 0
    const int opposite_edge[3] = {1, 2, 0};

    for (int iter = 0; iter < N_tri; iter++) {
      Scalar l1, l2, l3;
      compute_barycentric(t, sx, sy, sz, l1, l2, l3);

      if (l1 >= Scalar(-1e-10) && l2 >= Scalar(-1e-10) && l3 >= Scalar(-1e-10))
        return t;

      Scalar lam[3] = {l1, l2, l3};
      int min_idx = 0;
      if (lam[1] < lam[min_idx]) min_idx = 1;
      if (lam[2] < lam[min_idx]) min_idx = 2;

      int next = tri_neighbor[t * 3 + opposite_edge[min_idx]];
      if (next < 0) return t;
      t = next;
    }
    return t;  // fallback
  }
};

}  // namespace Aperture
