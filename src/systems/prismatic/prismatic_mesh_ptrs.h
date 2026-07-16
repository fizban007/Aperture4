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

  // --- Vertex positions (spherical) ---
  const Scalar* vert_r;          // [N_verts]
  const Scalar* vert_theta;
  const Scalar* vert_phi;

  // --- Primal geometry ---
  const Scalar* face_area;       // [N_faces]
  const Scalar* edge_length;     // [N_edges]
  const int* edge_v0;            // [N_edges]
  const int* edge_v1;

  // --- Face vertex indices ---
  const int* tri_face_v0;        // [N_tri * (N_r + 1)]
  const int* tri_face_v1;
  const int* tri_face_v2;
  const int* rect_face_v0;       // [N_edge_s * N_r]
  const int* rect_face_v1;
  const int* rect_face_v2;
  const int* rect_face_v3;

  // --- Sphere mesh data (for particle operations) ---
  const Scalar* sphere_vx;      // [N_vert_s]
  const Scalar* sphere_vy;
  const Scalar* sphere_vz;
  const Scalar* sphere_theta;   // [N_vert_s]  polar angle of unit-sphere vertex
  const Scalar* sphere_phi;     // [N_vert_s]  azimuth   of unit-sphere vertex
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
  // on the unit sphere, by central (gnomonic) projection: solve
  // p ∝ l1·v0 + l2·v1 + l3·v2 with l1+l2+l3 = ±1.
  //
  // Unlike the previous perpendicular-projection formula, this predicate
  // tiles the sphere EXACTLY (radial rays partition the convex icosphere
  // surface), so point location has no orphan slivers and find_triangle
  // always terminates on the containing triangle.  Normalizing by
  // |sum| keeps antipodal triangles all-negative (l1+l2+l3 = -1 there)
  // instead of letting the sign flip fool the containment test.
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

    // c_i = (v_{i+1} × v_{i+2}) · p  — cofactor expansion of the solve
    Scalar c12x = p1y * p2z - p1z * p2y;
    Scalar c12y = p1z * p2x - p1x * p2z;
    Scalar c12z = p1x * p2y - p1y * p2x;
    Scalar c20x = p2y * p0z - p2z * p0y;
    Scalar c20y = p2z * p0x - p2x * p0z;
    Scalar c20z = p2x * p0y - p2y * p0x;
    Scalar c01x = p0y * p1z - p0z * p1y;
    Scalar c01y = p0z * p1x - p0x * p1z;
    Scalar c01z = p0x * p1y - p0y * p1x;

    Scalar c0 = c12x * sx + c12y * sy + c12z * sz;
    Scalar c1 = c20x * sx + c20y * sy + c20z * sz;
    Scalar c2 = c01x * sx + c01y * sy + c01z * sz;

    // det = v0 · (v1 × v2); its sign accounts for either vertex winding
    Scalar det = c12x * p0x + c12y * p0y + c12z * p0z;
    Scalar sum = c0 + c1 + c2;
    Scalar denom = std::abs(sum);
    if (denom < Scalar(1e-30)) denom = Scalar(1e-30);
    Scalar inv = (det >= Scalar(0.0) ? Scalar(1.0) : Scalar(-1.0)) / denom;

    l1 = c0 * inv;
    l2 = c1 * inv;
    l3 = c2 * inv;
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
