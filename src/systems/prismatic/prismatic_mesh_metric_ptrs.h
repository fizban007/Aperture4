#pragma once

#include "systems/prismatic/prismatic_mesh_ptrs.h"

namespace Aperture {

// Extends prismatic_mesh_ptrs with per-element metric data from
// prismatic_mesh_metric.  Passed by value to GPU kernels.
struct prismatic_mesh_metric_ptrs : prismatic_mesh_ptrs {
  // --- Per-edge 3+1 data (at edge midpoints) ---
  const Scalar* edge_r_coord;          // [N_edges]
  const Scalar* edge_sth;              // [N_edges]
  const Scalar* edge_cth;              // [N_edges]
  const Scalar* edge_alpha;            // [N_edges]
  const Scalar* edge_sq_gamma_beta_r;  // [N_edges]

  // --- Per-face 3+1 data (at face centroids) ---
  const Scalar* face_r_coord;          // [N_faces]
  const Scalar* face_sth;              // [N_faces]
  const Scalar* face_cth;              // [N_faces]
  const Scalar* face_alpha;            // [N_faces]
  const Scalar* face_sq_gamma_beta_r;  // [N_faces]

  // --- Element type boundaries (precomputed from mesh counts) ---
  int N_h_edges;     // horizontal edge count = (N_r + 1) * N_edge_s
  int N_tri_faces;   // triangular face count = (N_r + 1) * N_tri

  // --- Convenience: is edge horizontal / is face triangular? ---
  HD_INLINE bool is_horizontal_edge(int e) const { return e < N_h_edges; }
  HD_INLINE bool is_vertical_edge(int e) const { return e >= N_h_edges; }
  HD_INLINE bool is_tri_face(int f) const { return f < N_tri_faces; }
  HD_INLINE bool is_rect_face(int f) const { return f >= N_tri_faces; }
};

}  // namespace Aperture
