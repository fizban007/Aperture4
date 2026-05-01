#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include <vector>

namespace Aperture {

class prismatic_mesh_metric;

// =========================================================================
// Local-sized per-cochain copies of prismatic_mesh_metric data.
//
// The metric mesh adds, per edge and per face, the 3+1 scalar quantities
// the GR-DEC solver needs to build E_aux and H_aux pointwise:
//   alpha               (lapse)
//   sq_gamma_beta_r     (√γ · β^r — the densitised radial shift)
//   sqrt_gamma          (√γ at the element's representative point)
// plus the (r, sin θ, cos θ) coordinates of each element's
// representative point (edge midpoint for edges, face circumcenter
// for tri faces, face centroid for rect faces).
//
// Like mesh_local, this class splits the metric-mesh's flat per-edge
// arrays into h_edge_* / v_edge_* and the per-face arrays into
// tri_face_* / rect_face_*.  Sizes are layout(cochain_type).local_size()
// from a prismatic_mesh_partition bundle.
//
// build() is called after compute_metric() has populated the global
// metric mesh; it copies the relevant slices into rank-local arrays.
// =========================================================================
class prismatic_mesh_metric_local {
 public:
  static prismatic_mesh_metric_local build(
      const prismatic_mesh_metric& mesh_metric,
      const prismatic_mesh_partition& mp);

  const prismatic_mesh_partition& partition() const { return *m_partition; }

  // ---- Tri-face metric scalars (sampled at tri circumcenter) ----
  std::vector<Scalar> tri_face_r_coord;
  std::vector<Scalar> tri_face_sth;
  std::vector<Scalar> tri_face_cth;
  std::vector<Scalar> tri_face_alpha;
  std::vector<Scalar> tri_face_sq_gamma_beta_r;
  std::vector<Scalar> tri_face_sqrt_gamma;

  // ---- Rect-face metric scalars (sampled at face centroid) ----
  std::vector<Scalar> rect_face_r_coord;
  std::vector<Scalar> rect_face_sth;
  std::vector<Scalar> rect_face_cth;
  std::vector<Scalar> rect_face_alpha;
  std::vector<Scalar> rect_face_sq_gamma_beta_r;
  std::vector<Scalar> rect_face_sqrt_gamma;

  // ---- H-edge metric scalars (sampled at edge midpoint) ----
  std::vector<Scalar> h_edge_r_coord;
  std::vector<Scalar> h_edge_sth;
  std::vector<Scalar> h_edge_cth;
  std::vector<Scalar> h_edge_alpha;
  std::vector<Scalar> h_edge_sq_gamma_beta_r;
  std::vector<Scalar> h_edge_sqrt_gamma;

  // ---- V-edge metric scalars (sampled at edge midpoint) ----
  std::vector<Scalar> v_edge_r_coord;
  std::vector<Scalar> v_edge_sth;
  std::vector<Scalar> v_edge_cth;
  std::vector<Scalar> v_edge_alpha;
  std::vector<Scalar> v_edge_sq_gamma_beta_r;
  std::vector<Scalar> v_edge_sqrt_gamma;

 private:
  const prismatic_mesh_partition* m_partition = nullptr;
};

}  // namespace Aperture
