#pragma once

#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_gr_ks_ptrs.h"

namespace Aperture {

// Prismatic mesh augmented with precomputed Kerr-Schild 3+1 metric
// coefficients at every edge midpoint and face centroid.
//
// Usage:
//   prismatic_mesh_gr_ks mesh;
//   mesh.build(L, N_r, r_min, r_max);   // inherited from prismatic_mesh
//   mesh.compute_metric(a);              // precompute KS coefficients
//
// The metric data is stationary (time-independent), so compute_metric()
// only needs to be called once after build().
class prismatic_mesh_gr_ks : public prismatic_mesh {
 public:
  prismatic_mesh_gr_ks() = default;
  ~prismatic_mesh_gr_ks() = default;

  // Precompute all metric coefficients for spin parameter a (M=1).
  // Must be called after build().
  void compute_metric(Scalar a);

  Scalar m_a = 0.0;  // spin parameter

  // --- Per-edge metric data (at edge midpoints) ---
  buffer<Scalar> edge_f;
  buffer<Scalar> edge_lx, edge_ly, edge_lz;
  buffer<Scalar> edge_alpha;
  buffer<Scalar> edge_sgb_x, edge_sgb_y, edge_sgb_z;
  buffer<Scalar> edge_r;

  // --- Per-face metric data (at face centroids) ---
  buffer<Scalar> face_f;
  buffer<Scalar> face_lx, face_ly, face_lz;
  buffer<Scalar> face_alpha;
  buffer<Scalar> face_sgb_x, face_sgb_y, face_sgb_z;
  buffer<Scalar> face_r;

  // --- Pointer access ---
  prismatic_mesh_gr_ks_ptrs host_ptrs_gr() const;

  prismatic_mesh_gr_ks_ptrs get_ptrs(exec_tags::host) const {
    return host_ptrs_gr();
  }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  prismatic_mesh_gr_ks_ptrs dev_ptrs_gr() const;

  prismatic_mesh_gr_ks_ptrs get_ptrs(exec_tags::device) const {
    return dev_ptrs_gr();
  }

  void copy_to_device();
#endif
};

}  // namespace Aperture
