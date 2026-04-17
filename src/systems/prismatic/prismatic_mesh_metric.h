#pragma once

#include "systems/physics/spherical_metric.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_metric_ptrs.h"

namespace Aperture {

// Prismatic mesh with metric-aware Hodge stars and precomputed 3+1
// quantities.  Extends the flat prismatic_mesh with any user-supplied
// spherical metric struct (see spherical_metric.hpp): flat space,
// Schwarzschild, Kerr-Schild, etc.
//
// Usage (GPU-aware; call copy_to_device() before compute_metric()):
//   prismatic_mesh_metric mesh;
//   mesh.build(L, N_r, r_min, r_max);
// #if GPU_ENABLED
//   mesh.copy_to_device();
// #endif
//   mesh.compute_metric(flat_spherical_metric{});       // flat space
//   mesh.compute_metric(ks_spherical_metric{0.9});      // Kerr a=0.9
//
// compute_metric() is templated on the concrete metric type so that
// the metric object (a plain-old-data struct with HD_INLINE methods,
// see spherical_metric.hpp) can be captured by value into GPU kernels.
// Two explicit instantiations — flat_spherical_metric and
// ks_spherical_metric — are provided in prismatic_mesh_metric.cpp /
// .hip.cpp so callers do not need to include the impl header.
class prismatic_mesh_metric : public prismatic_mesh {
 public:
  prismatic_mesh_metric() = default;
  ~prismatic_mesh_metric() = default;

  // Compute metric-weighted Hodge stars and per-element 3+1 data.
  // Must be called after build() and, on GPU builds, after
  // copy_to_device() (the mesh topology must be on the device where
  // the kernels run).  Overwrites hodge1_inv, hodge2 and populates the
  // per-element buffers below.  Templated on the concrete metric type.
  template <typename Metric>
  void compute_metric(const Metric& metric);

  // --- Per-edge data (at edge midpoints) ---
  buffer<Scalar> edge_r_coord;          // BL/KS radius         [N_edges]
  buffer<Scalar> edge_sth;              // sin(theta)           [N_edges]
  buffer<Scalar> edge_cth;              // cos(theta)           [N_edges]
  buffer<Scalar> edge_alpha;            // lapse                [N_edges]
  buffer<Scalar> edge_sq_gamma_beta_r;  // sqrt(gamma)*beta^r   [N_edges]
  buffer<Scalar> edge_sqrt_gamma;       // sqrt(det gamma)      [N_edges]

  // --- Per-face data (at face centroids) ---
  buffer<Scalar> face_r_coord;          // [N_faces]
  buffer<Scalar> face_sth;              // [N_faces]
  buffer<Scalar> face_cth;              // [N_faces]
  buffer<Scalar> face_alpha;            // [N_faces]
  buffer<Scalar> face_sq_gamma_beta_r;  // [N_faces]
  buffer<Scalar> face_sqrt_gamma;       // [N_faces]

  // Fixed-width adjacency tables, precomputed on host during
  // compute_metric() (cheap topological scan) and uploaded to the
  // device so the Hodge kernel can read them without dynamic vectors.
  //
  //   edge_tris[2*e + 0/1] = two triangles adjacent to sphere edge e
  //                          (-1 if boundary).
  //   vert_tri_count[s]    = number of triangles incident at sphere
  //                          vertex s (≤ max_vert_valence).
  //   vert_tris[max_vert_valence*s + k] = k-th incident triangle
  //                                       (0 <= k < count).
  //
  // max_vert_valence = 6 for icosphere subdivisions (12 vertices of
  // valence 5 at the original icosahedron corners, the rest valence 6).
  static constexpr int max_vert_valence = 6;
  buffer<int> edge_tris;                // [2 * N_edge_s]
  buffer<int> vert_tri_count;           // [N_vert_s]
  buffer<int> vert_tris;                // [max_vert_valence * N_vert_s]

  // --- Pointer access ---
  prismatic_mesh_metric_ptrs host_ptrs_metric() const;
  prismatic_mesh_metric_ptrs get_ptrs(exec_tags::host) const {
    return host_ptrs_metric();
  }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  prismatic_mesh_metric_ptrs dev_ptrs_metric() const;
  prismatic_mesh_metric_ptrs get_ptrs(exec_tags::device) const {
    return dev_ptrs_metric();
  }
  void copy_to_device();

  // After compute_metric has run on GPU, copy the updated hodge stars
  // and per-element metric buffers back to host.  Needed because the
  // data exporter writes mesh.hodge1_inv/hodge2 via host_ptr().
  void copy_metric_to_host();
#endif
};

}  // namespace Aperture
