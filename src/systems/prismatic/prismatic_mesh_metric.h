#pragma once

#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_metric_ptrs.h"
#include "systems/physics/spherical_metric.hpp"
#include <array>
#include <vector>

namespace Aperture {

// Prismatic mesh with metric-aware Hodge stars and precomputed 3+1
// quantities.  Replaces both prismatic_mesh (flat) and
// prismatic_mesh_gr_ks (Kerr-Schild) with a single class that takes
// any spherical_metric_t.
//
// Usage:
//   prismatic_mesh_metric mesh;
//   mesh.build(L, N_r, r_min, r_max);
//   mesh.compute_metric(flat_spherical_metric{});       // flat space
//   mesh.compute_metric(ks_spherical_metric{0.9});      // Kerr a=0.9
//
// After compute_metric():
//   - hodge1_inv and hodge2 are metric-weighted (overwritten)
//   - Per-element lapse, shift, and sqrt_gamma are stored
//   - Per-element spherical coordinates (r, sth, cth) are stored
class prismatic_mesh_metric : public prismatic_mesh {
 public:
  prismatic_mesh_metric() = default;
  ~prismatic_mesh_metric() = default;

  // Compute metric-weighted Hodge stars and per-element 3+1 data.
  // Must be called after build().
  void compute_metric(const spherical_metric_t& metric);

  // --- Per-edge data (at edge midpoints) ---
  buffer<Scalar> edge_r_coord;     // BL/KS radius          [N_edges]
  buffer<Scalar> edge_sth;         // sin(theta)             [N_edges]
  buffer<Scalar> edge_cth;         // cos(theta)             [N_edges]
  buffer<Scalar> edge_alpha;       // lapse                  [N_edges]
  buffer<Scalar> edge_sq_gamma_beta_r;  // sqrt(gamma)*beta^r [N_edges]

  // --- Per-face data (at face centroids) ---
  buffer<Scalar> face_r_coord;     // [N_faces]
  buffer<Scalar> face_sth;         // [N_faces]
  buffer<Scalar> face_cth;         // [N_faces]
  buffer<Scalar> face_alpha;       // [N_faces]
  buffer<Scalar> face_sq_gamma_beta_r;  // [N_faces]

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
#endif

 private:
  // Recompute circumcenter positions from persisted sphere data.
  // Returns (cx, cy, cz) arrays of size N_tri * N_r.
  void compute_circumcenters(std::vector<double>& cx,
                             std::vector<double>& cy,
                             std::vector<double>& cz) const;

  // Build edge-to-triangle adjacency from sphere data.
  // edge_tris[e] = {t0, t1} for each sphere edge.
  void build_edge_tris(
      std::vector<std::array<int, 2>>& edge_tris) const;

  // Compute metric-weighted Hodge stars.
  void compute_hodge_metric(const spherical_metric_t& metric,
                            const std::vector<double>& cx,
                            const std::vector<double>& cy,
                            const std::vector<double>& cz,
                            const std::vector<std::array<int, 2>>& edge_tris);
};

}  // namespace Aperture
