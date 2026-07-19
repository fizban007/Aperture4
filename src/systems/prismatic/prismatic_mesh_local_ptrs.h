#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_d1_local.h"
#include "systems/prismatic/prismatic_mesh_local.h"
#include "systems/prismatic/prismatic_mesh_partition.h"

namespace Aperture {

// =========================================================================
// Phase 4.1b.0 — POD pointer bundle for the per-rank local mesh data.
//
// Mirrors prismatic_mesh_ptrs but points into the LOCAL (owned + halo)
// arrays of prismatic_mesh_local and prismatic_d1_local.  Pass by value
// into GPU lambdas exactly like prismatic_mesh_ptrs.
//
// Index conventions (see prismatic_d1_local.h):
//   - every per-cochain array is local-indexed in its own cochain
//     layout; owned entries come first ([0, n_owned_X)), halo ghosts
//     after ([n_owned_X, n_local_X));
//   - d1/d1t block ROW indices are owned-local, COLUMN indices are
//     full-local (owned + halo) of the column-side type;
//   - h_edge_v0/v1 VALUES are GLOBAL vertex indices (translate via the
//     vertex layout at the call site if needed — sphere-side data is
//     replicated, so most kernels never need to).
//
// The flat dec_field_solver needs only the base mesh data; the GR
// metric fields (prismatic_mesh_metric_local) are not included here —
// a metric-ptrs extension can mirror this pattern if the GR solver is
// revived.
// =========================================================================
struct prismatic_mesh_local_ptrs {
  // Layout sizes (owned first, then owned + halo).
  int n_owned_tri = 0, n_local_tri = 0;
  int n_owned_rect = 0, n_local_rect = 0;
  int n_owned_he = 0, n_local_he = 0;
  int n_owned_ve = 0, n_local_ve = 0;
  int n_owned_v = 0, n_local_v = 0;

  // ---- Face-side mesh data ----
  const Scalar* tri_face_area = nullptr;
  const Scalar* tri_face_hodge2 = nullptr;
  const int* tri_face_boundary = nullptr;
  const int* tri_face_radial_layer = nullptr;
  const Scalar* rect_face_area = nullptr;
  const Scalar* rect_face_hodge2 = nullptr;
  const int* rect_face_boundary = nullptr;
  const int* rect_face_radial_layer = nullptr;

  // ---- Edge-side mesh data ----
  const Scalar* h_edge_length = nullptr;
  const Scalar* h_edge_hodge1_inv = nullptr;
  const gidx_t* h_edge_v0 = nullptr;  // GLOBAL 3D vertex ids (64-bit)
  const gidx_t* h_edge_v1 = nullptr;
  const int* h_edge_boundary = nullptr;
  const int* h_edge_radial_layer = nullptr;
  const Scalar* v_edge_length = nullptr;
  const Scalar* v_edge_hodge1_inv = nullptr;
  const gidx_t* v_edge_v0 = nullptr;
  const gidx_t* v_edge_v1 = nullptr;
  const int* v_edge_boundary = nullptr;
  const int* v_edge_radial_layer = nullptr;

  // ---- Vertex-side ----
  const Scalar* vert_r = nullptr;
  const Scalar* vert_theta = nullptr;
  const Scalar* vert_phi = nullptr;

  // ---- Local -> global index maps ----
  // Global index within the cochain's OWN global range (tri faces,
  // rect faces, h edges, v edges each start at 0).  Used by the
  // rarely-executed geometry kernels (ICs, inner-BC quadratures) to
  // reach the replicated global mesh via prismatic_mesh_ptrs.
  const gidx_t* tri_face_l2g = nullptr;
  const gidx_t* rect_face_l2g = nullptr;
  const gidx_t* h_edge_l2g = nullptr;
  const gidx_t* v_edge_l2g = nullptr;

  // ---- d1 / d1^T sparse blocks ----
  const int* d1_tri_h_row = nullptr;
  const int* d1_tri_h_col = nullptr;
  const Scalar* d1_tri_h_val = nullptr;
  const int* d1_rect_h_row = nullptr;
  const int* d1_rect_h_col = nullptr;
  const Scalar* d1_rect_h_val = nullptr;
  const int* d1_rect_v_row = nullptr;
  const int* d1_rect_v_col = nullptr;
  const Scalar* d1_rect_v_val = nullptr;
  const int* d1t_h_tri_row = nullptr;
  const int* d1t_h_tri_col = nullptr;
  const Scalar* d1t_h_tri_val = nullptr;
  const int* d1t_h_rect_row = nullptr;
  const int* d1t_h_rect_col = nullptr;
  const Scalar* d1t_h_rect_val = nullptr;
  const int* d1t_v_rect_row = nullptr;
  const int* d1t_v_rect_col = nullptr;
  const Scalar* d1t_v_rect_val = nullptr;
};

// Assemble host- or device-side pointer bundles.  The mesh_local and
// d1_local objects must outlive the returned struct; for device
// pointers their buffers must have been built with a device-capable
// MemType and copy_to_device() must have been called.
prismatic_mesh_local_ptrs make_local_ptrs_host(
    const prismatic_mesh_local& ml, const prismatic_d1_local& d1);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_mesh_local_ptrs make_local_ptrs_dev(
    const prismatic_mesh_local& ml, const prismatic_d1_local& d1);
#endif

}  // namespace Aperture
