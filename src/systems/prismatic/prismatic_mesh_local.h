#pragma once

#include "core/buffer.hpp"
#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_partition.h"

namespace Aperture {

class prismatic_mesh;

// =========================================================================
// Prismatic mesh — local-sized mesh-data buffers for one rank's partition.
//
// Copies the per-cochain mesh data from a globally-constructed
// prismatic_mesh into arrays sized to the rank's partition (owned +
// halo ghosts).  The input mesh stays in its original global form;
// this class holds an independent set of local buffers.
//
// Split conventions: the mesh stores edges in one flat array with
// horizontal edges first (global indices [0, N_h_edges)) followed by
// vertical edges ([N_h_edges, N_edges)); similarly faces are tri-first
// then rect.  The distributed layouts treat h_edge / v_edge and
// tri_face / rect_face as separate cochain types, each with its own
// local index space.  So this class exposes:
//
//   h_edge_*    — sized to layout(h_edge).local_size()
//   v_edge_*    — sized to layout(v_edge).local_size()
//   tri_face_*  — sized to layout(tri_face).local_size()
//   rect_face_* — sized to layout(rect_face).local_size()
//   vertex_*    — sized to layout(vertex).local_size()
//
// The mesh's original flat indexing (h_edge_idx, v_edge_idx, tri_face_
// idx, rect_face_idx) is replaced by local indexing via the layouts
// owned by the underlying prismatic_mesh_partition.
//
// This class is a pure-data container; the build() factory does all
// the work.  Phase 4.1a.1: covers face_area, hodge2, edge_length,
// hodge1_inv, vertex coords.  CSR (d1, d1t) and remaining per-element
// buffers come in follow-up steps.
// =========================================================================
class prismatic_mesh_local {
 public:
  // mem_type host_only (default) keeps the 4.1a host-staged behavior;
  // pass host_device (and call copy_to_device()) for the 4.1b solver
  // kernels.
  static prismatic_mesh_local build(const prismatic_mesh& mesh,
                                    const prismatic_mesh_partition& mp,
                                    MemType mem_type = MemType::host_only);

  const prismatic_mesh_partition& partition() const { return *m_partition; }

  // Copy every local buffer to the device (no-op for host_only).
  void copy_to_device();

  // ---- Face-side ----
  buffer<Scalar> tri_face_area;
  buffer<Scalar> tri_face_hodge2;
  buffer<int>       tri_face_boundary;
  buffer<int>       tri_face_radial_layer;

  buffer<Scalar> rect_face_area;
  buffer<Scalar> rect_face_hodge2;
  buffer<int>       rect_face_boundary;
  buffer<int>       rect_face_radial_layer;

  // ---- Edge-side ----
  buffer<Scalar> h_edge_length;
  buffer<Scalar> h_edge_hodge1_inv;
  buffer<int>       h_edge_v0;
  buffer<int>       h_edge_v1;
  buffer<int>       h_edge_boundary;
  buffer<int>       h_edge_radial_layer;

  buffer<Scalar> v_edge_length;
  buffer<Scalar> v_edge_hodge1_inv;
  buffer<int>       v_edge_v0;
  buffer<int>       v_edge_v1;
  buffer<int>       v_edge_boundary;
  buffer<int>       v_edge_radial_layer;

  // ---- Vertex-side ----
  buffer<Scalar> vert_r;
  buffer<Scalar> vert_theta;
  buffer<Scalar> vert_phi;

 private:
  const prismatic_mesh_partition* m_partition = nullptr;
};

}  // namespace Aperture
