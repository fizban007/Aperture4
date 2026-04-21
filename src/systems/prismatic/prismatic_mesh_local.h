#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include <vector>

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
  static prismatic_mesh_local build(const prismatic_mesh& mesh,
                                    const prismatic_mesh_partition& mp);

  const prismatic_mesh_partition& partition() const { return *m_partition; }

  // ---- Face-side ----
  std::vector<Scalar> tri_face_area;
  std::vector<Scalar> tri_face_hodge2;
  std::vector<int>    tri_face_boundary;
  std::vector<int>    tri_face_radial_layer;

  std::vector<Scalar> rect_face_area;
  std::vector<Scalar> rect_face_hodge2;
  std::vector<int>    rect_face_boundary;
  std::vector<int>    rect_face_radial_layer;

  // ---- Edge-side ----
  std::vector<Scalar> h_edge_length;
  std::vector<Scalar> h_edge_hodge1_inv;
  std::vector<int>    h_edge_v0;
  std::vector<int>    h_edge_v1;
  std::vector<int>    h_edge_boundary;
  std::vector<int>    h_edge_radial_layer;

  std::vector<Scalar> v_edge_length;
  std::vector<Scalar> v_edge_hodge1_inv;
  std::vector<int>    v_edge_v0;
  std::vector<int>    v_edge_v1;
  std::vector<int>    v_edge_boundary;
  std::vector<int>    v_edge_radial_layer;

  // ---- Vertex-side ----
  std::vector<Scalar> vert_r;
  std::vector<Scalar> vert_theta;
  std::vector<Scalar> vert_phi;

 private:
  const prismatic_mesh_partition* m_partition = nullptr;
};

}  // namespace Aperture
