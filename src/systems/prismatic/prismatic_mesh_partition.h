#pragma once

#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_partition.h"

namespace Aperture {

// =========================================================================
// Prismatic-mesh partition bundle.
//
// For a given (partition, topology), pre-computes everything downstream
// consumers need to localize a global mesh:
//
//   - A distributed_cochain_layout for each of the 5 cochain types,
//     giving the local-size buffer dimensions and the local ↔ global
//     index mapping.
//
//   - Angular and radial halo plans for each cochain type, both in
//     global indices (the original form produced by the plan builders)
//     AND in local indices (translated via layout.localize()).
//
// Downstream usage:
//
//   1. The solver / mesh-metric allocates local-sized buffers using
//      layout.local_size() and populates them by running the existing
//      global build and then copying with copy_global_to_local.
//
//   2. At simulation time, halo exchanges use the localized plans
//      directly against the local buffers via either the in_process
//      backend or the mpi_halo_backend.
//
// This struct is a pure-data bundle; all the real work is in the
// members' build() methods.  Copying / moving is allowed (shallow
// copy of vectors is fine for size — each cochain layout is ~O(local_size)
// ints).
// =========================================================================
class prismatic_mesh_partition {
 public:
  prismatic_mesh_partition() = default;

  // Build from a partition (with its topology attached) and the
  // icosphere topology it points to.  Populates all 5 layouts and all
  // 10 halo plans (5 cochain × 2 axes).
  static prismatic_mesh_partition build(const prismatic_partition& part,
                                        const icosphere_topology& topo);

  // ---- Layout accessors ----
  const distributed_cochain_layout& layout(cochain_type t) const {
    switch (t) {
      case cochain_type::tri_face:  return m_tri_face;
      case cochain_type::rect_face: return m_rect_face;
      case cochain_type::h_edge:    return m_h_edge;
      case cochain_type::v_edge:    return m_v_edge;
      case cochain_type::vertex:    return m_vertex;
    }
    return m_tri_face;  // unreachable
  }

  // ---- Plan accessors (global-indexed) ----
  const halo_plan& angular_plan_global(cochain_type t) const;
  const halo_plan& radial_plan_global(cochain_type t) const;

  // ---- Plan accessors (local-indexed, used for halo exchange) ----
  const halo_plan& angular_plan_local(cochain_type t) const;
  const halo_plan& radial_plan_local(cochain_type t) const;

  // ---- Environment ----
  const prismatic_partition& partition() const { return m_partition; }
  const icosphere_topology& topology() const { return *m_topology; }

 private:
  prismatic_partition m_partition;
  const icosphere_topology* m_topology = nullptr;

  distributed_cochain_layout m_tri_face;
  distributed_cochain_layout m_rect_face;
  distributed_cochain_layout m_h_edge;
  distributed_cochain_layout m_v_edge;
  distributed_cochain_layout m_vertex;

  // Per-cochain-type plans, stored both in global- and local-index form
  // (the latter resolved via layout.localize() at build time).
  struct plans_pair {
    halo_plan angular_global, angular_local;
    halo_plan radial_global, radial_local;
  };
  plans_pair m_tri_face_plans;
  plans_pair m_rect_face_plans;
  plans_pair m_h_edge_plans;
  plans_pair m_v_edge_plans;
  plans_pair m_vertex_plans;

  const plans_pair& plans(cochain_type t) const {
    switch (t) {
      case cochain_type::tri_face:  return m_tri_face_plans;
      case cochain_type::rect_face: return m_rect_face_plans;
      case cochain_type::h_edge:    return m_h_edge_plans;
      case cochain_type::v_edge:    return m_v_edge_plans;
      case cochain_type::vertex:    return m_vertex_plans;
    }
    return m_tri_face_plans;  // unreachable
  }
};

}  // namespace Aperture
