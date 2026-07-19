#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_partition.h"
#include <vector>

namespace Aperture {

class icosphere_topology;

// =========================================================================
// Halo exchange plan.
//
// Describes, for one cochain type on one axis (radial or angular), the set
// of global indices this rank must send to each peer and receive from each
// peer to populate its ghost region.
//
// The plan is pure data — constructed once per (partition, cochain-type,
// axis) and reused across time steps.  The actual data movement is done by
// a backend (in-process for Phase 2 testing, MPI in Phase 3).
//
// All indices are GLOBAL cochain indices (stable across partition shape).
// A Phase 3 refactor will translate these to local buffer offsets at plan-
// build time; for now buffers are global-sized and the global index is the
// direct buffer offset.
// =========================================================================
struct halo_plan {
  // One entry per peer this rank exchanges with.
  struct peer_entry {
    // Peer's rank in the relevant sub-communicator (comm_radial or
    // comm_angular).  For Phase 2 in-process testing, this is just an
    // index into a rank registry; for Phase 3 it's MPI_Comm_rank in
    // the associated communicator.
    int peer_rank = -1;

    // Global cochain indices we SEND to the peer.
    std::vector<int> send_global_idx;

    // Global cochain indices we RECEIVE from the peer (written into the
    // local buffer at exactly these global indices).
    std::vector<int> recv_global_idx;
  };
  std::vector<peer_entry> peers;

  // Total send/receive message sizes (for buffer pre-allocation).
  int total_send() const {
    int n = 0;
    for (auto const& p : peers) n += int(p.send_global_idx.size());
    return n;
  }
  int total_recv() const {
    int n = 0;
    for (auto const& p : peers) n += int(p.recv_global_idx.size());
    return n;
  }
};

// =========================================================================
// Cochain-type tag.
//
// The halo plan layout and global indexing depend on which cochain we're
// exchanging.  Ordered to match the global-index scheme documented in
// PARALLELIZATION_PLAN.md.
// =========================================================================
enum class cochain_type {
  tri_face,    // primal 2-cochain on shell tri faces    (shape: (N_r+1) * N_tri)
  rect_face,   // primal 2-cochain on radial rect faces  (shape: N_r * N_edge_s)
  h_edge,      // primal 1-cochain on shell h edges      (shape: (N_r+1) * N_edge_s)
  v_edge,      // primal 1-cochain on radial v edges     (shape: N_r * N_vert_s)
  vertex,      // primal 0-cochain on vertices           (shape: (N_r+1) * N_vert_s)
};

// Return the total cochain size for the full global mesh.
int global_cochain_size(cochain_type t, const prismatic_partition& p);

// =========================================================================
// Radial halo plan builder.
//
// Builds the radial axis halo plan for a single cochain type.  The plan
// has at most two peer entries (lower radial neighbor, upper radial
// neighbor); single-rank or boundary-rank plans are empty.
//
// Convention:
//   shell k is owned by the rank whose shell_k_lo ≤ k < shell_k_hi.
//   For tri faces / h-edges / vertices, data at shell k goes to the
//   rank owning shell k.
//   For rect faces / v-edges, the element lives between shells k and k+1;
//   we assign ownership to the rank owning shell k (the lower end).
//
// The plan's send list is populated with all global indices at the
// rank's owned shells immediately adjacent to the radial neighbor.
// The recv list mirrors this from the neighbor's perspective.
// =========================================================================
halo_plan build_radial_halo_plan(cochain_type t,
                                  const prismatic_partition& self);

// =========================================================================
// Angular halo plan builder.
//
// Builds the angular axis halo plan for a single cochain type on a rank
// owning exactly one ico-face (the `combined()` or `ico_face_angular()`
// configuration).  A rank has up to 9 angular peers: 3 edge-neighbors
// (shared ico-edge) + 6 vertex-diagonal-neighbors (shared valence-5
// corner only).
//
// The send/recv index sets are derived from the topology and the
// ownership rule "lowest-index incident ico-face owns":
//
//   tri_face cochain: for each ico-boundary sphere-edge e incident to
//     F, the two adjacent tri faces straddle the boundary (one per
//     incident ico-face).  If F owns e, F halos the peer's tri face
//     (for computing H_aux on it).  If the peer owns e, F sends its
//     own tri face (peer halos from F).
//
//   h_edge / rect_face cochains (data on sphere-edges): if F owns an
//     ico-boundary edge, F sends to the non-F incident.  If F doesn't
//     own it, F receives from the owner.
//
//   vertex / v_edge cochains (data on sphere-vertices): at ico-edge
//     vertices (valence 2) and valence-5 corners (valence 5), F sends
//     if owner or recvs from owner if non-owner.  Non-owners exchange
//     only with the owner, not with each other.
// =========================================================================
halo_plan build_angular_halo_plan(cochain_type t,
                                   const prismatic_partition& self,
                                   const icosphere_topology& topo);

// =========================================================================
// Generic angular halo plan builder (Phase 7A.3).
//
// Works for ANY canonical-rank-order partition (angular_units/combined
// factories): arbitrary patch level m and angular rank count A, including
// multi-face bands, sub-face units, valence-5 corners spanning up to 5
// distinct ranks, and ordinary patch corners at valence-6 vertices.
// Peer ranks are ANGULAR ranks under the path-ordered A-way split (not
// ico-face indices — use the legacy builder above for identity-ordered
// partitions until 7A.4 removes them).
//
// Per-element ghost rules (solver depth class, d1/d1t stencils; each
// reduces exactly to the legacy per-face rule at m=0):
//   tri_face:  recv non-owned tris adjacent (across a sphere-edge) to
//              self-owned sphere-edges; send owned tris to the owner
//              rank of any adjacent non-self-owned sphere-edge.
//   h_edge:    recv non-owned sphere-edges adjacent to self-owned tris;
//              send owned edges to owner ranks of adjacent tris.
//   rect_face: h_edge rule PLUS endpoint-vertex owners — the v_edge
//              update at an owned vertex reads the full fan of rect
//              faces, so a rect face is also ghosted to the owner of
//              each endpoint vertex (this uniformly subsumes the legacy
//              valence-5 corner fan patch).
//   vertex /
//   v_edge:    recv non-owned sphere-vertices with a self-owned fan
//              tri; send owned vertices to owner ranks of fan tris.
//
// Wire order: per-peer send/recv lists are sorted ascending by global
// cochain index and deduplicated — both peers derive identical lists
// independently, with no matched-iteration requirement.
//
// The ghost-set depth class follows PHASE_7_SCALABLE_PIC_PLAN.md F4;
// `pic` (1-ring T_halo + shells ±1 superset) lands in Phase 7B.
// =========================================================================
enum class halo_depth { solver, pic };

halo_plan build_angular_halo_plan_units(
    cochain_type t, const prismatic_partition& self,
    const icosphere_topology& topo, halo_depth depth = halo_depth::solver);

// =========================================================================
// In-process backend (test fixture).
//
// Runs multiple "ranks" in a single process and emulates MPI point-to-
// point halo exchange.  Each rank registers a buffer; exchange_all()
// matches each rank's recv entry with the peer's corresponding send
// entry (paired by position in the plan, which is the same semantics
// MPI_Isend / MPI_Irecv use) and copies value-by-value.
//
// Works with both global-indexed and local-indexed plans:
//   - Global-indexed: sender's send_global_idx[i] == receiver's
//     recv_global_idx[i] (same global index), so buf_b[send_idx] at B
//     and buf_a[recv_idx] at A have matching interpretations.
//   - Local-indexed: local indices differ across ranks; the copy uses
//     the sender's own local index to READ and the receiver's own
//     local index to WRITE, which is exactly what MPI_Isend/MPI_Irecv
//     do implicitly through pack/unpack.
//
// Plan construction must emit send and recv entries in a MATCHING order
// on the two sides (so the i-th sender entry matches the i-th receiver
// entry).  build_angular_halo_plan and build_radial_halo_plan satisfy
// this by construction (both sides iterate the same topology tables in
// the same order).
// =========================================================================
class in_process_halo_backend {
 public:
  // Register the buffer for a given rank.
  void register_rank(int rank, Scalar* buffer);

  // Legacy single-rank exchange: assumes global indexing so that reading
  // peer_buf[recv_idx] gives the right value.  Kept for the existing
  // Phase 2 tests; new code should prefer exchange_all() which handles
  // both global and local indexing correctly.
  void exchange(int my_rank, Scalar* my_buffer, const halo_plan& plan);

  // Collective: matches each rank's recv entry with the corresponding
  // peer's send entry by position, then copies.  Works with both
  // global-indexed and local-indexed plans.
  void exchange_all(const std::vector<halo_plan>& plans);

  int size() const { return int(m_rank_buffers.size()); }

 private:
  std::vector<Scalar*> m_rank_buffers;
};

}  // namespace Aperture
