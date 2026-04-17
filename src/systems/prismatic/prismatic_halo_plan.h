#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_partition.h"
#include <vector>

namespace Aperture {

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
// In-process backend (Phase 2 test fixture).
//
// Runs multiple "ranks" in a single process.  Each rank registers its
// buffer with the registry, then exchange() copies from peer buffers
// directly.  This is a stand-in for MPI_Neighbor_alltoallv until the
// MPI backend lands in Phase 3.
//
// Not thread-safe, not meant for production — purely a test harness to
// validate plan semantics before plumbing MPI.
// =========================================================================
class in_process_halo_backend {
 public:
  // Register the buffer for a given rank.  Called once per rank at
  // fixture setup.  The backend does NOT take ownership; the caller
  // must keep the buffer alive for the lifetime of the backend.
  void register_rank(int rank, Scalar* buffer);

  // Execute a halo exchange.  For each peer in the plan, the peer's
  // registered buffer is read for the send data, and the local buffer
  // receives into its recv indices.
  //
  // This is a two-way operation from the perspective of both sides, but
  // since it's in-process, the single exchange() call for `my_rank`
  // does only the "pull" side: it writes `my_buffer[recv_idx]` from
  // `peer_buffer[send_idx]`.  To fully update all ranks, call exchange()
  // once per rank.  (MPI will do both sides in one collective.)
  void exchange(int my_rank, Scalar* my_buffer, const halo_plan& plan);

  // Convenience: run exchange() on every registered rank in round-robin
  // using per-rank plans passed in as a parallel vector.
  void exchange_all(const std::vector<halo_plan>& plans);

  // Number of registered ranks.
  int size() const { return int(m_rank_buffers.size()); }

 private:
  std::vector<Scalar*> m_rank_buffers;
};

}  // namespace Aperture
