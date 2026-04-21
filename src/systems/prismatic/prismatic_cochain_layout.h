#pragma once

#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_partition.h"
#include <vector>

namespace Aperture {

class icosphere_topology;

// =========================================================================
// Distributed cochain layout.
//
// For a given (partition, cochain_type, topology), computes the mapping
// between global cochain indices and local buffer offsets on this rank.
//
// Buffer layout (local indices):
//   [0, n_owned)                  — owned cells, ascending global order.
//   [n_owned, n_owned + n_ghost)  — ghost cells, ascending global order.
//
// Per-rank buffer size becomes `local_size() = n_owned + n_ghost` instead
// of the full `global_size()`, which is the point: at L=8 / N_r=200 the
// global size is ~10^8 per cochain and cannot fit on a single GPU across
// 100-ish ranks.  With this layout a typical rank holds ~10^6 elements
// per cochain.
//
// Ghost indices are gathered from the recv_global_idx fields of the
// halo plans passed to build().  Typical usage: pass both the angular
// plan (for this partition+cochain) and the radial plan.  Other sources
// (e.g. a second cochain's plan) can be passed too if desired.
//
// Phase 4.0 (this file): layout + plan translation only — the global-
// indexed plans produced by build_angular_halo_plan / build_radial_halo_
// plan get translated via localize() into local-indexed plans suitable
// for local-sized buffers.  The mpi_halo_backend and in_process_halo_
// backend both accept local-indexed plans interchangeably with global-
// indexed ones; the backend is layout-agnostic.
// =========================================================================
class distributed_cochain_layout {
 public:
  distributed_cochain_layout() = default;

  // Construct the layout for (t, part, topo), gathering ghost indices
  // from the provided halo plans.  Pass all plans whose recv_global_idx
  // will address this buffer — typically one angular + one radial.
  //
  // Plan peer_rank values are not inspected; only recv_global_idx is.
  //
  // Owned indices are discovered by scanning [0, global_size) and
  // querying partition ownership.  O(global_size) build cost; acceptable
  // as a one-time setup (cost dominated by the actual simulation loop).
  static distributed_cochain_layout build(
      cochain_type t, const prismatic_partition& part,
      const icosphere_topology* topo,
      std::initializer_list<const halo_plan*> plans);

  // ---- Sizes ----
  int global_size() const { return m_global_size; }
  int owned_size() const { return m_n_owned; }
  int ghost_size() const { return m_n_ghost; }
  int local_size() const { return m_n_owned + m_n_ghost; }

  // ---- Mappings ----
  int to_global(int local_idx) const {
    return m_local_to_global[local_idx];
  }
  // Returns -1 if this rank doesn't have the given global index.
  int to_local(int global_idx) const;

  // ---- Plan translation ----
  // Given a halo_plan with global indices in its send/recv fields,
  // produce a new halo_plan with the same peer list and peer_rank
  // values, but with send/recv fields translated to LOCAL indices.
  // Every index in the input must be present in this layout (either as
  // owned or as ghost); otherwise the translation asserts.
  halo_plan localize(const halo_plan& global_plan) const;

 private:
  int m_global_size = 0;
  int m_n_owned = 0;
  int m_n_ghost = 0;

  // local_to_global[l] gives the global index of local element l.
  std::vector<int> m_local_to_global;

  // Sorted ascending, same content as m_local_to_global, for binary-
  // search lookups in to_local().  Using a sorted vector + lower_bound
  // is ~3x more memory-efficient than unordered_map and plenty fast
  // enough for build-time plan translation.
  //
  // Layout invariant: m_local_to_global IS already sorted ascending
  // (owned then ghost, each sorted), so this is just a second view of
  // the same data — kept separate so we can add non-sorted local index
  // orderings later (e.g., owned block + interleaved ghost) without
  // changing the lookup path.
  std::vector<int> m_sorted_global;   // sorted globals
  std::vector<int> m_sorted_local;    // parallel local indices
};

}  // namespace Aperture
