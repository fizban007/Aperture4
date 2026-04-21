#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/icosphere_topology.h"
#include <algorithm>
#include <cassert>
#include <set>

namespace Aperture {

namespace {

bool owns(cochain_type t, const prismatic_partition& p, int g) {
  switch (t) {
    case cochain_type::tri_face:  return p.owns_tri_face_cochain(g);
    case cochain_type::rect_face: return p.owns_rect_face_cochain(g);
    case cochain_type::h_edge:    return p.owns_h_edge_cochain(g);
    case cochain_type::v_edge:    return p.owns_v_edge_cochain(g);
    case cochain_type::vertex:    return p.owns_vertex_cochain(g);
  }
  return false;
}

}  // namespace

distributed_cochain_layout distributed_cochain_layout::build(
    cochain_type t, const prismatic_partition& part,
    const icosphere_topology* topo,
    std::initializer_list<const halo_plan*> plans) {
  distributed_cochain_layout out;
  out.m_global_size = global_cochain_size(t, part);

  // Caller is responsible for having attached a topology to `part` if
  // angular-axis ownership queries need it.  Document and trust.
  (void)topo;

  // ---- Collect owned global indices (ascending by construction) ----
  std::vector<int> owned;
  owned.reserve(out.m_global_size /
                std::max(1, part.n_angular_ranks * part.n_radial_ranks));
  for (int g = 0; g < out.m_global_size; ++g) {
    if (owns(t, part, g)) owned.push_back(g);
  }

  // ---- Collect ghost global indices from plans (dedup via std::set) ----
  std::set<int> ghost_set;
  for (const halo_plan* p : plans) {
    if (p == nullptr) continue;
    for (auto const& pe : p->peers) {
      for (int g : pe.recv_global_idx) {
        // Ownership & ghosthood should be disjoint; if a plan was
        // built consistently with this partition, no owned index
        // should appear in recv.  Check defensively.
        if (!owns(t, part, g)) ghost_set.insert(g);
      }
    }
  }

  // ---- Assemble local_to_global: owned | ghost ----
  out.m_n_owned = int(owned.size());
  out.m_n_ghost = int(ghost_set.size());
  out.m_local_to_global.reserve(out.m_n_owned + out.m_n_ghost);
  for (int g : owned) out.m_local_to_global.push_back(g);
  for (int g : ghost_set) out.m_local_to_global.push_back(g);

  // ---- Sorted view for global→local lookups ----
  const int n_local = out.local_size();
  out.m_sorted_global.resize(n_local);
  out.m_sorted_local.resize(n_local);
  for (int l = 0; l < n_local; ++l) {
    out.m_sorted_global[l] = out.m_local_to_global[l];
    out.m_sorted_local[l] = l;
  }
  // m_local_to_global is owned|ghost, each sorted; the concatenation is
  // sorted iff owned.back() < ghost_set.begin(), which isn't guaranteed
  // in general (an owned index might be larger than a ghost index).
  // So sort the paired view properly.
  std::vector<int> indices(n_local);
  for (int l = 0; l < n_local; ++l) indices[l] = l;
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
              return out.m_sorted_global[a] < out.m_sorted_global[b];
            });
  std::vector<int> sg(n_local), sl(n_local);
  for (int i = 0; i < n_local; ++i) {
    sg[i] = out.m_sorted_global[indices[i]];
    sl[i] = out.m_sorted_local[indices[i]];
  }
  out.m_sorted_global = std::move(sg);
  out.m_sorted_local = std::move(sl);

  return out;
}

int distributed_cochain_layout::to_local(int global_idx) const {
  auto it = std::lower_bound(m_sorted_global.begin(), m_sorted_global.end(),
                             global_idx);
  if (it == m_sorted_global.end() || *it != global_idx) return -1;
  return m_sorted_local[it - m_sorted_global.begin()];
}

halo_plan distributed_cochain_layout::localize(
    const halo_plan& global_plan) const {
  halo_plan out;
  out.peers.reserve(global_plan.peers.size());
  for (auto const& pe : global_plan.peers) {
    halo_plan::peer_entry local_pe;
    local_pe.peer_rank = pe.peer_rank;
    local_pe.send_global_idx.reserve(pe.send_global_idx.size());
    local_pe.recv_global_idx.reserve(pe.recv_global_idx.size());
    for (int g : pe.send_global_idx) {
      int l = to_local(g);
      assert(l >= 0 && "send index must be owned by this rank");
      local_pe.send_global_idx.push_back(l);
    }
    for (int g : pe.recv_global_idx) {
      int l = to_local(g);
      assert(l >= 0 && "recv index must be a ghost on this rank");
      local_pe.recv_global_idx.push_back(l);
    }
    out.peers.push_back(std::move(local_pe));
  }
  return out;
}

}  // namespace Aperture
