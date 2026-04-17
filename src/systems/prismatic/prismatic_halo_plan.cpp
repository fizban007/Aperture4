#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/icosphere_topology.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <utility>

namespace Aperture {

// =========================================================================
// Global cochain sizes.
// =========================================================================
int global_cochain_size(cochain_type t, const prismatic_partition& p) {
  switch (t) {
    case cochain_type::tri_face:  return p.N_tri_faces_global;
    case cochain_type::rect_face: return p.N_rect_faces_global;
    case cochain_type::h_edge:    return p.N_h_edges_global;
    case cochain_type::v_edge:    return p.N_v_edges_global;
    case cochain_type::vertex:    return p.N_verts_global;
  }
  return 0;  // unreachable
}

namespace {

// Per-shell (or per-slab) cochain width: how many scalars live on a
// single shell (or slab) interface for this cochain type.
int per_level_width(cochain_type t, const prismatic_partition& p) {
  switch (t) {
    case cochain_type::tri_face:  return p.N_tri_global;
    case cochain_type::rect_face: return p.N_edge_s_global;
    case cochain_type::h_edge:    return p.N_edge_s_global;
    case cochain_type::v_edge:    return p.N_vert_s_global;
    case cochain_type::vertex:    return p.N_vert_s_global;
  }
  return 0;
}

// True if the cochain lives ON a shell (tri_face, h_edge, vertex), false
// if it lives in a slab BETWEEN shells (rect_face, v_edge).  The two
// classes have different radial halo patterns because shell ownership
// and slab ownership are offset by one under the "slab k owned by the
// rank owning shell k" convention.
bool lives_on_shell(cochain_type t) {
  switch (t) {
    case cochain_type::tri_face:
    case cochain_type::h_edge:
    case cochain_type::vertex:
      return true;
    case cochain_type::rect_face:
    case cochain_type::v_edge:
      return false;
  }
  return true;
}

// Fill idx[0..width) with [base * width, (base + 1) * width).
void contiguous_range(int base, int width, std::vector<int>& out) {
  out.resize(width);
  for (int i = 0; i < width; ++i) out[i] = base * width + i;
}

}  // namespace

// =========================================================================
// Radial halo plan.
//
// Ownership convention:
//   - Shell k owned by the rank whose [shell_k_lo, shell_k_hi) contains k.
//   - Slab k (spans shells k, k+1) owned by the rank owning shell k.
//     Thus a rank owns slabs [shell_k_lo, shell_k_hi), where the last
//     slab straddles the upper partition boundary and needs shell k_hi
//     as an upper-ghost input.
//
// For shell-living cochains (tri_face, h_edge, vertex):
//   - Lower ghost: shell k_lo - 1.         Recv from lower.
//   - Upper ghost: shell k_hi.             Recv from upper.
//   - Send to lower: shell k_lo (lower's upper ghost).
//   - Send to upper: shell k_hi - 1 (upper's lower ghost).
//
// For slab-living cochains (rect_face, v_edge):
//   - Lower ghost: slab k_lo - 1.          Recv from lower.
//   - No upper ghost slab (slab k_hi is owned by upper neighbor).
//   - Send to lower: nothing (lower's top owned slab is k_lo - 1).
//   - Send to upper: slab k_hi - 1 (upper's lower ghost slab).
//
// Result: two-peer plans for shell-living cochains, and two-peer plans
// for slab-living cochains but with one empty send or one empty recv in
// each direction.  Boundary-rank plans drop the missing-neighbor entry.
// =========================================================================
halo_plan build_radial_halo_plan(cochain_type t,
                                  const prismatic_partition& self) {
  halo_plan out;
  if (self.n_radial_ranks <= 1) return out;

  const int width = per_level_width(t, self);
  const int kl = self.shell_k_lo;
  const int kh = self.shell_k_hi;
  const int r = self.radial_rank;
  const bool has_lower = r > 0;
  const bool has_upper = r < self.n_radial_ranks - 1;
  const bool shell_cochain = lives_on_shell(t);

  if (has_lower) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r - 1;
    if (shell_cochain) {
      contiguous_range(kl,     width, pe.send_global_idx);
      contiguous_range(kl - 1, width, pe.recv_global_idx);
    } else {
      // Slab cochain: nothing to send to lower; recv slab k_lo - 1.
      contiguous_range(kl - 1, width, pe.recv_global_idx);
    }
    out.peers.push_back(std::move(pe));
  }

  if (has_upper) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r + 1;
    if (shell_cochain) {
      contiguous_range(kh - 1, width, pe.send_global_idx);
      contiguous_range(kh,     width, pe.recv_global_idx);
    } else {
      // Slab cochain: send slab k_hi - 1 to upper's lower ghost; recv
      // nothing from upper.
      contiguous_range(kh - 1, width, pe.send_global_idx);
    }
    out.peers.push_back(std::move(pe));
  }

  return out;
}

// =========================================================================
// Angular halo plan.
// =========================================================================
halo_plan build_angular_halo_plan(cochain_type t,
                                   const prismatic_partition& self,
                                   const icosphere_topology& topo) {
  halo_plan out;

  // One ico-face per rank is the target configuration for the 20-way
  // angular decomposition.  If more ico-faces are owned (e.g. single-
  // rank fallback), there are no angular halos to build.
  if (self.ico_face_hi - self.ico_face_lo != 1) return out;
  const int F = self.ico_face_lo;

  const int pow4L = topo.N_tri() / 20;

  // Per-cochain data: width per k-level, and whether the level index is
  // a shell (valid 0..N_r) or a slab (valid 0..N_r-1).
  int width = 0;
  bool shell_cochain = true;
  switch (t) {
    case cochain_type::tri_face:
      width = topo.N_tri();
      shell_cochain = true;
      break;
    case cochain_type::h_edge:
      width = topo.N_edge_s();
      shell_cochain = true;
      break;
    case cochain_type::rect_face:
      width = topo.N_edge_s();
      shell_cochain = false;
      break;
    case cochain_type::v_edge:
      width = topo.N_vert_s();
      shell_cochain = false;
      break;
    case cochain_type::vertex:
      width = topo.N_vert_s();
      shell_cochain = true;
      break;
  }
  const int k_lo = self.shell_k_lo;
  const int k_hi = shell_cochain
                       ? self.shell_k_hi
                       : std::min(self.shell_k_hi, self.N_r_global);

  std::map<int, std::vector<int>> recv_per_peer;
  std::map<int, std::vector<int>> send_per_peer;

  // --------------------------------------------------------------------
  // tri_face: halo is the "other" adjacent tri face across a boundary
  // sphere-edge incident to F.  Direction depends on who owns the edge.
  // --------------------------------------------------------------------
  if (t == cochain_type::tri_face) {
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      if (topo.edge_valence(e) < 2) continue;
      const int* incs = topo.edge_ico_faces(e);
      if (incs[0] != F && incs[1] != F) continue;
      const int G = (incs[0] == F) ? incs[1] : incs[0];
      const int owner = incs[0];  // incidents sorted ascending, so incs[0] is min

      const int tri_a = topo.edge_tri_a(e);
      const int tri_b = topo.edge_tri_b(e);
      const int ico_a = tri_a / pow4L;
      const int tri_in_F = (ico_a == F) ? tri_a : tri_b;
      const int tri_in_G = (ico_a == F) ? tri_b : tri_a;

      if (F == owner) {
        // F owns e → halos the adjacent tri in G.
        for (int k = k_lo; k < k_hi; ++k) {
          recv_per_peer[G].push_back(k * width + tri_in_G);
        }
      } else {
        // G owns e → sends our adjacent tri (in F) to G.
        for (int k = k_lo; k < k_hi; ++k) {
          send_per_peer[G].push_back(k * width + tri_in_F);
        }
      }
    }
  }

  // --------------------------------------------------------------------
  // h_edge and rect_face: data lives on sphere-edges.  If F owns an
  // ico-boundary edge, send to the non-F incident; otherwise recv.
  // --------------------------------------------------------------------
  else if (t == cochain_type::h_edge || t == cochain_type::rect_face) {
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      if (topo.edge_valence(e) < 2) continue;
      const int* incs = topo.edge_ico_faces(e);
      if (incs[0] != F && incs[1] != F) continue;
      const int G = (incs[0] == F) ? incs[1] : incs[0];
      const int owner = incs[0];

      if (F == owner) {
        for (int k = k_lo; k < k_hi; ++k) {
          send_per_peer[G].push_back(k * width + e);
        }
      } else {
        for (int k = k_lo; k < k_hi; ++k) {
          recv_per_peer[G].push_back(k * width + e);
        }
      }
    }
  }

  // --------------------------------------------------------------------
  // vertex and v_edge: data lives on sphere-vertices.  At a vertex F
  // is incident to, F sends to every other incident if it owns, or
  // receives from the single owner if it doesn't.  Non-owners do NOT
  // exchange with each other.
  // --------------------------------------------------------------------
  else {  // cochain_type::vertex or cochain_type::v_edge
    for (int v = 0; v < topo.N_vert_s(); ++v) {
      const int val = topo.vertex_valence(v);
      if (val < 2) continue;  // interior to one ico-face
      const int* incs = topo.vertex_ico_faces(v);
      bool F_incident = false;
      for (int j = 0; j < val; ++j)
        if (incs[j] == F) {
          F_incident = true;
          break;
        }
      if (!F_incident) continue;
      const int owner = incs[0];

      if (F == owner) {
        // Send to each non-F incident.
        for (int j = 0; j < val; ++j) {
          if (incs[j] == F) continue;
          const int G = incs[j];
          for (int k = k_lo; k < k_hi; ++k) {
            send_per_peer[G].push_back(k * width + v);
          }
        }
      } else {
        // Recv from the owner only.
        for (int k = k_lo; k < k_hi; ++k) {
          recv_per_peer[owner].push_back(k * width + v);
        }
      }
    }
  }

  // Assemble peers in ascending rank order.
  std::set<int> peer_set;
  for (auto const& p : recv_per_peer) peer_set.insert(p.first);
  for (auto const& p : send_per_peer) peer_set.insert(p.first);
  out.peers.reserve(peer_set.size());
  for (int g : peer_set) {
    halo_plan::peer_entry pe;
    pe.peer_rank = g;
    auto itr = recv_per_peer.find(g);
    if (itr != recv_per_peer.end()) pe.recv_global_idx = std::move(itr->second);
    auto its = send_per_peer.find(g);
    if (its != send_per_peer.end()) pe.send_global_idx = std::move(its->second);
    out.peers.push_back(std::move(pe));
  }

  return out;
}

// =========================================================================
// In-process backend.
// =========================================================================
void in_process_halo_backend::register_rank(int rank, Scalar* buffer) {
  if (int(m_rank_buffers.size()) <= rank) {
    m_rank_buffers.resize(rank + 1, nullptr);
  }
  m_rank_buffers[rank] = buffer;
}

void in_process_halo_backend::exchange(int my_rank, Scalar* my_buffer,
                                        const halo_plan& plan) {
  (void)my_rank;  // not used: direct pull from peer.
  // All ranks share the global-index convention, so a peer's buffer
  // already has the correct value at the index we want to receive —
  // we just read it directly.  This collapses the send/recv pair into
  // a one-sided pull, which is exactly what we want for a test fixture.
  for (auto const& pe : plan.peers) {
    assert(pe.peer_rank >= 0 &&
           pe.peer_rank < int(m_rank_buffers.size()));
    Scalar* peer_buf = m_rank_buffers[pe.peer_rank];
    assert(peer_buf != nullptr);
    for (int i = 0; i < int(pe.recv_global_idx.size()); ++i) {
      int idx = pe.recv_global_idx[i];
      my_buffer[idx] = peer_buf[idx];
    }
  }
}

void in_process_halo_backend::exchange_all(const std::vector<halo_plan>& plans) {
  assert(int(plans.size()) == size());
  for (int r = 0; r < size(); ++r) {
    if (m_rank_buffers[r] != nullptr) {
      exchange(r, m_rank_buffers[r], plans[r]);
    }
  }
}

}  // namespace Aperture
