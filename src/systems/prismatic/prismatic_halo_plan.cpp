#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/icosphere_topology.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <set>
#include <stdexcept>
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
// Returns true if the given global cochain index is in this rank's
// angular ownership — independent of radial ownership.  Equivalent to
// "would self.owns_*_cochain be true if the shell/slab were owned".
static bool angular_owns(cochain_type t, const prismatic_partition& self,
                          int global_idx) {
  int width = 0;
  switch (t) {
    case cochain_type::tri_face:  width = self.N_tri_global;    break;
    case cochain_type::rect_face: width = self.N_edge_s_global; break;
    case cochain_type::h_edge:    width = self.N_edge_s_global; break;
    case cochain_type::v_edge:    width = self.N_vert_s_global; break;
    case cochain_type::vertex:    width = self.N_vert_s_global; break;
  }
  int sub = global_idx % width;
  switch (t) {
    case cochain_type::tri_face:  return self.owns_sub_tri(sub);
    case cochain_type::h_edge:
    case cochain_type::rect_face: return self.owns_sphere_edge(sub);
    case cochain_type::v_edge:
    case cochain_type::vertex:    return self.owns_sphere_vertex(sub);
  }
  return false;
}

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

  // Fill `out` with [k*width + i for i in 0..width) filtered to the
  // indices whose angular side is owned by `self`.  Under the radial
  // peer's convention the peer shares the same angular ownership, so
  // this filter is symmetric between sender and receiver.
  auto angular_filtered = [&](int k, std::vector<int>& out) {
    for (int i = 0; i < width; ++i) {
      int g = k * width + i;
      if (angular_owns(t, self, g)) out.push_back(g);
    }
  };

  if (has_lower) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r - 1;
    if (shell_cochain) {
      angular_filtered(kl,     pe.send_global_idx);
      angular_filtered(kl - 1, pe.recv_global_idx);
    } else {
      angular_filtered(kl - 1, pe.recv_global_idx);
    }
    out.peers.push_back(std::move(pe));
  }

  if (has_upper) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r + 1;
    if (shell_cochain) {
      angular_filtered(kh - 1, pe.send_global_idx);
      angular_filtered(kh,     pe.recv_global_idx);
    } else {
      angular_filtered(kh - 1, pe.send_global_idx);
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

    // --- Multi-incidence vertex fan halos (rect_face only) ------------
    //
    // compute_dD_dt on a v_edge at sphere vertex v uses H_aux at the
    // fan of rect faces radiating from v (one per sphere-edge incident
    // to v).  When v sits on an ico-edge boundary (vertex_valence ≥ 2)
    // some of those fan rect faces are at sphere-edges that F is NOT
    // topologically incident to; the normal rect_face halo (above)
    // only covers F-incident sphere-edges, leaving a gap.
    //
    //   valence 1 (interior to F):  no gap — F is incident to all
    //                               sphere-edges in the fan.
    //   valence 2 (on an ico-edge): F-incident to ~4 of 6 fan edges;
    //                               2 are gaps.
    //   valence 5 (icosahedron corner): F-incident to 2 of 5 fan edges;
    //                                   3 are gaps.
    //
    // Fix: for every multi-incidence vertex where F owns the v_edge,
    // recv rect_face values at every fan sphere-edge F is NOT incident
    // to from that edge's owner.  Symmetric send entries on those
    // owners keep the exchange consistent.
    //
    // Applies only to rect_face (not h_edge) because the stencil gap
    // is specific to the compute_dD_dt fan on v_edges.
    if (t == cochain_type::rect_face) {
      for (int v = 0; v < topo.N_vert_s(); ++v) {
        const int v_val = topo.vertex_valence(v);
        if (v_val < 2) continue;  // interior to one ico-face — no gap
        const int* incs = topo.vertex_ico_faces(v);
        const int v_owner = incs[0];

        // Check if F is incident to v.
        bool F_incident = false;
        for (int j = 0; j < v_val; ++j)
          if (incs[j] == F) { F_incident = true; break; }
        if (!F_incident) continue;

        // Enumerate the 5 sphere-edges at v.
        const int ve_count = topo.vertex_edge_count(v);
        const int* ve = topo.vertex_edges(v);

        if (F == v_owner) {
          // F owns the v_edge at v.  For each fan sphere-edge not
          // incident to F, recv rect_face values from the owner.
          for (int i = 0; i < ve_count; ++i) {
            int e = ve[i];
            const int* einc = topo.edge_ico_faces(e);
            if (einc[0] == F || einc[1] == F) continue;  // F-incident: already handled
            const int e_owner = einc[0];
            for (int k = k_lo; k < k_hi; ++k) {
              recv_per_peer[e_owner].push_back(k * width + e);
            }
          }
        } else if (F == incs[0]) {
          // Unreachable — incs[0] is v_owner by our convention.
        } else {
          // F is incident to v but doesn't own v_edge.  Symmetric send:
          // if F owns a fan sphere-edge NOT incident to v_owner, F
          // sends its rect_face values to v_owner.
          for (int i = 0; i < ve_count; ++i) {
            int e = ve[i];
            const int* einc = topo.edge_ico_faces(e);
            // Only F-owned edges produce sends from F.
            if (einc[0] != F) continue;  // F is not the edge's owner
            // Only edges not incident to v_owner produce extra sends.
            if (einc[0] == v_owner || einc[1] == v_owner) continue;
            for (int k = k_lo; k < k_hi; ++k) {
              send_per_peer[v_owner].push_back(k * width + e);
            }
          }
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
// pic depth-class machinery (Phase 7B).
//
// TP(t) = "halo consumer" angular ranks of prism t: the ranks owning any
// tri that shares ≥ 1 sphere-vertex with t (owner(t) included, since t
// shares its own vertices).  Element x is in rank R's pic halo iff
// R ∈ TP(t) for some prism t incident to x; the owner of x sends to
// exactly ⋃ TP(tris(x)) \ {owner}.  Built once per builder call —
// O(N_tri · 18) rank lookups, negligible at build time.
// =========================================================================
namespace {

std::vector<std::vector<int>> build_tri_halo_consumers(
    const prismatic_partition& self, const icosphere_topology& topo) {
  const int n_tri = topo.N_tri();
  auto rank_of_tri = [&](int tri) {
    return self.angular_rank_of_path_unit(
        self.path_of_unit(self.unit_of_tri(tri)));
  };

  // Invert the vertex→tri fans into tri→verts (the topology does not
  // store tri_verts; every tri appears in exactly 3 fans).
  std::vector<std::array<int, 3>> tri_verts(n_tri, {-1, -1, -1});
  std::vector<int> nfill(n_tri, 0);
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    const int* tris = topo.vertex_tris(v);
    const int n = topo.vertex_tri_count(v);
    for (int j = 0; j < n; ++j) {
      const int t = tris[j];
      assert(nfill[t] < 3 && "tri appears in more than 3 vertex fans");
      tri_verts[t][nfill[t]++] = v;
    }
  }

  std::vector<std::vector<int>> out(n_tri);
  for (int t = 0; t < n_tri; ++t) {
    auto& s = out[t];
    for (int v : tri_verts[t]) {
      const int* fan = topo.vertex_tris(v);
      const int n = topo.vertex_tri_count(v);
      for (int j = 0; j < n; ++j) s.push_back(rank_of_tri(fan[j]));
    }
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
  }
  return out;
}

// Union of TP over the prisms incident to sphere element x, per cochain
// kind, into `out` (cleared; sorted ascending, unique).
void pic_consumer_ranks(cochain_type t, int sub,
                        const icosphere_topology& topo,
                        const std::vector<std::vector<int>>& TP,
                        std::vector<int>& out) {
  out.clear();
  int tris[6];
  int n_tris = 0;
  switch (t) {
    case cochain_type::tri_face:
      tris[n_tris++] = sub;
      break;
    case cochain_type::h_edge:
    case cochain_type::rect_face:
      tris[n_tris++] = topo.edge_tri_a(sub);
      tris[n_tris++] = topo.edge_tri_b(sub);
      break;
    case cochain_type::v_edge:
    case cochain_type::vertex: {
      const int* fan = topo.vertex_tris(sub);
      n_tris = topo.vertex_tri_count(sub);
      for (int j = 0; j < n_tris; ++j) tris[j] = fan[j];
      break;
    }
  }
  for (int j = 0; j < n_tris; ++j) {
    const auto& tp = TP[tris[j]];
    out.insert(out.end(), tp.begin(), tp.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

// Owner angular rank of sphere element `sub` of the given cochain kind.
int pic_owner_rank(cochain_type t, int sub, const prismatic_partition& self,
                   const icosphere_topology& topo) {
  auto rank_of_unit = [&](int unit) {
    return self.angular_rank_of_path_unit(self.path_of_unit(unit));
  };
  switch (t) {
    case cochain_type::tri_face:
      return rank_of_unit(self.unit_of_tri(sub));
    case cochain_type::h_edge:
    case cochain_type::rect_face:
      return rank_of_unit(std::min(self.unit_of_tri(topo.edge_tri_a(sub)),
                                   self.unit_of_tri(topo.edge_tri_b(sub))));
    case cochain_type::v_edge:
    case cochain_type::vertex: {
      const int* tris = topo.vertex_tris(sub);
      const int n = topo.vertex_tri_count(sub);
      int u = self.unit_of_tri(tris[0]);
      for (int j = 1; j < n; ++j) u = std::min(u, self.unit_of_tri(tris[j]));
      return rank_of_unit(u);
    }
  }
  return -1;  // unreachable
}

int pic_n_elems(cochain_type t, const icosphere_topology& topo) {
  switch (t) {
    case cochain_type::tri_face:  return topo.N_tri();
    case cochain_type::h_edge:
    case cochain_type::rect_face: return topo.N_edge_s();
    case cochain_type::v_edge:
    case cochain_type::vertex:    return topo.N_vert_s();
  }
  return 0;
}

}  // namespace

// =========================================================================
// Generic angular halo plan (Phase 7A.3 solver class, 7B pic class) —
// see header for the per-element ghost rules.  Loops are element-major
// over the sphere tables with the radial k range inner; per-peer lists
// are sorted by global cochain index and deduplicated at the end, which
// is the canonical wire order both sides derive independently.
// =========================================================================
halo_plan build_angular_halo_plan_units(cochain_type t,
                                        const prismatic_partition& self,
                                        const icosphere_topology& topo,
                                        halo_depth depth) {
  halo_plan out;
  if (self.owns_all_angular()) return out;

  // Angular rank of the owner of each sphere-element kind, from the
  // partition's unit arithmetic (min incident unit owns shared
  // elements).  All O(1) per query.
  auto rank_of_unit = [&](int unit) {
    return self.angular_rank_of_path_unit(self.path_of_unit(unit));
  };
  auto rank_of_tri = [&](int tri) {
    return rank_of_unit(self.unit_of_tri(tri));
  };
  auto rank_of_edge = [&](int e) {
    return rank_of_unit(std::min(self.unit_of_tri(topo.edge_tri_a(e)),
                                 self.unit_of_tri(topo.edge_tri_b(e))));
  };
  auto rank_of_vertex = [&](int v) {
    const int* tris = topo.vertex_tris(v);
    const int n = topo.vertex_tri_count(v);
    int u = self.unit_of_tri(tris[0]);
    for (int j = 1; j < n; ++j) u = std::min(u, self.unit_of_tri(tris[j]));
    return rank_of_unit(u);
  };

  const int R = self.angular_rank;
  const bool shell_cochain = lives_on_shell(t);
  const int width = per_level_width(t, self);
  const int k_lo = self.shell_k_lo;
  const int k_hi = shell_cochain
                       ? self.shell_k_hi
                       : std::min(self.shell_k_hi, self.N_r_global);

  std::map<int, std::vector<int>> recv_per_peer;
  std::map<int, std::vector<int>> send_per_peer;
  auto push_levels = [&](std::map<int, std::vector<int>>& dst, int peer,
                         int sub) {
    auto& v = dst[peer];
    for (int k = k_lo; k < k_hi; ++k) v.push_back(k * width + sub);
  };

  if (depth == halo_depth::pic) {
    // One generic rule for every cochain kind (see header): the owner of
    // x sends to every rank whose T_halo contains a prism incident to x.
    const auto TP = build_tri_halo_consumers(self, topo);
    const int n_elems = pic_n_elems(t, topo);
    std::vector<int> peers;
    for (int x = 0; x < n_elems; ++x) {
      const int S = pic_owner_rank(t, x, self, topo);
      pic_consumer_ranks(t, x, topo, TP, peers);
      if (S == R) {
        for (int p : peers) {
          if (p != R) push_levels(send_per_peer, p, x);
        }
      } else {
        for (int p : peers) {
          if (p == R) {
            push_levels(recv_per_peer, S, x);
            break;
          }
        }
      }
    }
  } else if (t == cochain_type::tri_face) {
    // Iterate sphere-edges: the edge owner needs the non-owned adjacent
    // tri (d1t/curl on the edge reads both adjacent tri faces).
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      const int tri_a = topo.edge_tri_a(e);
      const int tri_b = topo.edge_tri_b(e);
      const int r_e = rank_of_edge(e);
      const int r_a = rank_of_tri(tri_a);
      const int r_b = rank_of_tri(tri_b);
      if (r_e == R) {
        if (r_a != R) push_levels(recv_per_peer, r_a, tri_a);
        if (r_b != R) push_levels(recv_per_peer, r_b, tri_b);
      } else {
        if (r_a == R) push_levels(send_per_peer, r_e, tri_a);
        if (r_b == R) push_levels(send_per_peer, r_e, tri_b);
      }
    }
  } else if (t == cochain_type::h_edge || t == cochain_type::rect_face) {
    // Sphere-edge data.  Consumers: d1 rows of adjacent tri faces; for
    // rect_face additionally the v_edge update fan at each endpoint
    // vertex.  Peers of edge e = owner ranks of those consumer anchors.
    const bool fan = (t == cochain_type::rect_face);
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      const int r_e = rank_of_edge(e);
      int peers[4];
      int n_peers = 0;
      auto add_peer = [&](int r) {
        for (int j = 0; j < n_peers; ++j)
          if (peers[j] == r) return;
        peers[n_peers++] = r;
      };
      add_peer(rank_of_tri(topo.edge_tri_a(e)));
      add_peer(rank_of_tri(topo.edge_tri_b(e)));
      if (fan) {
        add_peer(rank_of_vertex(topo.edge_v0(e)));
        add_peer(rank_of_vertex(topo.edge_v1(e)));
      }
      if (r_e == R) {
        for (int j = 0; j < n_peers; ++j) {
          if (peers[j] != R) push_levels(send_per_peer, peers[j], e);
        }
      } else {
        for (int j = 0; j < n_peers; ++j) {
          if (peers[j] == R) {
            push_levels(recv_per_peer, r_e, e);
            break;
          }
        }
      }
    }
  } else {  // vertex or v_edge
    // Sphere-vertex data.  Consumers: d1 rows of incident edges /
    // rect-face updates — all anchored on the fan tris.
    for (int v = 0; v < topo.N_vert_s(); ++v) {
      const int r_v = rank_of_vertex(v);
      const int* tris = topo.vertex_tris(v);
      const int n = topo.vertex_tri_count(v);
      if (r_v == R) {
        int sent_to[6];
        int n_sent = 0;
        for (int j = 0; j < n; ++j) {
          const int r_t = rank_of_tri(tris[j]);
          if (r_t == R) continue;
          bool dup = false;
          for (int w = 0; w < n_sent; ++w)
            if (sent_to[w] == r_t) { dup = true; break; }
          if (dup) continue;
          sent_to[n_sent++] = r_t;
          push_levels(send_per_peer, r_t, v);
        }
      } else {
        for (int j = 0; j < n; ++j) {
          if (rank_of_tri(tris[j]) == R) {
            push_levels(recv_per_peer, r_v, v);
            break;
          }
        }
      }
    }
  }

  // Canonical wire order: ascending global cochain index, unique.
  auto canonicalize = [](std::map<int, std::vector<int>>& m) {
    for (auto& kv : m) {
      auto& v = kv.second;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    }
  };
  canonicalize(recv_per_peer);
  canonicalize(send_per_peer);

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
// Radial halo plan, depth-aware (Phase 7B) — see header for the layer
// pattern and the exchange/reduce order contract.
// =========================================================================
halo_plan build_radial_halo_plan_depth(cochain_type t,
                                       const prismatic_partition& self,
                                       const icosphere_topology& topo,
                                       halo_depth depth) {
  if (depth == halo_depth::solver) return build_radial_halo_plan(t, self);

  halo_plan out;
  if (self.n_radial_ranks <= 1) return out;

  // The depth-2 upper shell (k_hi+1) must be owned by the IMMEDIATE
  // upper peer; under the uniform slab split the minimum span is
  // N_r / K shells.
  if (self.N_r_global / self.n_radial_ranks < 2) {
    throw std::invalid_argument(
        "pic radial halo: every radial slab must own >= 2 shells "
        "(N_r / K < 2)");
  }

  const int width = per_level_width(t, self);

  // Column filter: owned OR angular pic-ghost columns.  Including the
  // ghost columns is what delivers corner ghosts by forwarding (radial
  // peers share the angular rank, hence the same column set).
  std::vector<char> in_halo(width, 1);
  if (!self.owns_all_angular()) {
    const auto TP = build_tri_halo_consumers(self, topo);
    const int R = self.angular_rank;
    std::vector<int> peers;
    for (int x = 0; x < width; ++x) {
      in_halo[x] = 0;
      pic_consumer_ranks(t, x, topo, TP, peers);
      for (int p : peers) {
        if (p == R) {
          in_halo[x] = 1;
          break;
        }
      }
    }
  }

  const int kl = self.shell_k_lo;
  const int kh = self.shell_k_hi;
  const int r = self.radial_rank;
  const bool has_lower = r > 0;
  const bool has_upper = r < self.n_radial_ranks - 1;
  const bool shell_cochain = lives_on_shell(t);

  auto filtered = [&](int k, std::vector<int>& dst) {
    for (int i = 0; i < width; ++i) {
      if (in_halo[i]) dst.push_back(k * width + i);
    }
  };

  if (has_lower) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r - 1;
    if (shell_cochain) {
      // Ghost prism kl-1 spans shells {kl-1, kl}; only kl-1 is new.
      // The lower peer's ghost prism kh(=kl) spans shells {kl, kl+1}.
      filtered(kl, pe.send_global_idx);
      filtered(kl + 1, pe.send_global_idx);
      filtered(kl - 1, pe.recv_global_idx);
    } else {
      filtered(kl, pe.send_global_idx);      // lower's upper ghost slab
      filtered(kl - 1, pe.recv_global_idx);  // our lower ghost slab
    }
    out.peers.push_back(std::move(pe));
  }

  if (has_upper) {
    halo_plan::peer_entry pe;
    pe.peer_rank = r + 1;
    if (shell_cochain) {
      filtered(kh - 1, pe.send_global_idx);  // upper's shell kl-1
      filtered(kh, pe.recv_global_idx);      // ghost prism kh: shells
      filtered(kh + 1, pe.recv_global_idx);  // {kh, kh+1}
    } else {
      filtered(kh - 1, pe.send_global_idx);  // upper's lower ghost slab
      filtered(kh, pe.recv_global_idx);      // our upper ghost slab
    }
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
  // Collective emulation of MPI_Isend/MPI_Irecv pairs:
  //   rank A recv from rank B at local index A.recv[i]
  //   rank B send to rank A at local index B.send[i]  (paired by i)
  // The copy is buf_a[A.recv[i]] = buf_b[B.send[i]].
  //
  // Under global indexing, A.recv[i] == B.send[i] so the copy is
  // tautological when viewed on a single buffer.  Under local indexing
  // the two sides differ and we must go through both.
  for (int a = 0; a < size(); ++a) {
    Scalar* buf_a = m_rank_buffers[a];
    if (buf_a == nullptr) continue;
    auto const& plan_a = plans[a];
    for (auto const& pe_a : plan_a.peers) {
      const int b = pe_a.peer_rank;
      Scalar* buf_b = m_rank_buffers[b];
      if (buf_b == nullptr) continue;

      // Find the peer entry on b's plan that targets a.
      const halo_plan::peer_entry* pe_b = nullptr;
      for (auto const& ppe : plans[b].peers) {
        if (ppe.peer_rank == a) { pe_b = &ppe; break; }
      }
      if (pe_b == nullptr) continue;

      // Recv at a is paired with send at b: same i-th position.
      assert(pe_a.recv_global_idx.size() == pe_b->send_global_idx.size());
      for (size_t i = 0; i < pe_a.recv_global_idx.size(); ++i) {
        const int a_idx = pe_a.recv_global_idx[i];
        const int b_idx = pe_b->send_global_idx[i];
        buf_a[a_idx] = buf_b[b_idx];
      }
    }
  }
}

void in_process_halo_backend::reduce_all(const std::vector<halo_plan>& plans) {
  assert(int(plans.size()) == size());
  // Exact reverse of exchange_all: rank a's ghost slots accumulate into
  // peer b's paired send slots, then a's ghosts are zeroed.  Within one
  // call no += target is ever a slot this call zeroes (an axis' send
  // slots are never that same axis' ghosts), so rank iteration order is
  // immaterial; ACROSS calls the caller must run the radial axis before
  // the angular one (pic corner forwarding — see header).
  for (int a = 0; a < size(); ++a) {
    Scalar* buf_a = m_rank_buffers[a];
    if (buf_a == nullptr) continue;
    auto const& plan_a = plans[a];
    for (auto const& pe_a : plan_a.peers) {
      const int b = pe_a.peer_rank;
      Scalar* buf_b = m_rank_buffers[b];
      if (buf_b == nullptr) continue;

      const halo_plan::peer_entry* pe_b = nullptr;
      for (auto const& ppe : plans[b].peers) {
        if (ppe.peer_rank == a) { pe_b = &ppe; break; }
      }
      if (pe_b == nullptr) continue;

      assert(pe_a.recv_global_idx.size() == pe_b->send_global_idx.size());
      for (size_t i = 0; i < pe_a.recv_global_idx.size(); ++i) {
        buf_b[pe_b->send_global_idx[i]] += buf_a[pe_a.recv_global_idx[i]];
        buf_a[pe_a.recv_global_idx[i]] = Scalar(0);
      }
    }
  }
}

}  // namespace Aperture
