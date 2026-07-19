#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_partition.h"
#include <memory>
#include <queue>
#include <set>
#include <vector>

using namespace Aperture;

TEST_CASE("icosahedron edge-adjacency: each face has exactly 3 edge neighbors",
          "[prismatic][partition]") {
  const auto& edges = prismatic_partition::edge_neighbors();
  for (int f = 0; f < 20; ++f) {
    std::set<int> s;
    for (int g : edges[f]) {
      REQUIRE(g >= 0);
      REQUIRE(g < 20);
      REQUIRE(g != f);
      s.insert(g);
    }
    REQUIRE(s.size() == 3);
  }
}

TEST_CASE("icosahedron diagonal-adjacency: each face has exactly 6 vertex-only "
          "neighbors, and they are disjoint from the edge neighbors",
          "[prismatic][partition]") {
  const auto& edges = prismatic_partition::edge_neighbors();
  const auto& diags = prismatic_partition::diagonal_neighbors();
  for (int f = 0; f < 20; ++f) {
    std::set<int> e_set(edges[f].begin(), edges[f].end());
    std::set<int> d_set;
    for (int g : diags[f]) {
      REQUIRE(g >= 0);
      REQUIRE(g < 20);
      REQUIRE(g != f);
      REQUIRE(e_set.find(g) == e_set.end());  // no overlap with edge neighbors
      d_set.insert(g);
    }
    REQUIRE(d_set.size() == 6);
  }
}

TEST_CASE("icosahedron adjacency is symmetric: if g ∈ N(f), then f ∈ N(g)",
          "[prismatic][partition]") {
  const auto& edges = prismatic_partition::edge_neighbors();
  const auto& diags = prismatic_partition::diagonal_neighbors();

  auto contains = [](auto const& arr, int v) {
    for (int x : arr)
      if (x == v) return true;
    return false;
  };

  for (int f = 0; f < 20; ++f) {
    for (int g : edges[f])
      REQUIRE(contains(edges[g], f));
    for (int g : diags[f])
      REQUIRE(contains(diags[g], f));
  }
}

TEST_CASE("single-rank partition owns every global element",
          "[prismatic][partition]") {
  const int L = 3;
  const int N_r = 16;
  auto p = prismatic_partition::single_rank(L, N_r);

  REQUIRE(p.is_single_rank());
  REQUIRE(p.N_tri_global == 20 * 64);   // 20 · 4^3
  REQUIRE(p.N_vert_s_global == 10 * 64 + 2);
  REQUIRE(p.N_edge_s_global == 30 * 64);
  REQUIRE(p.ico_face_lo == 0);
  REQUIRE(p.ico_face_hi == 20);
  REQUIRE(p.shell_k_lo == 0);
  // Shells live at k ∈ [0, N_r] — the single-rank partition owns all
  // N_r + 1 of them, so shell_k_hi = N_r + 1.
  REQUIRE(p.shell_k_hi == N_r + 1);

  // Ownership covers the full global index range, no gaps or overlaps.
  for (int shell_k = 0; shell_k <= N_r; ++shell_k) {
    REQUIRE(p.owns_shell(shell_k));
  }
  for (int slab_k = 0; slab_k < N_r; ++slab_k) {
    REQUIRE(p.owns_slab(slab_k));
  }
  REQUIRE_FALSE(p.owns_slab(N_r));   // slab N_r doesn't exist
  REQUIRE_FALSE(p.owns_slab(-1));
  for (int fi = 0; fi < p.N_tri_global; ++fi) {
    REQUIRE(p.owns_sub_tri(fi));
  }
  for (int f = 0; f < 20; ++f) REQUIRE(p.owns_ico_face(f));

  // Angular neighbor list empty in single-rank mode.
  REQUIRE(p.angular_neighbors.empty());
}

TEST_CASE("ico-face sub-triangle ranges are contiguous and non-overlapping",
          "[prismatic][partition]") {
  const int L = 4;
  auto p = prismatic_partition::single_rank(L, 8);
  const int sz = p.N_tri_global / 20;
  REQUIRE(sz == 256);  // 4^4
  for (int f = 0; f < 20; ++f) {
    for (int j = 0; j < sz; ++j) {
      int tri_idx = f * sz + j;
      REQUIRE(tri_idx / sz == f);  // ico_face_of(tri_idx) == f
    }
  }
}

// =========================================================================
// Phase 1.3 — partition tiling invariants.
//
// For every multi-rank decomposition we construct, check that:
//   (a) every global cochain index is owned by EXACTLY ONE rank, and
//   (b) the union of owned sets covers the full global index range.
//
// This must hold for every cochain type.  For cochains whose angular
// ownership requires subdivision topology (rect_face, h_edge, v_edge,
// vertex), only the radial-only decomposition case is checked here —
// the angular decomposition cases are deferred to Phase 3 when the
// topology tables land.
// =========================================================================

namespace {

// Count how many partitions in `parts` own the given global index for
// the given cochain type.  Used to verify exactly-one-owner tiling.
template <typename PartVec, typename OwnFn>
void check_tiling(const PartVec& parts, int global_size, OwnFn own) {
  std::vector<int> owner_count(global_size, 0);
  for (auto const& p : parts) {
    for (int g = 0; g < global_size; ++g) {
      if (own(p, g)) ++owner_count[g];
    }
  }
  for (int g = 0; g < global_size; ++g) {
    REQUIRE(owner_count[g] == 1);
  }
}

}  // namespace

TEST_CASE("single-rank covers every cochain type without overlap",
          "[prismatic][partition][tiling]") {
  const int L = 2;
  const int N_r = 4;
  std::vector<prismatic_partition> parts{
      prismatic_partition::single_rank(L, N_r)};

  check_tiling(parts, parts[0].N_tri_faces_global,
               [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
  check_tiling(parts, parts[0].N_rect_faces_global,
               [](auto const& p, int g) { return p.owns_rect_face_cochain(g); });
  check_tiling(parts, parts[0].N_h_edges_global,
               [](auto const& p, int g) { return p.owns_h_edge_cochain(g); });
  check_tiling(parts, parts[0].N_v_edges_global,
               [](auto const& p, int g) { return p.owns_v_edge_cochain(g); });
  check_tiling(parts, parts[0].N_verts_global,
               [](auto const& p, int g) { return p.owns_vertex_cochain(g); });
}

TEST_CASE("20-way angular decomposition: tri_face cochain tiles the mesh",
          "[prismatic][partition][tiling]") {
  const int L = 2;
  const int N_r = 4;
  std::vector<prismatic_partition> parts;
  for (int f = 0; f < 20; ++f) {
    parts.push_back(prismatic_partition::ico_face_angular(L, N_r, f));
  }

  check_tiling(parts, parts[0].N_tri_faces_global,
               [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
}

TEST_CASE("radial-only decomposition tiles every cochain type",
          "[prismatic][partition][tiling]") {
  const int L = 2;
  const int N_r = 12;
  const int K = 4;
  std::vector<prismatic_partition> parts;
  for (int r = 0; r < K; ++r) {
    parts.push_back(prismatic_partition::radial_slab(L, N_r, K, r));
  }

  // Each partition covers the full angular span (ico_face_lo=0, hi=20),
  // so the topology-dependent cochain ownership reduces to radial shells.
  for (auto const& p : parts) REQUIRE(p.owns_all_angular());

  check_tiling(parts, parts[0].N_tri_faces_global,
               [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
  check_tiling(parts, parts[0].N_rect_faces_global,
               [](auto const& p, int g) { return p.owns_rect_face_cochain(g); });
  check_tiling(parts, parts[0].N_h_edges_global,
               [](auto const& p, int g) { return p.owns_h_edge_cochain(g); });
  check_tiling(parts, parts[0].N_v_edges_global,
               [](auto const& p, int g) { return p.owns_v_edge_cochain(g); });
  check_tiling(parts, parts[0].N_verts_global,
               [](auto const& p, int g) { return p.owns_vertex_cochain(g); });
}

TEST_CASE("combined angular × radial decomposition: tri_face cochain tiles",
          "[prismatic][partition][tiling]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  std::vector<prismatic_partition> parts;
  parts.reserve(K * 20);
  for (int r = 0; r < K; ++r) {
    for (int f = 0; f < 20; ++f) {
      parts.push_back(prismatic_partition::combined_ico_face(L, N_r, K, r, f));
    }
  }

  REQUIRE(parts.size() == size_t(K * 20));
  check_tiling(parts, parts[0].N_tri_faces_global,
               [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
}

TEST_CASE("partition global counts are internally consistent",
          "[prismatic][partition]") {
  const int L = 3;
  const int N_r = 17;
  auto p = prismatic_partition::single_rank(L, N_r);
  REQUIRE(p.N_tri_faces_global  == (N_r + 1) * p.N_tri_global);
  REQUIRE(p.N_rect_faces_global == N_r * p.N_edge_s_global);
  REQUIRE(p.N_h_edges_global    == (N_r + 1) * p.N_edge_s_global);
  REQUIRE(p.N_v_edges_global    == N_r * p.N_vert_s_global);
  REQUIRE(p.N_verts_global      == (N_r + 1) * p.N_vert_s_global);
}

// =========================================================================
// Phase 7A.1 — generalized angular units (PHASE_7_SCALABLE_PIC_PLAN.md F3):
// Hamiltonian face path, unit descriptor arithmetic, min-incident-unit
// ownership, and the A=20/m=0 owned-set equivalence anchor.
// =========================================================================

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, 2, 1.0, 2.0);
  return mesh;
}

// Owned-set signature of a partition: ownership bit for every element of
// every cochain type, in a fixed order.  Two partitions with equal
// signatures own exactly the same global elements.
std::vector<bool> owned_signature(const prismatic_partition& p) {
  std::vector<bool> out;
  out.reserve(p.N_tri_faces_global + p.N_rect_faces_global +
              p.N_h_edges_global + p.N_v_edges_global + p.N_verts_global);
  for (int g = 0; g < p.N_tri_faces_global; ++g)
    out.push_back(p.owns_tri_face_cochain(g));
  for (int g = 0; g < p.N_rect_faces_global; ++g)
    out.push_back(p.owns_rect_face_cochain(g));
  for (int g = 0; g < p.N_h_edges_global; ++g)
    out.push_back(p.owns_h_edge_cochain(g));
  for (int g = 0; g < p.N_v_edges_global; ++g)
    out.push_back(p.owns_v_edge_cochain(g));
  for (int g = 0; g < p.N_verts_global; ++g)
    out.push_back(p.owns_vertex_cochain(g));
  return out;
}

// BFS connected-component check of this rank's owned sub-triangle set,
// with adjacency from the topology's sphere-edge → 2-triangle table.
bool owned_tris_connected(const prismatic_partition& p,
                          const icosphere_topology& topo) {
  std::vector<std::vector<int>> adj(topo.N_tri());
  for (int e = 0; e < topo.N_edge_s(); ++e) {
    int a = topo.edge_tri_a(e), b = topo.edge_tri_b(e);
    adj[a].push_back(b);
    adj[b].push_back(a);
  }
  std::vector<char> owned(topo.N_tri(), 0), seen(topo.N_tri(), 0);
  int n_owned = 0, start = -1;
  for (int t = 0; t < topo.N_tri(); ++t) {
    if (p.owns_sub_tri(t)) {
      owned[t] = 1;
      ++n_owned;
      if (start < 0) start = t;
    }
  }
  if (n_owned == 0) return false;
  std::queue<int> q;
  q.push(start);
  seen[start] = 1;
  int n_seen = 1;
  while (!q.empty()) {
    int t = q.front();
    q.pop();
    for (int u : adj[t]) {
      if (owned[u] && !seen[u]) {
        seen[u] = 1;
        ++n_seen;
        q.push(u);
      }
    }
  }
  return n_seen == n_owned;
}

}  // namespace

TEST_CASE("face path is a Hamiltonian cycle on the face-adjacency graph, "
          "and face_path_pos is its inverse",
          "[prismatic][partition][units]") {
  const auto& path = prismatic_partition::face_path();
  const auto& pos = prismatic_partition::face_path_pos();
  const auto& edges = prismatic_partition::edge_neighbors();

  // Permutation of [0, 20).
  std::set<int> s(path.begin(), path.end());
  REQUIRE(s.size() == 20);
  REQUIRE(*s.begin() == 0);
  REQUIRE(*s.rbegin() == 19);

  // Inverse table.
  for (int p = 0; p < 20; ++p) REQUIRE(pos[path[p]] == p);

  // Consecutive faces along the path share an ico-edge — cyclically, so
  // entry 19 must also be adjacent to entry 0.
  auto adjacent = [&](int f, int g) {
    for (int x : edges[f])
      if (x == g) return true;
    return false;
  };
  for (int p = 0; p < 20; ++p) {
    REQUIRE(adjacent(path[p], path[(p + 1) % 20]));
  }
}

TEST_CASE("min_patch_level_for: valid A values get the smallest level, "
          "invalid A values get -1",
          "[prismatic][partition][units]") {
  // A | 20·4^m with 20·4^m = 2^(2m+2)·5.
  REQUIRE(prismatic_partition::min_patch_level_for(1) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(2) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(4) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(5) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(10) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(20) == 0);
  REQUIRE(prismatic_partition::min_patch_level_for(8) == 1);
  REQUIRE(prismatic_partition::min_patch_level_for(16) == 1);
  REQUIRE(prismatic_partition::min_patch_level_for(40) == 1);
  REQUIRE(prismatic_partition::min_patch_level_for(80) == 1);
  REQUIRE(prismatic_partition::min_patch_level_for(64) == 2);
  REQUIRE(prismatic_partition::min_patch_level_for(160) == 2);
  REQUIRE(prismatic_partition::min_patch_level_for(320) == 2);
  // Only one factor of 5 is available in 20·4^m = 2^(2m+2)·5, so 5²
  // can never divide: the 8000-GCD shapes put the second 5 on the
  // RADIAL axis (A=320 × K=25, A=80 × K=100).
  REQUIRE(prismatic_partition::min_patch_level_for(100) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(25) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(3) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(6) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(12) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(15) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(0) == -1);
  REQUIRE(prismatic_partition::min_patch_level_for(-4) == -1);
}

TEST_CASE("angular_units factory validates A, m, and rank",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  // Invalid A.
  REQUIRE_THROWS_AS(prismatic_partition::angular_units(L, N_r, 3, 0),
                    std::invalid_argument);
  // Explicit m too small for A.
  REQUIRE_THROWS_AS(prismatic_partition::angular_units(L, N_r, 80, 0, 0),
                    std::invalid_argument);
  // Required m exceeds L: A=320 needs m=2 > L=1.
  REQUIRE_THROWS_AS(prismatic_partition::angular_units(1, N_r, 320, 0),
                    std::invalid_argument);
  // Rank out of range.
  REQUIRE_THROWS_AS(prismatic_partition::angular_units(L, N_r, 4, 4),
                    std::invalid_argument);
  // Valid: explicit m larger than the minimum is allowed.
  REQUIRE_NOTHROW(prismatic_partition::angular_units(L, N_r, 4, 0, 2));
}

TEST_CASE("unit descriptor arithmetic: contiguous tri blocks and path "
          "ordering structure",
          "[prismatic][partition][units]") {
  const int L = 3, N_r = 4;
  // A=80 at L=3 → m=1: units are quarter-faces of 4^(L-1)=16 tris.
  auto p = prismatic_partition::angular_units(L, N_r, 80, 17);
  REQUIRE(p.patch_level == 1);
  REQUIRE(p.n_units() == 80);
  REQUIRE(p.units_per_face() == 4);
  REQUIRE(p.unit_hi - p.unit_lo == 1);

  const int tris_per_unit = p.N_tri_global / p.n_units();
  REQUIRE(tris_per_unit == 16);
  // Each global unit u covers exactly tris [u·16, (u+1)·16).
  for (int t = 0; t < p.N_tri_global; ++t) {
    REQUIRE(p.unit_of_tri(t) == t / tris_per_unit);
  }
  // Path ordering: face f's units occupy path positions
  // [face_path_pos[f]·4^m, +4^m), preserving child order within f.
  const auto& pos = prismatic_partition::face_path_pos();
  for (int u = 0; u < p.n_units(); ++u) {
    int f = u / p.units_per_face();
    int child = u % p.units_per_face();
    REQUIRE(p.path_of_unit(u) == pos[f] * p.units_per_face() + child);
  }
  // The owned tri set is exactly one contiguous 16-tri block.
  int n_owned = 0;
  for (int t = 0; t < p.N_tri_global; ++t)
    if (p.owns_sub_tri(t)) ++n_owned;
  REQUIRE(n_owned == tris_per_unit);
}

TEST_CASE("A=20/m=0 equivalence anchor: generalized units reproduce the "
          "historical per-ico-face owned sets exactly, all cochain types",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  const auto& pos = prismatic_partition::face_path_pos();
  for (int f = 0; f < 20; ++f) {
    auto old_p = prismatic_partition::ico_face_angular(L, N_r, f);
    old_p.set_topology(&topo);
    // The generalized rank owning exactly face f sits at path position
    // pos[f] under the A=20 split.
    auto new_p = prismatic_partition::angular_units(L, N_r, 20, pos[f]);
    new_p.set_topology(&topo);
    REQUIRE(new_p.patch_level == 0);
    REQUIRE(new_p.ico_face_lo == f);  // legacy view synced
    REQUIRE(new_p.ico_face_hi == f + 1);
    REQUIRE(owned_signature(new_p) == owned_signature(old_p));
  }
}

TEST_CASE("min-incident-unit ownership at m=0 matches the topology's "
          "lowest-incident-ico-face owner on every sphere element",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto p = prismatic_partition::angular_units(L, N_r, 4, 1);
  p.set_topology(&topo);
  REQUIRE(p.patch_level == 0);  // A=4 divides 20 at m=0

  for (int e = 0; e < topo.N_edge_s(); ++e) {
    REQUIRE(p.owner_unit_of_sphere_edge(e) == topo.edge_owner_ico_face(e));
  }
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    REQUIRE(p.owner_unit_of_sphere_vertex(v) ==
            topo.vertex_owner_ico_face(v));
  }
}

TEST_CASE("owned sets are invariant under the patch level used at fixed A",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int A : {2, 4, 5, 10, 20}) {
    for (int a = 0; a < A; ++a) {
      auto p0 = prismatic_partition::angular_units(L, N_r, A, a, 0);
      auto p1 = prismatic_partition::angular_units(L, N_r, A, a, 1);
      auto p2 = prismatic_partition::angular_units(L, N_r, A, a, 2);
      p0.set_topology(&topo);
      p1.set_topology(&topo);
      p2.set_topology(&topo);
      auto sig = owned_signature(p0);
      REQUIRE(owned_signature(p1) == sig);
      REQUIRE(owned_signature(p2) == sig);
    }
  }
}

TEST_CASE("generalized angular decomposition tiles every cochain type "
          "for all valid A at L=2",
          "[prismatic][partition][units][tiling]") {
  const int L = 2, N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int A : {2, 4, 5, 8, 10, 16, 20, 40, 80, 160, 320}) {
    std::vector<prismatic_partition> parts;
    for (int a = 0; a < A; ++a) {
      auto p = prismatic_partition::angular_units(L, N_r, A, a);
      p.set_topology(&topo);
      parts.push_back(p);
    }
    check_tiling(parts, parts[0].N_tri_faces_global,
                 [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
    check_tiling(parts, parts[0].N_rect_faces_global,
                 [](auto const& p, int g) { return p.owns_rect_face_cochain(g); });
    check_tiling(parts, parts[0].N_h_edges_global,
                 [](auto const& p, int g) { return p.owns_h_edge_cochain(g); });
    check_tiling(parts, parts[0].N_v_edges_global,
                 [](auto const& p, int g) { return p.owns_v_edge_cochain(g); });
    check_tiling(parts, parts[0].N_verts_global,
                 [](auto const& p, int g) { return p.owns_vertex_cochain(g); });
  }
}

TEST_CASE("combined A×K decomposition tiles all cochains at the 8-rank "
          "single-node Frontier shape (A=4, K=2)",
          "[prismatic][partition][units][tiling]") {
  const int L = 2, N_r = 8;
  const int A = 4, K = 2;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  std::vector<prismatic_partition> parts;
  for (int w = 0; w < A * K; ++w) {
    auto p = prismatic_partition::combined(L, N_r, A, K, w);
    p.set_topology(&topo);
    REQUIRE(p.radial_rank == w / A);
    REQUIRE(p.angular_rank == w % A);
    parts.push_back(p);
  }
  check_tiling(parts, parts[0].N_tri_faces_global,
               [](auto const& p, int g) { return p.owns_tri_face_cochain(g); });
  check_tiling(parts, parts[0].N_rect_faces_global,
               [](auto const& p, int g) { return p.owns_rect_face_cochain(g); });
  check_tiling(parts, parts[0].N_h_edges_global,
               [](auto const& p, int g) { return p.owns_h_edge_cochain(g); });
  check_tiling(parts, parts[0].N_v_edges_global,
               [](auto const& p, int g) { return p.owns_v_edge_cochain(g); });
  check_tiling(parts, parts[0].N_verts_global,
               [](auto const& p, int g) { return p.owns_vertex_cochain(g); });
}

TEST_CASE("path ordering guarantees connected owned tri sets for "
          "whole-face groups and single-patch configurations",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  // Whole-face groups (20/A faces per rank, consecutive along the
  // Hamiltonian path) and single-patch configurations (one aligned
  // power-of-4 unit per rank: A = 20·4^m).
  for (int A : {1, 2, 4, 5, 10, 20, 80, 320}) {
    for (int a = 0; a < A; ++a) {
      auto p = prismatic_partition::angular_units(L, N_r, A, a);
      REQUIRE(owned_tris_connected(p, topo));
    }
  }
}

TEST_CASE("angular_rank_of_path_unit maps owned units back to their rank",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  for (int A : {4, 8, 20, 80}) {
    for (int a = 0; a < A; ++a) {
      auto p = prismatic_partition::angular_units(L, N_r, A, a);
      for (int u = p.unit_lo; u < p.unit_hi; ++u) {
        REQUIRE(p.angular_rank_of_path_unit(u) == a);
      }
    }
  }
}

TEST_CASE("legacy whole-face view sync: full sphere, single face, and "
          "unrepresentable unit partitions",
          "[prismatic][partition][units]") {
  const int L = 2, N_r = 4;
  // Full sphere.
  auto p1 = prismatic_partition::angular_units(L, N_r, 1, 0);
  REQUIRE(p1.ico_face_lo == 0);
  REQUIRE(p1.ico_face_hi == 20);
  REQUIRE(p1.owns_all_angular());
  // Multi-face group: not representable as a face range.
  auto p4 = prismatic_partition::angular_units(L, N_r, 4, 2);
  REQUIRE(p4.ico_face_lo == -1);
  REQUIRE(p4.ico_face_hi == -1);
  // Sub-face units: not representable.
  auto p80 = prismatic_partition::angular_units(L, N_r, 80, 3);
  REQUIRE(p80.ico_face_lo == -1);
  REQUIRE(p80.ico_face_hi == -1);
}
