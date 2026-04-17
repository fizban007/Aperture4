#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_partition.h"
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
      parts.push_back(prismatic_partition::combined(L, N_r, K, r, f));
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
