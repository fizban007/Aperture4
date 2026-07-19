#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_partition.h"
#include <memory>
#include <set>
#include <vector>

using namespace Aperture;

// Smallest possible mesh: L=?, N_r=2 (just so build() succeeds).  The
// radial structure is irrelevant to the topology — we only use the
// sphere-side tables.
static std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, 2, 1.0, 2.0);
  return mesh;
}

// =========================================================================
// Basic structural invariants of the topology at every subdivision level.
// =========================================================================
TEST_CASE("icosphere topology L=0: valences match base icosahedron",
          "[prismatic][topology]") {
  auto mesh = make_mesh(0);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  REQUIRE(topo.L() == 0);
  REQUIRE(topo.N_tri() == 20);
  REQUIRE(topo.N_vert_s() == 12);
  REQUIRE(topo.N_edge_s() == 30);

  // Every base-icosahedron vertex touches exactly 5 ico-faces.
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    REQUIRE(topo.vertex_valence(v) == 5);
  }
  // Every base-icosahedron edge is shared by exactly 2 ico-faces.
  for (int e = 0; e < topo.N_edge_s(); ++e) {
    REQUIRE(topo.edge_valence(e) == 2);
  }
}

TEST_CASE("icosphere topology: valences are only 1, 2, or 5 (vertices) "
          "and 1 or 2 (edges)",
          "[prismatic][topology]") {
  for (int L : {1, 2, 3}) {
    auto mesh = make_mesh(L);
    auto topo = icosphere_topology::build_from_mesh(*mesh);

    for (int v = 0; v < topo.N_vert_s(); ++v) {
      int val = topo.vertex_valence(v);
      REQUIRE((val == 1 || val == 2 || val == 5));
    }
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      int val = topo.edge_valence(e);
      REQUIRE((val == 1 || val == 2));
    }
  }
}

TEST_CASE("icosphere topology: exactly 12 vertices have valence 5 "
          "(the original icosahedron corners)",
          "[prismatic][topology]") {
  for (int L : {0, 1, 2, 3}) {
    auto mesh = make_mesh(L);
    auto topo = icosphere_topology::build_from_mesh(*mesh);
    int n_val5 = 0;
    for (int v = 0; v < topo.N_vert_s(); ++v) {
      if (topo.vertex_valence(v) == 5) ++n_val5;
    }
    REQUIRE(n_val5 == 12);
  }
}

TEST_CASE("icosphere topology: vertex counts at L=1 and L=2 match the "
          "expected interior / ico-edge / corner split",
          "[prismatic][topology]") {
  // At L=1: 12 corners (val 5) + 30 ico-edge midpoints (val 2) = 42.
  {
    auto mesh = make_mesh(1);
    auto topo = icosphere_topology::build_from_mesh(*mesh);
    REQUIRE(topo.N_vert_s() == 42);
    int n5 = 0, n2 = 0, n1 = 0;
    for (int v = 0; v < topo.N_vert_s(); ++v) {
      switch (topo.vertex_valence(v)) {
        case 5: ++n5; break;
        case 2: ++n2; break;
        case 1: ++n1; break;
      }
    }
    REQUIRE(n5 == 12);
    REQUIRE(n2 == 30);
    REQUIRE(n1 == 0);
  }
  // At L=2: 12 corners + (30 + 60) = 90 ico-edge vertices + 60 interior = 162.
  {
    auto mesh = make_mesh(2);
    auto topo = icosphere_topology::build_from_mesh(*mesh);
    REQUIRE(topo.N_vert_s() == 162);
    int n5 = 0, n2 = 0, n1 = 0;
    for (int v = 0; v < topo.N_vert_s(); ++v) {
      switch (topo.vertex_valence(v)) {
        case 5: ++n5; break;
        case 2: ++n2; break;
        case 1: ++n1; break;
      }
    }
    REQUIRE(n5 == 12);
    REQUIRE(n2 == 90);
    REQUIRE(n1 == 60);
  }
}

TEST_CASE("icosphere topology: edge counts at L=1 and L=2 match expected "
          "interior / ico-edge split",
          "[prismatic][topology]") {
  // At L=1: 30 base ico-edges × 2 subdivide = 60 ico-edge edges (val 2),
  //         20 base faces × 3 new interior edges = 60 interior (val 1).
  {
    auto mesh = make_mesh(1);
    auto topo = icosphere_topology::build_from_mesh(*mesh);
    REQUIRE(topo.N_edge_s() == 120);
    int n2 = 0, n1 = 0;
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      if (topo.edge_valence(e) == 2) ++n2;
      else if (topo.edge_valence(e) == 1) ++n1;
    }
    REQUIRE(n2 == 60);
    REQUIRE(n1 == 60);
  }
  // At L=2: 60 · 2 = 120 ico-edge edges, remaining 360 interior.
  {
    auto mesh = make_mesh(2);
    auto topo = icosphere_topology::build_from_mesh(*mesh);
    REQUIRE(topo.N_edge_s() == 480);
    int n2 = 0, n1 = 0;
    for (int e = 0; e < topo.N_edge_s(); ++e) {
      if (topo.edge_valence(e) == 2) ++n2;
      else if (topo.edge_valence(e) == 1) ++n1;
    }
    REQUIRE(n2 == 120);
    REQUIRE(n1 == 360);
  }
}

TEST_CASE("icosphere topology: owner is the minimum incident ico-face index",
          "[prismatic][topology]") {
  auto mesh = make_mesh(2);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int v = 0; v < topo.N_vert_s(); ++v) {
    const int* faces = topo.vertex_ico_faces(v);
    int val = topo.vertex_valence(v);
    int minf = faces[0];
    for (int j = 1; j < val; ++j) minf = std::min(minf, faces[j]);
    REQUIRE(topo.vertex_owner_ico_face(v) == minf);
  }
  for (int e = 0; e < topo.N_edge_s(); ++e) {
    const int* faces = topo.edge_ico_faces(e);
    int val = topo.edge_valence(e);
    int minf = faces[0];
    for (int j = 1; j < val; ++j) minf = std::min(minf, faces[j]);
    REQUIRE(topo.edge_owner_ico_face(e) == minf);
  }
}

TEST_CASE("icosphere topology: incidence lists are sorted ascending and unique",
          "[prismatic][topology]") {
  auto mesh = make_mesh(2);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    const int* f = topo.vertex_ico_faces(v);
    int val = topo.vertex_valence(v);
    for (int j = 1; j < val; ++j) REQUIRE(f[j] > f[j - 1]);
  }
  for (int e = 0; e < topo.N_edge_s(); ++e) {
    const int* f = topo.edge_ico_faces(e);
    int val = topo.edge_valence(e);
    for (int j = 1; j < val; ++j) REQUIRE(f[j] > f[j - 1]);
  }
}

// =========================================================================
// Integrate topology with partition — full-fidelity cochain tiling.
// =========================================================================
template <typename PartVec, typename OwnFn>
static void check_tiling(const PartVec& parts, int global_size, OwnFn own) {
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

TEST_CASE("20-way angular decomposition with topology: all cochain types tile",
          "[prismatic][topology][tiling]") {
  const int L = 2;
  const int N_r = 4;

  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  std::vector<prismatic_partition> parts;
  for (int f = 0; f < 20; ++f) {
    auto p = prismatic_partition::ico_face_angular(L, N_r, f);
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

TEST_CASE("combined 4×20 decomposition with topology: all cochain types tile",
          "[prismatic][topology][tiling]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;

  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  std::vector<prismatic_partition> parts;
  parts.reserve(K * 20);
  for (int r = 0; r < K; ++r) {
    for (int f = 0; f < 20; ++f) {
      auto p = prismatic_partition::combined_ico_face(L, N_r, K, r, f);
      p.set_topology(&topo);
      parts.push_back(p);
    }
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
