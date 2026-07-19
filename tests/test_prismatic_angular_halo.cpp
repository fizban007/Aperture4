#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_partition.h"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>

using namespace Aperture;
using Catch::Approx;

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, 2, 1.0, 2.0);
  return mesh;
}

std::vector<prismatic_partition>
make_20_angular(int L, int N_r, const icosphere_topology& topo) {
  std::vector<prismatic_partition> out;
  out.reserve(20);
  for (int f = 0; f < 20; ++f) {
    auto p = prismatic_partition::ico_face_angular(L, N_r, f);
    p.set_topology(&topo);
    out.push_back(p);
  }
  return out;
}

int global_width(cochain_type t, const icosphere_topology& topo) {
  switch (t) {
    case cochain_type::tri_face:  return topo.N_tri();
    case cochain_type::h_edge:    return topo.N_edge_s();
    case cochain_type::rect_face: return topo.N_edge_s();
    case cochain_type::v_edge:    return topo.N_vert_s();
    case cochain_type::vertex:    return topo.N_vert_s();
  }
  return 0;
}

}  // namespace

// =========================================================================
// Angular peer set: each rank has at most 3 edge-neighbor peers plus up
// to 6 vertex-diagonal peers.
// =========================================================================
TEST_CASE("angular halo plan: peer counts are consistent with ico-adjacency",
          "[prismatic][angular_halo]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_20_angular(L, N_r, topo);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::rect_face, cochain_type::v_edge,
                         cochain_type::vertex}) {
    for (auto const& p : parts) {
      auto plan = build_angular_halo_plan(t, p, topo);
      std::set<int> peers;
      for (auto const& pe : plan.peers) peers.insert(pe.peer_rank);
      // Every peer must be one of the 9 topological neighbors.
      const auto& en = prismatic_partition::edge_neighbors();
      const auto& dn = prismatic_partition::diagonal_neighbors();
      std::set<int> allowed;
      int F = p.ico_face_lo;
      for (int g : en[F]) allowed.insert(g);
      for (int g : dn[F]) allowed.insert(g);
      for (int pr : peers) REQUIRE(allowed.count(pr) == 1);

      // Each peer entry must have at least one of send or recv non-empty.
      for (auto const& pe : plan.peers) {
        REQUIRE((!pe.send_global_idx.empty() ||
                 !pe.recv_global_idx.empty()));
      }
    }
  }
}

// =========================================================================
// Send/recv symmetry: what A says it sends to B must match what B says
// it receives from A (set equality, order-independent).
// =========================================================================
TEST_CASE("angular halo plan: send_i→j == recv_j←i for every cochain type",
          "[prismatic][angular_halo]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_20_angular(L, N_r, topo);

  auto find_peer = [](const halo_plan& p, int pr) {
    for (auto const& pe : p.peers)
      if (pe.peer_rank == pr) return &pe;
    return (const halo_plan::peer_entry*)nullptr;
  };

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::rect_face, cochain_type::v_edge,
                         cochain_type::vertex}) {
    std::vector<halo_plan> plans;
    plans.reserve(20);
    for (auto const& p : parts) plans.push_back(build_angular_halo_plan(t, p, topo));

    for (int f = 0; f < 20; ++f) {
      for (auto const& pe : plans[f].peers) {
        int g = pe.peer_rank;
        auto* mirror = find_peer(plans[g], f);
        // Sort both sides for set comparison.
        auto sort_copy = [](std::vector<int> v) {
          std::sort(v.begin(), v.end());
          return v;
        };
        if (pe.send_global_idx.empty()) {
          // g doesn't recv from f for this cochain.
          if (mirror != nullptr) {
            REQUIRE(mirror->recv_global_idx.empty());
          }
        } else {
          REQUIRE(mirror != nullptr);
          REQUIRE(sort_copy(pe.send_global_idx) ==
                  sort_copy(mirror->recv_global_idx));
        }
        if (pe.recv_global_idx.empty()) {
          if (mirror != nullptr) {
            REQUIRE(mirror->send_global_idx.empty());
          }
        } else {
          REQUIRE(mirror != nullptr);
          REQUIRE(sort_copy(pe.recv_global_idx) ==
                  sort_copy(mirror->send_global_idx));
        }
      }
    }
  }
}

// =========================================================================
// Ownership via halo: if rank F does not own a cochain but receives it
// via halo, after exchange the halo value must match what the owner had.
// =========================================================================

static Scalar stamp(int global_idx) {
  return Scalar(1) + Scalar(global_idx) * Scalar(1e-4);
}

static void fill_owned(std::vector<Scalar>& buf, cochain_type t,
                        const prismatic_partition& p) {
  int N = global_cochain_size(t, p);
  buf.assign(N, Scalar(0));
  for (int g = 0; g < N; ++g) {
    bool owned = false;
    switch (t) {
      case cochain_type::tri_face:  owned = p.owns_tri_face_cochain(g);  break;
      case cochain_type::rect_face: owned = p.owns_rect_face_cochain(g); break;
      case cochain_type::h_edge:    owned = p.owns_h_edge_cochain(g);    break;
      case cochain_type::v_edge:    owned = p.owns_v_edge_cochain(g);    break;
      case cochain_type::vertex:    owned = p.owns_vertex_cochain(g);    break;
    }
    if (owned) buf[g] = stamp(g);
  }
}

TEST_CASE("angular halo exchange populates ghost slots with owner values",
          "[prismatic][angular_halo][in_process]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_20_angular(L, N_r, topo);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::rect_face, cochain_type::v_edge,
                         cochain_type::vertex}) {
    std::vector<std::vector<Scalar>> bufs(20);
    in_process_halo_backend backend;
    for (int f = 0; f < 20; ++f) {
      fill_owned(bufs[f], t, parts[f]);
      backend.register_rank(f, bufs[f].data());
    }

    std::vector<halo_plan> plans;
    plans.reserve(20);
    for (auto const& p : parts) plans.push_back(build_angular_halo_plan(t, p, topo));

    backend.exchange_all(plans);

    // Every recv slot now holds the owner's stamped value.
    for (int f = 0; f < 20; ++f) {
      for (auto const& pe : plans[f].peers) {
        for (int idx : pe.recv_global_idx) {
          REQUIRE(bufs[f][idx] == Approx(stamp(idx)));
        }
      }
    }

    // Owned slots are untouched (still stamped with their global-index value).
    int N = global_cochain_size(t, parts[0]);
    for (int f = 0; f < 20; ++f) {
      for (int g = 0; g < N; ++g) {
        bool owned = false;
        switch (t) {
          case cochain_type::tri_face:  owned = parts[f].owns_tri_face_cochain(g);  break;
          case cochain_type::rect_face: owned = parts[f].owns_rect_face_cochain(g); break;
          case cochain_type::h_edge:    owned = parts[f].owns_h_edge_cochain(g);    break;
          case cochain_type::v_edge:    owned = parts[f].owns_v_edge_cochain(g);    break;
          case cochain_type::vertex:    owned = parts[f].owns_vertex_cochain(g);    break;
        }
        if (owned) {
          REQUIRE(bufs[f][g] == Approx(stamp(g)));
        }
      }
    }
  }
}

// =========================================================================
// After exchange, for each cochain type, the union of (owned ∪ received)
// on every rank should cover every index that rank is incident to in
// the DEC stencil sense.  This is a weaker check than end-to-end
// correctness but catches missing halo cases on valence-5 corners.
// =========================================================================
// =========================================================================
// Phase 3.3 — valence-5 corner rect_face halo.
//
// The v_owner at a valence-5 corner computes dD/dt on its v_edge using
// the fan of 5 rect faces radiating from the corner.  Only 2 of those
// 5 are at sphere-edges the v_owner is topologically incident to; the
// other 3 need extra halos from diagonal neighbors.  After exchange,
// every fan rect face must have the stamped value on the v_owner's
// buffer.
// =========================================================================
TEST_CASE("valence-5 corners: rect_face fan is fully halo'd at v_owner",
          "[prismatic][angular_halo][valence5]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_20_angular(L, N_r, topo);

  // Collect all valence-5 corners.
  std::vector<int> corners;
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    if (topo.vertex_valence(v) == 5) corners.push_back(v);
  }
  REQUIRE(corners.size() == 12);  // canonical icosahedron vertex count

  // Stamp-filled buffers + exchange for rect_face.
  std::vector<std::vector<Scalar>> bufs(20);
  in_process_halo_backend backend;
  for (int f = 0; f < 20; ++f) {
    fill_owned(bufs[f], cochain_type::rect_face, parts[f]);
    backend.register_rank(f, bufs[f].data());
  }
  std::vector<halo_plan> plans;
  plans.reserve(20);
  for (auto const& p : parts)
    plans.push_back(build_angular_halo_plan(cochain_type::rect_face, p, topo));
  backend.exchange_all(plans);

  // For every valence-5 corner, the v_owner must have every fan
  // sphere-edge's rect_face value filled at every slab.
  for (int v : corners) {
    const int* incs = topo.vertex_ico_faces(v);
    const int v_owner = incs[0];
    const int ve_count = topo.vertex_edge_count(v);
    REQUIRE(ve_count == 5);
    const int* ve = topo.vertex_edges(v);

    for (int i = 0; i < ve_count; ++i) {
      int e = ve[i];
      for (int slab = 0; slab < N_r; ++slab) {
        int idx = slab * topo.N_edge_s() + e;
        // Either v_owner owns this rect_face (then stamped from the
        // start) or the halo must have populated it.
        REQUIRE(bufs[v_owner][idx] == Approx(stamp(idx)));
      }
    }
  }
}

// =========================================================================
// Phase 7A.3 — generic unit-based angular halo plan builder.
// =========================================================================

namespace {

const cochain_type kAllCochains[] = {
    cochain_type::tri_face, cochain_type::h_edge, cochain_type::rect_face,
    cochain_type::v_edge, cochain_type::vertex};

std::vector<prismatic_partition>
make_unit_angular(int L, int N_r, int A, const icosphere_topology& topo) {
  std::vector<prismatic_partition> out;
  out.reserve(A);
  for (int a = 0; a < A; ++a) {
    auto p = prismatic_partition::angular_units(L, N_r, A, a);
    p.set_topology(&topo);
    out.push_back(p);
  }
  return out;
}

std::vector<int> sorted_unique(std::vector<int> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

const halo_plan::peer_entry* find_peer(const halo_plan& p, int pr) {
  for (auto const& pe : p.peers)
    if (pe.peer_rank == pr) return &pe;
  return nullptr;
}

}  // namespace

TEST_CASE("generic unit builder at A=20/m=0 reproduces the legacy per-face "
          "plans exactly (up to the path rank relabeling)",
          "[prismatic][angular_halo][units]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto legacy_parts = make_20_angular(L, N_r, topo);
  auto unit_parts = make_unit_angular(L, N_r, 20, topo);
  const auto& pos = prismatic_partition::face_path_pos();

  for (cochain_type t : kAllCochains) {
    for (int f = 0; f < 20; ++f) {
      auto legacy = build_angular_halo_plan(t, legacy_parts[f], topo);
      auto unit = build_angular_halo_plan_units(t, unit_parts[pos[f]], topo);

      // Same peer set under the face→path-position relabeling, same
      // send/recv index sets per peer.
      REQUIRE(unit.peers.size() == legacy.peers.size());
      for (auto const& lpe : legacy.peers) {
        auto* upe = find_peer(unit, pos[lpe.peer_rank]);
        REQUIRE(upe != nullptr);
        REQUIRE(upe->send_global_idx ==
                sorted_unique(lpe.send_global_idx));
        REQUIRE(upe->recv_global_idx ==
                sorted_unique(lpe.recv_global_idx));
      }
    }
  }
}

TEST_CASE("generic unit builder: send/recv lists are wire-consistent and "
          "canonically ordered for general A",
          "[prismatic][angular_halo][units]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int A : {2, 4, 5, 8, 10, 40, 80}) {
    auto parts = make_unit_angular(L, N_r, A, topo);
    for (cochain_type t : kAllCochains) {
      std::vector<halo_plan> plans;
      for (auto const& p : parts)
        plans.push_back(build_angular_halo_plan_units(t, p, topo));

      for (int a = 0; a < A; ++a) {
        for (auto const& pe : plans[a].peers) {
          // Sorted ascending, unique — the canonical wire order.
          REQUIRE(pe.send_global_idx == sorted_unique(pe.send_global_idx));
          REQUIRE(pe.recv_global_idx == sorted_unique(pe.recv_global_idx));
          // Exact list match with the mirror side (not just set match:
          // MPI pairs entries positionally).
          auto* mirror = find_peer(plans[pe.peer_rank], a);
          if (!pe.recv_global_idx.empty()) {
            REQUIRE(mirror != nullptr);
            REQUIRE(pe.recv_global_idx == mirror->send_global_idx);
          }
          if (!pe.send_global_idx.empty()) {
            REQUIRE(mirror != nullptr);
            REQUIRE(pe.send_global_idx == mirror->recv_global_idx);
          }
        }
      }
    }
  }
}

TEST_CASE("generic unit builder: exchange fills ghosts with owner values "
          "and owned slots stay untouched, general A",
          "[prismatic][angular_halo][units][in_process]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int A : {4, 8, 80}) {
    auto parts = make_unit_angular(L, N_r, A, topo);
    for (cochain_type t : kAllCochains) {
      std::vector<std::vector<Scalar>> bufs(A);
      in_process_halo_backend backend;
      for (int a = 0; a < A; ++a) {
        fill_owned(bufs[a], t, parts[a]);
        backend.register_rank(a, bufs[a].data());
      }
      std::vector<halo_plan> plans;
      for (auto const& p : parts)
        plans.push_back(build_angular_halo_plan_units(t, p, topo));
      backend.exchange_all(plans);

      for (int a = 0; a < A; ++a) {
        for (auto const& pe : plans[a].peers) {
          for (int idx : pe.recv_global_idx) {
            REQUIRE(bufs[a][idx] == Approx(stamp(idx)));
          }
        }
      }
    }
  }
}

TEST_CASE("generic unit builder: solver stencils are fully covered after "
          "exchange (owned or ghosted) for general A",
          "[prismatic][angular_halo][units][in_process]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (int A : {4, 8, 80}) {
    auto parts = make_unit_angular(L, N_r, A, topo);

    // Exchange every cochain type on every rank.
    std::vector<std::vector<std::vector<Scalar>>> bufs(5);
    for (int ti = 0; ti < 5; ++ti) {
      cochain_type t = kAllCochains[ti];
      bufs[ti].resize(A);
      in_process_halo_backend backend;
      for (int a = 0; a < A; ++a) {
        fill_owned(bufs[ti][a], t, parts[a]);
        backend.register_rank(a, bufs[ti][a].data());
      }
      std::vector<halo_plan> plans;
      for (auto const& p : parts)
        plans.push_back(build_angular_halo_plan_units(t, p, topo));
      backend.exchange_all(plans);
    }
    auto& tri_bufs = bufs[0];
    auto& h_edge_bufs = bufs[1];
    auto& rect_bufs = bufs[2];
    auto& v_edge_bufs = bufs[3];
    auto& vert_bufs = bufs[4];

    const int W_tri = topo.N_tri();
    const int W_e = topo.N_edge_s();
    const int W_v = topo.N_vert_s();

    for (int a = 0; a < A; ++a) {
      auto const& p = parts[a];
      // (1) d1t on an owned sphere-edge reads both adjacent tri faces,
      //     at every shell.
      for (int e = 0; e < W_e; ++e) {
        if (!p.owns_sphere_edge(e)) continue;
        for (int k = 0; k <= N_r; ++k) {
          REQUIRE(tri_bufs[a][k * W_tri + topo.edge_tri_a(e)] ==
                  Approx(stamp(k * W_tri + topo.edge_tri_a(e))));
          REQUIRE(tri_bufs[a][k * W_tri + topo.edge_tri_b(e)] ==
                  Approx(stamp(k * W_tri + topo.edge_tri_b(e))));
        }
      }
      // (2) d1 on an owned tri face reads its incident sphere-edges:
      //     equivalently every edge adjacent to an owned tri must be
      //     available as h_edge (shells) and rect_face (slabs).
      for (int e = 0; e < W_e; ++e) {
        bool touches_owned_tri = p.owns_sub_tri(topo.edge_tri_a(e)) ||
                                 p.owns_sub_tri(topo.edge_tri_b(e));
        if (!touches_owned_tri) continue;
        for (int k = 0; k <= N_r; ++k) {
          REQUIRE(h_edge_bufs[a][k * W_e + e] == Approx(stamp(k * W_e + e)));
        }
        for (int k = 0; k < N_r; ++k) {
          REQUIRE(rect_bufs[a][k * W_e + e] == Approx(stamp(k * W_e + e)));
        }
      }
      // (3) the v_edge update at an owned sphere-vertex reads the FULL
      //     fan of rect faces (the valence-5 corner case generalized to
      //     arbitrary unit corners).
      for (int v = 0; v < W_v; ++v) {
        if (!p.owns_sphere_vertex(v)) continue;
        const int* ve = topo.vertex_edges(v);
        for (int i = 0; i < topo.vertex_edge_count(v); ++i) {
          for (int k = 0; k < N_r; ++k) {
            int idx = k * W_e + ve[i];
            REQUIRE(rect_bufs[a][idx] == Approx(stamp(idx)));
          }
        }
      }
      // (4) d1 on an owned h_edge reads its 2 endpoint vertices; owned
      //     rect faces read the v_edges at their endpoint vertices.
      for (int e = 0; e < W_e; ++e) {
        if (!p.owns_sphere_edge(e)) continue;
        for (int v : {topo.edge_v0(e), topo.edge_v1(e)}) {
          for (int k = 0; k <= N_r; ++k) {
            REQUIRE(vert_bufs[a][k * W_v + v] == Approx(stamp(k * W_v + v)));
          }
          for (int k = 0; k < N_r; ++k) {
            REQUIRE(v_edge_bufs[a][k * W_v + v] ==
                    Approx(stamp(k * W_v + v)));
          }
        }
      }
    }
  }
}

TEST_CASE("generic unit builder: peers are unit-adjacent and pic depth "
          "class is rejected until 7B",
          "[prismatic][angular_halo][units]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_unit_angular(L, N_r, 8, topo);

  REQUIRE_THROWS_AS(
      build_angular_halo_plan_units(cochain_type::vertex, parts[0], topo,
                                    halo_depth::pic),
      std::invalid_argument);

  // A=1 (full angular span): no angular peers.
  auto p1 = prismatic_partition::angular_units(L, N_r, 1, 0);
  p1.set_topology(&topo);
  auto plan =
      build_angular_halo_plan_units(cochain_type::tri_face, p1, topo);
  REQUIRE(plan.peers.empty());
}

TEST_CASE("angular halo at valence-5 corners: non-owner recvs from owner",
          "[prismatic][angular_halo]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto parts = make_20_angular(L, N_r, topo);

  // Find a valence-5 vertex and its 5 incident ico-faces.
  int v_corner = -1;
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    if (topo.vertex_valence(v) == 5) { v_corner = v; break; }
  }
  REQUIRE(v_corner >= 0);

  std::vector<int> incidents(topo.vertex_ico_faces(v_corner),
                             topo.vertex_ico_faces(v_corner) + 5);
  int owner = incidents[0];  // min (sorted)

  // Build vertex-cochain plans for the 5 incident ranks.
  for (int f : incidents) {
    auto plan = build_angular_halo_plan(cochain_type::vertex, parts[f], topo);
    if (f == owner) {
      // Owner sends vertex v to each of the other 4 incidents.
      int sends_to_non_owners = 0;
      for (auto const& pe : plan.peers) {
        bool has_v = false;
        for (int idx : pe.send_global_idx) {
          if (idx % topo.N_vert_s() == v_corner) { has_v = true; break; }
        }
        if (has_v) ++sends_to_non_owners;
      }
      REQUIRE(sends_to_non_owners == 4);
    } else {
      // Non-owner receives vertex v from exactly the owner.
      int recvs_of_v_from_owner = 0;
      int recvs_of_v_total = 0;
      for (auto const& pe : plan.peers) {
        for (int idx : pe.recv_global_idx) {
          if (idx % topo.N_vert_s() == v_corner) {
            ++recvs_of_v_total;
            if (pe.peer_rank == owner) ++recvs_of_v_from_owner;
          }
        }
      }
      REQUIRE(recvs_of_v_total == recvs_of_v_from_owner);
      REQUIRE(recvs_of_v_total >= 1);
    }
  }
}
