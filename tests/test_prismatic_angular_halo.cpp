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
