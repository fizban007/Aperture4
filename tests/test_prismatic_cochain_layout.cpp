#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <chrono>
#include <memory>
#include <set>
#include <vector>

using namespace Aperture;

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, 2, 1.0, 2.0);
  return mesh;
}

}  // namespace

// =========================================================================
// Single-rank partition: local == global (no halos).
// =========================================================================
TEST_CASE("single-rank layout: local_size == global_size, no ghost",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    auto layout = distributed_cochain_layout::build(t, part, &topo, {});
    REQUIRE(layout.global_size() == global_cochain_size(t, part));
    REQUIRE(layout.owned_size() == layout.global_size());
    REQUIRE(layout.ghost_size() == 0);
    REQUIRE(layout.local_size() == layout.global_size());

    // Identity: to_global(l) == l and to_local(g) == g.
    for (int g = 0; g < layout.global_size(); ++g) {
      REQUIRE(layout.to_global(g) == g);
      REQUIRE(layout.to_local(g) == g);
    }
  }
}

// =========================================================================
// Radial decomposition: local layout covers owned + lower/upper ghosts.
// =========================================================================
TEST_CASE("radial layout: owned + ghost covers every index the plan touches",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 12;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    for (int r = 0; r < K; ++r) {
      auto part = prismatic_partition::radial_slab(L, N_r, K, r);
      part.set_topology(&topo);
      auto plan = build_radial_halo_plan(t, part);
      auto layout = distributed_cochain_layout::build(t, part, &topo, {&plan});

      // Every owned global index must be in the layout at a local slot
      // in [0, n_owned).
      int owned_count = 0;
      for (int g = 0; g < layout.global_size(); ++g) {
        bool owned = false;
        switch (t) {
          case cochain_type::tri_face:  owned = part.owns_tri_face_cochain(g); break;
          case cochain_type::rect_face: owned = part.owns_rect_face_cochain(g); break;
          case cochain_type::h_edge:    owned = part.owns_h_edge_cochain(g); break;
          case cochain_type::v_edge:    owned = part.owns_v_edge_cochain(g); break;
          case cochain_type::vertex:    owned = part.owns_vertex_cochain(g); break;
        }
        if (owned) {
          int l = layout.to_local(g);
          REQUIRE(l >= 0);
          REQUIRE(l < layout.owned_size());
          ++owned_count;
        }
      }
      REQUIRE(owned_count == layout.owned_size());

      // Every index mentioned in the plan's recv set must be a ghost.
      for (auto const& pe : plan.peers) {
        for (int g : pe.recv_global_idx) {
          int l = layout.to_local(g);
          REQUIRE(l >= layout.owned_size());
          REQUIRE(l < layout.local_size());
        }
      }

      // Round-trip: to_global(to_local(g)) == g for every global idx
      // present in the layout.
      for (int l = 0; l < layout.local_size(); ++l) {
        int g = layout.to_global(l);
        REQUIRE(layout.to_local(g) == l);
      }
    }
  }
}

// =========================================================================
// Angular decomposition: same coverage with angular plans.
// =========================================================================
TEST_CASE("angular layout: layout covers owned + angular ghost",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    for (int f = 0; f < 20; ++f) {
      auto part = prismatic_partition::ico_face_angular(L, N_r, f);
      part.set_topology(&topo);
      auto plan = build_angular_halo_plan(t, part, topo);
      auto layout = distributed_cochain_layout::build(t, part, &topo, {&plan});

      for (auto const& pe : plan.peers) {
        for (int g : pe.send_global_idx) {
          int l = layout.to_local(g);
          REQUIRE(l >= 0);
          REQUIRE(l < layout.owned_size());
        }
        for (int g : pe.recv_global_idx) {
          int l = layout.to_local(g);
          REQUIRE(l >= layout.owned_size());
          REQUIRE(l < layout.local_size());
        }
      }
    }
  }
}

// =========================================================================
// Combined (angular × radial) decomposition: layout absorbs BOTH the
// angular and radial ghost sets; owned + ghost sets are disjoint.
// =========================================================================
TEST_CASE("combined layout: owned and ghost sets are disjoint",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    for (int r = 0; r < K; ++r) {
      for (int f = 0; f < 20; ++f) {
        auto part = prismatic_partition::combined_ico_face(L, N_r, K, r, f);
        part.set_topology(&topo);
        auto plan_a = build_angular_halo_plan(t, part, topo);
        auto plan_r = build_radial_halo_plan(t, part);
        auto layout = distributed_cochain_layout::build(
            t, part, &topo, {&plan_a, &plan_r});

        // Owned and ghost are disjoint: no global index appears twice.
        std::set<int> seen;
        for (int l = 0; l < layout.local_size(); ++l) {
          int g = layout.to_global(l);
          REQUIRE(seen.find(g) == seen.end());
          seen.insert(g);
        }
        REQUIRE(int(seen.size()) == layout.local_size());
      }
    }
  }
}

// =========================================================================
// Plan translation: localize a global-indexed plan, verify every entry
// round-trips through the layout.
// =========================================================================
TEST_CASE("localize: translated indices round-trip via to_global",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto part = prismatic_partition::combined_ico_face(L, N_r, K, 1, 5);  // interior
  part.set_topology(&topo);
  auto plan_a_g = build_angular_halo_plan(cochain_type::tri_face, part, topo);
  auto plan_r_g = build_radial_halo_plan(cochain_type::tri_face, part);
  auto layout = distributed_cochain_layout::build(
      cochain_type::tri_face, part, &topo, {&plan_a_g, &plan_r_g});

  auto plan_a_l = layout.localize(plan_a_g);
  auto plan_r_l = layout.localize(plan_r_g);

  // Angular: same peer ordering, same peer_rank, round-tripped indices.
  REQUIRE(plan_a_l.peers.size() == plan_a_g.peers.size());
  for (size_t p = 0; p < plan_a_g.peers.size(); ++p) {
    REQUIRE(plan_a_l.peers[p].peer_rank == plan_a_g.peers[p].peer_rank);
    REQUIRE(plan_a_l.peers[p].send_global_idx.size() ==
            plan_a_g.peers[p].send_global_idx.size());
    for (size_t i = 0; i < plan_a_g.peers[p].send_global_idx.size(); ++i) {
      int g = plan_a_g.peers[p].send_global_idx[i];
      int l = plan_a_l.peers[p].send_global_idx[i];
      REQUIRE(layout.to_global(l) == g);
    }
    for (size_t i = 0; i < plan_a_g.peers[p].recv_global_idx.size(); ++i) {
      int g = plan_a_g.peers[p].recv_global_idx[i];
      int l = plan_a_l.peers[p].recv_global_idx[i];
      REQUIRE(layout.to_global(l) == g);
    }
  }

  // Radial: same round-trip check.
  REQUIRE(plan_r_l.peers.size() == plan_r_g.peers.size());
  for (size_t p = 0; p < plan_r_g.peers.size(); ++p) {
    REQUIRE(plan_r_l.peers[p].peer_rank == plan_r_g.peers[p].peer_rank);
    for (size_t i = 0; i < plan_r_g.peers[p].send_global_idx.size(); ++i) {
      int g = plan_r_g.peers[p].send_global_idx[i];
      int l = plan_r_l.peers[p].send_global_idx[i];
      REQUIRE(layout.to_global(l) == g);
    }
    for (size_t i = 0; i < plan_r_g.peers[p].recv_global_idx.size(); ++i) {
      int g = plan_r_g.peers[p].recv_global_idx[i];
      int l = plan_r_l.peers[p].recv_global_idx[i];
      REQUIRE(layout.to_global(l) == g);
    }
  }
}

// =========================================================================
// End-to-end: local-indexed plan produces bit-identical exchange result
// as the global-indexed plan did in Phase 2 / Phase 3 tests.  This is
// the real payoff: the mpi_halo_backend and in_process_halo_backend
// operate on indices without knowing whether they're global or local.
// =========================================================================
TEST_CASE("local-indexed plan: in-process exchange fills ghost slots",
          "[prismatic][layout][in_process]") {
  using Catch::Approx;
  const int L = 2;
  const int N_r = 12;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto stamp = [](int global) {
    return Scalar(1) + Scalar(global) * Scalar(1e-4);
  };

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge}) {
    std::vector<prismatic_partition> parts;
    std::vector<halo_plan> global_plans;
    std::vector<distributed_cochain_layout> layouts;
    std::vector<halo_plan> local_plans;
    std::vector<std::vector<Scalar>> bufs(K);
    in_process_halo_backend backend;

    for (int r = 0; r < K; ++r) {
      auto p = prismatic_partition::radial_slab(L, N_r, K, r);
      p.set_topology(&topo);
      auto gp = build_radial_halo_plan(t, p);
      auto layout = distributed_cochain_layout::build(t, p, &topo, {&gp});
      auto lp = layout.localize(gp);
      parts.push_back(p);
      global_plans.push_back(std::move(gp));
      layouts.push_back(std::move(layout));
      local_plans.push_back(std::move(lp));
    }

    // Fill LOCAL buffers: owned slots get the stamp of their global
    // index; ghost slots zero.
    for (int r = 0; r < K; ++r) {
      auto const& L_ = layouts[r];
      bufs[r].assign(L_.local_size(), Scalar(0));
      for (int l = 0; l < L_.owned_size(); ++l) {
        bufs[r][l] = stamp(L_.to_global(l));
      }
      backend.register_rank(r, bufs[r].data());
    }

    backend.exchange_all(local_plans);

    // Every ghost slot now carries the stamp of its global index.
    for (int r = 0; r < K; ++r) {
      auto const& L_ = layouts[r];
      for (int l = L_.owned_size(); l < L_.local_size(); ++l) {
        REQUIRE(bufs[r][l] == Approx(stamp(L_.to_global(l))));
      }
      // Owned slots untouched.
      for (int l = 0; l < L_.owned_size(); ++l) {
        REQUIRE(bufs[r][l] == Approx(stamp(L_.to_global(l))));
      }
    }
  }
}

// =========================================================================
// Factorized owned enumeration (checkpoint plan appendix item 2): the
// (owned shell/slab range × owned sphere list) enumeration must produce
// EXACTLY the set the full O(global) ownership scan produced, in the
// same ascending order, for every cochain on every rank of assorted
// A×K shapes (including a patch level above 0 and pure-radial).
// =========================================================================
TEST_CASE("layout owned enumeration equals the brute-force ownership scan",
          "[prismatic][layout]") {
  const int L = 2;
  const int N_r = 8;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto scan_owned = [&](cochain_type t, const prismatic_partition& p) {
    std::vector<int> v;
    const int n = global_cochain_size(t, p);
    for (int g = 0; g < n; ++g) {
      bool o = false;
      switch (t) {
        case cochain_type::tri_face:  o = p.owns_tri_face_cochain(g); break;
        case cochain_type::rect_face: o = p.owns_rect_face_cochain(g); break;
        case cochain_type::h_edge:    o = p.owns_h_edge_cochain(g); break;
        case cochain_type::v_edge:    o = p.owns_v_edge_cochain(g); break;
        case cochain_type::vertex:    o = p.owns_vertex_cochain(g); break;
      }
      if (o) v.push_back(g);
    }
    return v;
  };

  struct shape { int A, K, m; };
  for (auto sh : {shape{1, 4, -1}, shape{4, 2, -1}, shape{8, 2, -1},
                  shape{20, 2, -1}, shape{8, 2, 2}, shape{16, 1, -1}}) {
    for (int w = 0; w < sh.A * sh.K; ++w) {
      auto part = prismatic_partition::combined(L, N_r, sh.A, sh.K, w, sh.m);
      part.set_topology(&topo);
      for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                             cochain_type::h_edge, cochain_type::v_edge,
                             cochain_type::vertex}) {
        auto layout = distributed_cochain_layout::build(t, part, &topo, {});
        const auto ref = scan_owned(t, part);
        REQUIRE(layout.owned_size() == int(ref.size()));
        for (size_t i = 0; i < ref.size(); ++i) {
          REQUIRE(layout.to_global(int(i)) == ref[i]);
        }
      }
    }
  }
}

// =========================================================================
// Hidden benchmark ([.]): isolates the mesh_partition::build cost (5
// layouts + 10 plans + localization) at a production-ish scale.  Run
// explicitly with:  ./tests '[layout_bench]'
// =========================================================================
TEST_CASE("layout build timing at L6", "[.][layout_bench]") {
  const int L = 6, N_r = 204;
  prismatic_mesh mesh;
  mesh.build_sphere_only(L, N_r, 1.0, 45.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  auto t0 = std::chrono::steady_clock::now();
  auto part = prismatic_partition::combined(L, N_r, 8, 2, 0);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo, halo_depth::pic);
  auto t1 = std::chrono::steady_clock::now();
  WARN("mesh_partition::build(L=6, N_r=204, 8x2, pic) took "
       << std::chrono::duration<double>(t1 - t0).count() << " s; owned tri "
       << mp.layout(cochain_type::tri_face).owned_size());
}
