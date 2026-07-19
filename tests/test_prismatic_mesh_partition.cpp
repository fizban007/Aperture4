#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_cochain_layout.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <memory>
#include <vector>

using namespace Aperture;
using Catch::Approx;

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, 2, 1.0, 2.0);
  return mesh;
}

}  // namespace

// =========================================================================
// build(): layouts and plans are consistent with independent construction.
// =========================================================================
TEST_CASE("mesh_partition build: matches independent layout + plan builders",
          "[prismatic][mesh_partition]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto part = prismatic_partition::combined_ico_face(L, N_r, K, 1, 5);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    // Independently-built plans.
    auto plan_a = build_angular_halo_plan(t, part, topo);
    auto plan_r = build_radial_halo_plan(t, part);
    auto layout =
        distributed_cochain_layout::build(t, part, &topo, {&plan_a, &plan_r});

    // Layout matches.
    REQUIRE(mp.layout(t).local_size() == layout.local_size());
    REQUIRE(mp.layout(t).owned_size() == layout.owned_size());
    REQUIRE(mp.layout(t).ghost_size() == layout.ghost_size());
    REQUIRE(mp.layout(t).global_size() == layout.global_size());

    // Global plans match.
    REQUIRE(mp.angular_plan_global(t).peers.size() == plan_a.peers.size());
    REQUIRE(mp.radial_plan_global(t).peers.size()  == plan_r.peers.size());

    // Local plans round-trip through the layout.
    for (auto const& pe : mp.angular_plan_local(t).peers) {
      for (int l : pe.send_global_idx) {
        REQUIRE(l >= 0);
        REQUIRE(l < mp.layout(t).local_size());
      }
      for (int l : pe.recv_global_idx) {
        REQUIRE(l >= mp.layout(t).owned_size());
        REQUIRE(l < mp.layout(t).local_size());
      }
    }
  }
}

// =========================================================================
// copy_global_to_local: a global buffer's local portion matches.
// =========================================================================
TEST_CASE("copy_global_to_local: local buffer matches global at mapped indices",
          "[prismatic][mesh_partition]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto part = prismatic_partition::combined_ico_face(L, N_r, K, 2, 11);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::rect_face,
                         cochain_type::h_edge, cochain_type::v_edge,
                         cochain_type::vertex}) {
    auto const& layout = mp.layout(t);
    const int N_global = layout.global_size();
    const int N_local = layout.local_size();

    // Fill a global buffer with known stamps.
    std::vector<Scalar> global_buf(N_global);
    for (int g = 0; g < N_global; ++g) {
      global_buf[g] = Scalar(1) + Scalar(g) * Scalar(1e-4);
    }

    // Copy into local.
    std::vector<Scalar> local_buf(N_local, Scalar(-999));
    copy_global_to_local(global_buf.data(), layout, local_buf.data());

    // For every local slot, local[l] == global[to_global(l)].
    for (int l = 0; l < N_local; ++l) {
      REQUIRE(local_buf[l] == Approx(global_buf[layout.to_global(l)]));
    }
  }
}

// =========================================================================
// End-to-end: global-indexed exchange and local-indexed exchange produce
// equivalent state on a 20-way angular decomposition.
// =========================================================================
TEST_CASE("mesh_partition local exchange matches global exchange",
          "[prismatic][mesh_partition][in_process]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  // Build 20 angular partitions + their bundles.
  std::vector<prismatic_mesh_partition> bundles;
  bundles.reserve(20);
  for (int f = 0; f < 20; ++f) {
    auto part = prismatic_partition::ico_face_angular(L, N_r, f);
    part.set_topology(&topo);
    bundles.push_back(prismatic_mesh_partition::build(part, topo));
  }

  auto stamp = [](int global) {
    return Scalar(1) + Scalar(global) * Scalar(1e-4);
  };

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::rect_face, cochain_type::v_edge,
                         cochain_type::vertex}) {
    // --- Reference run: global-indexed exchange ---
    std::vector<std::vector<Scalar>> gbufs(20);
    in_process_halo_backend gbackend;
    std::vector<halo_plan> gplans;
    gplans.reserve(20);
    for (int f = 0; f < 20; ++f) {
      const int N = global_cochain_size(t, bundles[f].partition());
      gbufs[f].assign(N, Scalar(0));
      auto const& layout = bundles[f].layout(t);
      for (int l = 0; l < layout.owned_size(); ++l) {
        gbufs[f][layout.to_global(l)] = stamp(layout.to_global(l));
      }
      gbackend.register_rank(f, gbufs[f].data());
      gplans.push_back(bundles[f].angular_plan_global(t));
    }
    gbackend.exchange_all(gplans);

    // --- Local-indexed exchange ---
    std::vector<std::vector<Scalar>> lbufs(20);
    in_process_halo_backend lbackend;
    std::vector<halo_plan> lplans;
    lplans.reserve(20);
    for (int f = 0; f < 20; ++f) {
      auto const& layout = bundles[f].layout(t);
      lbufs[f].assign(layout.local_size(), Scalar(0));
      for (int l = 0; l < layout.owned_size(); ++l) {
        lbufs[f][l] = stamp(layout.to_global(l));
      }
      lbackend.register_rank(f, lbufs[f].data());
      lplans.push_back(bundles[f].angular_plan_local(t));
    }
    lbackend.exchange_all(lplans);

    // --- Compare: for every local slot, local[l] should equal
    //     global[to_global(l)].
    for (int f = 0; f < 20; ++f) {
      auto const& layout = bundles[f].layout(t);
      for (int l = 0; l < layout.local_size(); ++l) {
        int g = layout.to_global(l);
        REQUIRE(lbufs[f][l] == Approx(gbufs[f][g]));
      }
    }
  }
}
