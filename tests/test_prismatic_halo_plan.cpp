#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_partition.h"
#include <set>
#include <vector>

using namespace Aperture;
using Catch::Approx;

// =========================================================================
// Build a set of partitions covering a simulated K-slab radial decomposition
// at subdivision level L on N_r shells total.
// =========================================================================
static std::vector<prismatic_partition>
make_radial_slabs(int L, int N_r, int K) {
  std::vector<prismatic_partition> out;
  out.reserve(K);
  for (int r = 0; r < K; ++r) {
    out.push_back(prismatic_partition::radial_slab(L, N_r, K, r));
  }
  return out;
}

// =========================================================================
// Plan construction invariants.
// =========================================================================
TEST_CASE("radial halo plan: owned shell ranges tile the mesh",
          "[prismatic][halo_plan]") {
  const int L = 2, N_r = 12, K = 4;
  auto parts = make_radial_slabs(L, N_r, K);

  // Shells live at k ∈ [0, N_r] — that's N_r + 1 shells.  Slabs are at
  // k ∈ [0, N_r).
  std::vector<bool> shell_covered(N_r + 1, false);
  std::vector<bool> slab_covered(N_r, false);
  for (auto const& p : parts) {
    for (int k = p.shell_k_lo; k < p.shell_k_hi; ++k) {
      REQUIRE(k >= 0);
      REQUIRE(k <= N_r);
      REQUIRE_FALSE(shell_covered[k]);
      shell_covered[k] = true;
    }
    for (int k = 0; k < N_r; ++k) {
      if (p.owns_slab(k)) {
        REQUIRE_FALSE(slab_covered[k]);
        slab_covered[k] = true;
      }
    }
  }
  for (bool c : shell_covered) REQUIRE(c);
  for (bool c : slab_covered) REQUIRE(c);
}

TEST_CASE("radial halo plan: peer entries agree with neighbor ownership",
          "[prismatic][halo_plan]") {
  const int L = 2, N_r = 16, K = 4;
  auto parts = make_radial_slabs(L, N_r, K);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::vertex, cochain_type::rect_face,
                         cochain_type::v_edge}) {
    for (int r = 0; r < K; ++r) {
      auto plan = build_radial_halo_plan(t, parts[r]);

      // Interior ranks have 2 peers; boundary ranks have 1.
      int expected_peers = (r == 0 || r == K - 1) ? 1 : 2;
      REQUIRE(int(plan.peers.size()) == expected_peers);

      for (auto const& pe : plan.peers) {
        REQUIRE(pe.peer_rank >= 0);
        REQUIRE(pe.peer_rank < K);
        REQUIRE(std::abs(pe.peer_rank - r) == 1);  // only direct neighbors
      }
    }
  }
}

TEST_CASE("radial halo plan: send indices of rank r "
          "match recv indices of peer for shell cochains",
          "[prismatic][halo_plan]") {
  const int L = 3, N_r = 20, K = 5;
  auto parts = make_radial_slabs(L, N_r, K);
  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::vertex}) {
    for (int r = 0; r < K - 1; ++r) {
      auto plan_lo = build_radial_halo_plan(t, parts[r]);      // lower side
      auto plan_hi = build_radial_halo_plan(t, parts[r + 1]);  // upper side

      // Find the entry in plan_lo for peer r+1 and in plan_hi for peer r.
      auto find_peer = [](const halo_plan& p, int pr) {
        for (auto const& pe : p.peers)
          if (pe.peer_rank == pr) return &pe;
        return (const halo_plan::peer_entry*)nullptr;
      };
      auto* upper_of_lo = find_peer(plan_lo, r + 1);
      auto* lower_of_hi = find_peer(plan_hi, r);
      REQUIRE(upper_of_lo != nullptr);
      REQUIRE(lower_of_hi != nullptr);
      // Lower side sends to upper side's lower ghost == upper side's recv.
      REQUIRE(upper_of_lo->send_global_idx == lower_of_hi->recv_global_idx);
      // And upper side sends to lower side's upper ghost.
      REQUIRE(lower_of_hi->send_global_idx == upper_of_lo->recv_global_idx);
    }
  }
}

TEST_CASE("radial halo plan: slab cochain has only one-directional exchange "
          "on each side",
          "[prismatic][halo_plan]") {
  const int L = 2, N_r = 12, K = 3;
  auto parts = make_radial_slabs(L, N_r, K);
  for (cochain_type t : {cochain_type::rect_face, cochain_type::v_edge}) {
    // Interior rank (r=1) has lower peer and upper peer.
    auto plan = build_radial_halo_plan(t, parts[1]);
    REQUIRE(plan.peers.size() == 2);

    // Lower peer: we only RECV (lower's top slab), we don't SEND.
    // Upper peer: we only SEND (our top slab = upper's lower ghost), no RECV.
    for (auto const& pe : plan.peers) {
      if (pe.peer_rank < 1) {
        REQUIRE(pe.send_global_idx.empty());
        REQUIRE_FALSE(pe.recv_global_idx.empty());
      } else {
        REQUIRE_FALSE(pe.send_global_idx.empty());
        REQUIRE(pe.recv_global_idx.empty());
      }
    }
  }
}

// =========================================================================
// End-to-end exchange: fill each rank's buffer with a rank-stamped pattern,
// run halo exchange, and verify ghost slots now match the neighbor's owned
// slots.
// =========================================================================

// Stamp: global index -> unique scalar value.  Use a linear function so we
// can tell which rank "wrote" a value (each rank writes only its owned
// indices; ghost slots start at 0 and should be overwritten by the peer).
static Scalar stamp_value(int global_idx, int /*rank*/) {
  return Scalar(1) + Scalar(global_idx) * Scalar(1e-4);
}

// Fill `buf` (of size total_global) with stamp_value at each global index
// this rank OWNS, and zero elsewhere (simulating ghost slots).
static void fill_owned_only(std::vector<Scalar>& buf,
                             cochain_type t,
                             const prismatic_partition& p) {
  int N = global_cochain_size(t, p);
  buf.assign(N, Scalar(0));
  int width = 0;
  switch (t) {
    case cochain_type::tri_face:  width = p.N_tri_global;    break;
    case cochain_type::rect_face: width = p.N_edge_s_global; break;
    case cochain_type::h_edge:    width = p.N_edge_s_global; break;
    case cochain_type::v_edge:    width = p.N_vert_s_global; break;
    case cochain_type::vertex:    width = p.N_vert_s_global; break;
  }
  // Shell-living cochains own shells [k_lo, k_hi).
  // Slab-living own slabs [k_lo, k_hi) (conveniently same range).
  int kl = p.shell_k_lo, kh = p.shell_k_hi;
  // For slab cochains at interior ranks we own slabs [k_lo, k_hi), but
  // the top owned slab is kh - 1 which needs upper ghost shell.  Here
  // we're populating the buffer, so just write everything owned.
  // For the boundary ranks the slab range might go up to N_r_global.
  bool slab_cochain = (t == cochain_type::rect_face ||
                        t == cochain_type::v_edge);
  if (slab_cochain) {
    // Slab k exists only if k ∈ [0, N_r_global).
    int max_slab = std::min(kh, p.N_r_global);
    for (int k = kl; k < max_slab; ++k) {
      for (int i = 0; i < width; ++i) {
        buf[k * width + i] = stamp_value(k * width + i, p.radial_rank);
      }
    }
  } else {
    // Shell cochain: shells in [k_lo, k_hi).  Global range [0, N_r_global].
    for (int k = kl; k < kh; ++k) {
      for (int i = 0; i < width; ++i) {
        buf[k * width + i] = stamp_value(k * width + i, p.radial_rank);
      }
    }
  }
}

TEST_CASE("in-process halo exchange fills ghost slots with neighbor data",
          "[prismatic][halo_plan][in_process]") {
  const int L = 2, N_r = 12, K = 4;
  auto parts = make_radial_slabs(L, N_r, K);

  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::vertex, cochain_type::rect_face,
                         cochain_type::v_edge}) {
    // Allocate per-rank global-sized buffers, populate with owned-only
    // stamps, register with the backend.
    std::vector<std::vector<Scalar>> bufs(K);
    in_process_halo_backend backend;
    for (int r = 0; r < K; ++r) {
      fill_owned_only(bufs[r], t, parts[r]);
      backend.register_rank(r, bufs[r].data());
    }

    // Build per-rank plans.
    std::vector<halo_plan> plans;
    plans.reserve(K);
    for (int r = 0; r < K; ++r) {
      plans.push_back(build_radial_halo_plan(t, parts[r]));
    }

    // One-shot exchange.
    backend.exchange_all(plans);

    // Verify: every global index in rank r's recv set now holds the
    // stamped value (i.e., was written by the peer).
    for (int r = 0; r < K; ++r) {
      for (auto const& pe : plans[r].peers) {
        for (int idx : pe.recv_global_idx) {
          Scalar expected = stamp_value(idx, pe.peer_rank);
          REQUIRE(bufs[r][idx] == Approx(expected));
        }
      }
    }
  }
}

TEST_CASE("in-process halo exchange leaves owned slots unchanged",
          "[prismatic][halo_plan][in_process]") {
  const int L = 2, N_r = 12, K = 4;
  auto parts = make_radial_slabs(L, N_r, K);
  cochain_type t = cochain_type::tri_face;

  std::vector<std::vector<Scalar>> bufs(K);
  std::vector<std::vector<Scalar>> bufs_before(K);
  in_process_halo_backend backend;
  for (int r = 0; r < K; ++r) {
    fill_owned_only(bufs[r], t, parts[r]);
    bufs_before[r] = bufs[r];  // snapshot
    backend.register_rank(r, bufs[r].data());
  }

  std::vector<halo_plan> plans;
  for (int r = 0; r < K; ++r) plans.push_back(build_radial_halo_plan(t, parts[r]));
  backend.exchange_all(plans);

  // Owned slots: untouched.  Ghost slots: overwritten.  Build the set of
  // recv indices to distinguish them.
  for (int r = 0; r < K; ++r) {
    std::set<int> recv_set;
    for (auto const& pe : plans[r].peers)
      for (int idx : pe.recv_global_idx) recv_set.insert(idx);

    int width = parts[r].N_tri_global;
    for (int k = parts[r].shell_k_lo; k < parts[r].shell_k_hi; ++k) {
      for (int i = 0; i < width; ++i) {
        int idx = k * width + i;
        REQUIRE(recv_set.find(idx) == recv_set.end());  // not a ghost
        REQUIRE(bufs[r][idx] == Approx(bufs_before[r][idx]));
      }
    }
  }
}
