// =========================================================================
// Phase 7B — pic depth-class ghost sets + reduce() operator tests.
//
// 1. The plan-built pic ghost sets (angular ∪ radial recv) equal a
//    literal brute-force transcription of the F4 spec: all elements
//    incident to the prisms of T_halo = (T_own ∪ 1-ring) × slabs
//    [k_lo−1, k_hi], minus owned.  Includes the valence-5 corners, the
//    slab-corner ghosts, and the depth-2 upper shell k_hi+1.
// 2. Lockstep exchange (angular round, then radial) fills EVERY pic
//    ghost slot with the owner's value — corner ghosts arrive by radial
//    forwarding of the freshly-exchanged angular ghost columns.
// 3. Staged reduce (radial round, then angular — the exact reverse)
//    accumulates every rank's synthetic ghost deposits into the owner
//    slot, matching a global reference sum; all ghost slots end zero.
//    Corner deposits take the two-hop relay through the radial peer.
// =========================================================================
#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

using namespace Aperture;

namespace {

constexpr int TL = 2;
constexpr int TN_r = 8;

const cochain_type kAllCochains[] = {
    cochain_type::tri_face, cochain_type::h_edge, cochain_type::rect_face,
    cochain_type::v_edge, cochain_type::vertex};

std::unique_ptr<prismatic_mesh> make_mesh(int L) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, TN_r, 1.0, 2.0);
  return mesh;
}

// All A·K rank bundles at pic depth, indexed by world rank = rad·A + ang.
struct rank_set {
  int A = 0, K = 0;
  std::vector<prismatic_partition> parts;
  std::vector<std::unique_ptr<prismatic_mesh_partition>> mps;
};

rank_set make_rank_set(int A, int K, const icosphere_topology& topo) {
  rank_set rs;
  rs.A = A;
  rs.K = K;
  for (int w = 0; w < A * K; ++w) {
    auto part = prismatic_partition::combined(TL, TN_r, A, K, w);
    part.set_topology(&topo);
    rs.parts.push_back(part);
    rs.mps.push_back(std::make_unique<prismatic_mesh_partition>(
        prismatic_mesh_partition::build(rs.parts.back(), topo,
                                        halo_depth::pic)));
  }
  return rs;
}

bool owns(cochain_type t, const prismatic_partition& p, int g) {
  switch (t) {
    case cochain_type::tri_face:  return p.owns_tri_face_cochain(g);
    case cochain_type::rect_face: return p.owns_rect_face_cochain(g);
    case cochain_type::h_edge:    return p.owns_h_edge_cochain(g);
    case cochain_type::v_edge:    return p.owns_v_edge_cochain(g);
    case cochain_type::vertex:    return p.owns_vertex_cochain(g);
  }
  return false;
}

int global_size(cochain_type t, const prismatic_partition& p) {
  return global_cochain_size(t, p);
}

// -------------------------------------------------------------------------
// Literal F4 transcription: availability of every global cochain index on
// rank R = incidence with the prisms T_halo × S_halo, where T_halo =
// owned tris ∪ 1-ring and S_halo = owned slabs padded by one on each
// side (clamped to [0, N_r)).
// -------------------------------------------------------------------------
std::vector<char> brute_force_available(cochain_type t,
                                        const prismatic_partition& p,
                                        const icosphere_topology& topo) {
  const int n_tri = topo.N_tri();

  // T_halo: tri τ qualifies iff any vertex of τ has an owned tri in its
  // fan.  Recover tri→verts by inverting the vertex fans.
  std::vector<std::vector<int>> tri_verts(n_tri);
  for (int v = 0; v < topo.N_vert_s(); ++v) {
    for (int j = 0; j < topo.vertex_tri_count(v); ++j) {
      tri_verts[topo.vertex_tris(v)[j]].push_back(v);
    }
  }
  std::vector<char> in_halo_tri(n_tri, 0);
  for (int tau = 0; tau < n_tri; ++tau) {
    REQUIRE(tri_verts[tau].size() == 3);
    for (int v : tri_verts[tau]) {
      for (int j = 0; j < topo.vertex_tri_count(v) && !in_halo_tri[tau];
           ++j) {
        if (p.owns_sub_tri(topo.vertex_tris(v)[j])) in_halo_tri[tau] = 1;
      }
    }
  }

  // S_halo (slab indices).
  auto slab_in_halo = [&](int s) {
    if (s < 0 || s >= p.N_r_global) return false;
    return s >= p.shell_k_lo - 1 && s <= std::min(p.shell_k_hi, p.N_r_global);
  };

  // Angular-side availability per sub-element.
  int width = 0;
  std::vector<char> col;
  switch (t) {
    case cochain_type::tri_face: {
      width = topo.N_tri();
      col.assign(width, 0);
      for (int x = 0; x < width; ++x) col[x] = in_halo_tri[x];
      break;
    }
    case cochain_type::h_edge:
    case cochain_type::rect_face: {
      width = topo.N_edge_s();
      col.assign(width, 0);
      for (int x = 0; x < width; ++x) {
        col[x] = in_halo_tri[topo.edge_tri_a(x)] ||
                 in_halo_tri[topo.edge_tri_b(x)];
      }
      break;
    }
    case cochain_type::v_edge:
    case cochain_type::vertex: {
      width = topo.N_vert_s();
      col.assign(width, 0);
      for (int x = 0; x < width; ++x) {
        for (int j = 0; j < topo.vertex_tri_count(x); ++j) {
          if (in_halo_tri[topo.vertex_tris(x)[j]]) {
            col[x] = 1;
            break;
          }
        }
      }
      break;
    }
  }

  const bool shell_kind = (t == cochain_type::tri_face ||
                           t == cochain_type::h_edge ||
                           t == cochain_type::vertex);
  const int N = global_size(t, p);
  std::vector<char> avail(N, 0);
  for (int g = 0; g < N; ++g) {
    const int k = g / width;
    const int x = g % width;
    if (!col[x]) continue;
    // A shell-k element is incident to prisms (·, k−1) and (·, k); a
    // slab-k element to prism (·, k).
    const bool radial_ok =
        shell_kind ? (slab_in_halo(k - 1) || slab_in_halo(k))
                   : slab_in_halo(k);
    avail[g] = radial_ok ? 1 : 0;
  }
  return avail;
}

// Union of a rank's plan recv sets (global-indexed), per axis.
std::vector<int> recv_set(const halo_plan& plan) {
  std::vector<int> out;
  for (auto const& pe : plan.peers) {
    out.insert(out.end(), pe.recv_global_idx.begin(),
               pe.recv_global_idx.end());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// -------------------------------------------------------------------------
// Lockstep drivers over global-size buffers.  In global-indexed plans a
// paired recv/send entry names the SAME global index on both sides, so
// exchange copies bufs[peer][g] → bufs[self][g] and reduce accumulates
// bufs[peer][g] += bufs[self][g] then zeroes.  Order: exchange angular
// → radial (forwarding), reduce radial → angular (relay back).
// -------------------------------------------------------------------------
void lockstep_exchange(cochain_type t, const rank_set& rs,
                       std::vector<std::vector<Scalar>>& bufs) {
  auto axis = [&](bool angular) {
    for (int w = 0; w < rs.A * rs.K; ++w) {
      const auto& plan = angular ? rs.mps[w]->angular_plan_global(t)
                                 : rs.mps[w]->radial_plan_global(t);
      const int ang = w % rs.A, rad = w / rs.A;
      for (auto const& pe : plan.peers) {
        const int peer_w =
            angular ? rad * rs.A + pe.peer_rank : pe.peer_rank * rs.A + ang;
        for (int g : pe.recv_global_idx) bufs[w][g] = bufs[peer_w][g];
      }
    }
  };
  axis(true);   // angular round first
  axis(false);  // then radial (forwards angular ghosts)
}

void lockstep_reduce(cochain_type t, const rank_set& rs,
                     std::vector<std::vector<Scalar>>& bufs) {
  auto axis = [&](bool angular) {
    for (int w = 0; w < rs.A * rs.K; ++w) {
      const auto& plan = angular ? rs.mps[w]->angular_plan_global(t)
                                 : rs.mps[w]->radial_plan_global(t);
      const int ang = w % rs.A, rad = w / rs.A;
      for (auto const& pe : plan.peers) {
        const int peer_w =
            angular ? rad * rs.A + pe.peer_rank : pe.peer_rank * rs.A + ang;
        for (int g : pe.recv_global_idx) {
          bufs[peer_w][g] += bufs[w][g];
          bufs[w][g] = Scalar(0);
        }
      }
    }
  };
  axis(false);  // radial round first (fold corners into the relay)
  axis(true);   // then angular (forward to the owner)
}

Scalar stamp(int g) { return Scalar(1) + Scalar(g) * Scalar(1e-4); }
Scalar contrib(int w, int g) {
  return Scalar(0.01) * Scalar(w + 1) + Scalar(g) * Scalar(1e-6);
}

const std::pair<int, int> kShapes[] = {
    {4, 1}, {4, 2}, {20, 2}, {80, 1}, {8, 3}};

}  // namespace

TEST_CASE("pic ghost sets equal the brute-force F4 spec (T_halo prisms, "
          "corners, depth-2 upper shell) and the two axes are disjoint",
          "[prismatic][pic_reduce]") {
  auto mesh = make_mesh(TL);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (auto [A, K] : kShapes) {
    auto rs = make_rank_set(A, K, topo);
    for (cochain_type t : kAllCochains) {
      for (int w = 0; w < A * K; ++w) {
        auto const& p = rs.parts[w];
        auto expect_avail = brute_force_available(t, p, topo);
        std::vector<int> expected;
        for (int g = 0; g < global_size(t, p); ++g) {
          if (expect_avail[g] && !owns(t, p, g)) expected.push_back(g);
        }

        auto ang = recv_set(rs.mps[w]->angular_plan_global(t));
        auto rad = recv_set(rs.mps[w]->radial_plan_global(t));
        // Each ghost is delivered (and zeroed on reduce) by exactly one
        // axis.
        std::vector<int> inter;
        std::set_intersection(ang.begin(), ang.end(), rad.begin(), rad.end(),
                              std::back_inserter(inter));
        REQUIRE(inter.empty());

        std::vector<int> actual;
        std::merge(ang.begin(), ang.end(), rad.begin(), rad.end(),
                   std::back_inserter(actual));
        REQUIRE(actual == expected);
      }
    }
  }
}

TEST_CASE("pic exchange (angular then radial) fills every ghost slot with "
          "the owner's value, including forwarded corners",
          "[prismatic][pic_reduce]") {
  auto mesh = make_mesh(TL);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (auto [A, K] : kShapes) {
    auto rs = make_rank_set(A, K, topo);
    for (cochain_type t : kAllCochains) {
      const int N = global_size(t, rs.parts[0]);
      std::vector<std::vector<Scalar>> bufs(A * K);
      for (int w = 0; w < A * K; ++w) {
        bufs[w].assign(N, Scalar(0));
        for (int g = 0; g < N; ++g) {
          if (owns(t, rs.parts[w], g)) bufs[w][g] = stamp(g);
        }
      }
      lockstep_exchange(t, rs, bufs);

      for (int w = 0; w < A * K; ++w) {
        auto avail = brute_force_available(t, rs.parts[w], topo);
        for (int g = 0; g < N; ++g) {
          if (avail[g]) {
            REQUIRE(bufs[w][g] == stamp(g));
          }
        }
      }
    }
  }
}

TEST_CASE("staged pic reduce (radial then angular) matches the global "
          "contribution sum on owners and zeroes every ghost",
          "[prismatic][pic_reduce]") {
  auto mesh = make_mesh(TL);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  for (auto [A, K] : kShapes) {
    auto rs = make_rank_set(A, K, topo);
    for (cochain_type t : kAllCochains) {
      const int N = global_size(t, rs.parts[0]);

      // Precompute each rank's ghost set (validated against F4 above).
      std::vector<std::vector<char>> is_ghost(A * K);
      for (int w = 0; w < A * K; ++w) {
        is_ghost[w].assign(N, 0);
        auto avail = brute_force_available(t, rs.parts[w], topo);
        for (int g = 0; g < N; ++g) {
          if (avail[g] && !owns(t, rs.parts[w], g)) is_ghost[w][g] = 1;
        }
      }

      // Owned slots hold stamp(g); ghost slots hold this rank's
      // synthetic deposit contribution.
      std::vector<std::vector<Scalar>> bufs(A * K);
      for (int w = 0; w < A * K; ++w) {
        bufs[w].assign(N, Scalar(0));
        for (int g = 0; g < N; ++g) {
          if (owns(t, rs.parts[w], g)) {
            bufs[w][g] = stamp(g);
          } else if (is_ghost[w][g]) {
            bufs[w][g] = contrib(w, g);
          }
        }
      }

      lockstep_reduce(t, rs, bufs);

      // Reference: owner total = stamp + sum of all ghosting ranks'
      // contributions (order-independent up to float re-association).
      for (int g = 0; g < N; ++g) {
        double expect = 0.0;
        int owner = -1;
        for (int w = 0; w < A * K; ++w) {
          if (owns(t, rs.parts[w], g)) owner = w;
          if (is_ghost[w][g]) expect += double(contrib(w, g));
        }
        REQUIRE(owner >= 0);
        expect += double(stamp(g));
        REQUIRE(double(bufs[owner][g]) ==
                Catch::Approx(expect).epsilon(1e-4));
        // Every non-owner slot (ghost or untouched) is zero.
        for (int w = 0; w < A * K; ++w) {
          if (w != owner) REQUIRE(bufs[w][g] == Scalar(0));
        }
      }
    }
  }
}
