// Phase 7C.1/7C.2 — prismatic_ptc_mesh_local unit tests.
//
// 1. Identity partition: local tables and tensor→layout maps are the
//    identity (bit-compatible with the global prismatic_mesh_ptrs
//    formulas).
// 2. Canonical A×K partitions at pic depth: every remapped table entry
//    equals its global counterpart under l2g; every map slot points at
//    the layout slot of the right global cochain index.
// 3. HOST bit-exactness: pushing the same particles through the global
//    mesh ptrs and the identity local mesh produces bit-identical
//    state and deposits (the 7C.2 landmark).
// 4. Cell ownership tiles across ranks and migrate_dest returns the
//    owning world rank for every non-owned halo cell.
#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_ptc_mesh_local.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include <cmath>
#include <memory>
#include <vector>

using namespace Aperture;

namespace {

constexpr int TL = 2;
constexpr int TN_r = 8;

struct fixture {
  prismatic_mesh mesh;
  icosphere_topology topo;

  fixture() {
    mesh.build(TL, TN_r, 1.0, 2.0);
    topo = icosphere_topology::build_from_mesh(mesh);
  }

  prismatic_mesh_partition bundle(const prismatic_partition& p,
                                  halo_depth d) {
    auto part = p;
    part.set_topology(&topo);
    return prismatic_mesh_partition::build(part, topo, d);
  }
};

}  // namespace

TEST_CASE("ptc_mesh_local identity: tables and maps equal the global mesh",
          "[prismatic][ptc_mesh_local]") {
  fixture fx;
  auto mp_b = fx.bundle(prismatic_partition::single_rank(TL, TN_r),
                        halo_depth::solver);
  prismatic_ptc_mesh_local lm;
  lm.build(fx.mesh, mp_b);
  auto l = lm.host_ptrs();
  auto g = fx.mesh.host_ptrs();

  REQUIRE(l.N_tri == g.N_tri);
  REQUIRE(l.N_r == g.N_r);
  REQUIRE(l.N_vert_s == g.N_vert_s);
  REQUIRE(l.N_edge_s == g.N_edge_s);
  REQUIRE(l.N_verts == g.N_verts);
  REQUIRE(l.N_edges == g.N_edges);
  REQUIRE(l.N_faces == g.N_faces);
  REQUIRE(l.n_tri_own == g.N_tri);
  REQUIRE(l.k0 == 0);
  REQUIRE(l.lay_own_lo == 0);
  REQUIRE(l.lay_own_hi == g.N_r);

  for (int t = 0; t < g.N_tri * 3; ++t) {
    REQUIRE(l.tri_verts[t] == g.tri_verts[t]);
    REQUIRE(l.tri_edges_s[t] == g.tri_edges_s[t]);
    REQUIRE(l.tri_edge_signs[t] == g.tri_edge_signs[t]);
    REQUIRE(l.tri_neighbor[t] == g.tri_neighbor[t]);
  }
  for (int v = 0; v < g.N_vert_s; ++v) {
    REQUIRE(l.sphere_vx[v] == g.sphere_vx[v]);
    REQUIRE(l.sphere_vy[v] == g.sphere_vy[v]);
    REQUIRE(l.sphere_vz[v] == g.sphere_vz[v]);
  }
  for (int k = 0; k <= g.N_r; ++k) {
    REQUIRE(l.radii[k] == g.radii[k]);
    for (int e = 0; e < g.N_edge_s; ++e) {
      REQUIRE(l.h_edge_idx(k, e) == g.h_edge_idx(k, e));
    }
    for (int t = 0; t < g.N_tri; ++t) {
      REQUIRE(l.tri_face_idx(k, t) == g.tri_face_idx(k, t));
    }
    for (int v = 0; v < g.N_vert_s; ++v) {
      REQUIRE(l.vertex_idx(k, v) == g.vertex_idx(k, v));
    }
  }
  for (int k = 0; k < g.N_r; ++k) {
    for (int e = 0; e < g.N_edge_s; ++e) {
      REQUIRE(l.rect_face_idx(k, e) == g.rect_face_idx(k, e));
    }
    for (int v = 0; v < g.N_vert_s; ++v) {
      REQUIRE(l.v_edge_idx(k, v) == g.v_edge_idx(k, v));
    }
  }
  for (int e = 0; e < g.N_edges; ++e) {
    REQUIRE(l.hodge1_inv[e] == g.hodge1_inv[e]);
  }
  for (int v = 0; v < g.N_verts; ++v) {
    REQUIRE(l.vert_dual_vol[v] == g.vert_dual_vol[v]);
  }
}

TEST_CASE("ptc_mesh_local canonical: remapped entries equal global under "
          "l2g on every rank",
          "[prismatic][ptc_mesh_local]") {
  fixture fx;
  const int A = 8, K = 2;
  auto g = fx.mesh.host_ptrs();

  for (int w = 0; w < A * K; ++w) {
    auto mp_b = fx.bundle(
        prismatic_partition::combined(TL, TN_r, A, K, w), halo_depth::pic);
    prismatic_ptc_mesh_local lm;
    lm.build(fx.mesh, mp_b);
    auto l = lm.host_ptrs();
    const auto& t_l2g = lm.tri_l2g_host();
    const auto& v_l2g = lm.vert_l2g();
    const auto& e_l2g = lm.edge_l2g();

    // Owned-then-ghost, each ascending global.
    for (int i = 1; i < lm.n_tri_own(); ++i)
      REQUIRE(t_l2g[i] > t_l2g[i - 1]);
    for (int i = lm.n_tri_own() + 1; i < lm.n_tri_local(); ++i)
      REQUIRE(t_l2g[i] > t_l2g[i - 1]);

    // Remapped tables round-trip through l2g.
    for (int lt = 0; lt < lm.n_tri_local(); ++lt) {
      const int gt = t_l2g[lt];
      for (int j = 0; j < 3; ++j) {
        REQUIRE(v_l2g[l.tri_verts[lt * 3 + j]] == g.tri_verts[gt * 3 + j]);
        REQUIRE(e_l2g[l.tri_edges_s[lt * 3 + j]] ==
                g.tri_edges_s[gt * 3 + j]);
        REQUIRE(l.tri_edge_signs[lt * 3 + j] == g.tri_edge_signs[gt * 3 + j]);
        const int ln = l.tri_neighbor[lt * 3 + j];
        const int gn = g.tri_neighbor[gt * 3 + j];
        if (ln >= 0) {
          REQUIRE(t_l2g[ln] == gn);
        } else {
          // Off-halo: the global neighbor must not be a local tri.
          REQUIRE(lm.tri_g2l()[gn] == -1);
        }
      }
    }
    for (int lv = 0; lv < l.N_vert_s; ++lv) {
      REQUIRE(l.sphere_vx[lv] == g.sphere_vx[v_l2g[lv]]);
      REQUIRE(l.sphere_vy[lv] == g.sphere_vy[v_l2g[lv]]);
      REQUIRE(l.sphere_vz[lv] == g.sphere_vz[v_l2g[lv]]);
    }

    // Maps point at the layout slot of the right global cochain index.
    const auto& L_h = mp_b.layout(cochain_type::h_edge);
    const auto& L_v = mp_b.layout(cochain_type::v_edge);
    const auto& L_tri = mp_b.layout(cochain_type::tri_face);
    const auto& L_rect = mp_b.layout(cochain_type::rect_face);
    const auto& L_vert = mp_b.layout(cochain_type::vertex);
    for (int k = 0; k <= l.N_r; ++k) {
      const int kg = l.k0 + k;
      for (int e = 0; e < l.N_edge_s; ++e) {
        REQUIRE(L_h.to_global(l.h_edge_idx(k, e)) ==
                kg * g.N_edge_s + e_l2g[e]);
      }
      for (int t = 0; t < l.N_tri; ++t) {
        REQUIRE(L_tri.to_global(l.tri_face_idx(k, t)) ==
                kg * g.N_tri + t_l2g[t]);
      }
      for (int v = 0; v < l.N_vert_s; ++v) {
        REQUIRE(L_vert.to_global(l.vertex_idx(k, v)) ==
                kg * g.N_vert_s + v_l2g[v]);
      }
    }
    for (int k = 0; k < l.N_r; ++k) {
      const int kg = l.k0 + k;
      for (int e = 0; e < l.N_edge_s; ++e) {
        REQUIRE(L_rect.to_global(l.rect_face_idx(k, e) - l.b_split) ==
                kg * g.N_edge_s + e_l2g[e]);
      }
      for (int v = 0; v < l.N_vert_s; ++v) {
        REQUIRE(L_v.to_global(l.v_edge_idx(k, v) - l.e_split) ==
                kg * g.N_vert_s + v_l2g[v]);
      }
    }

    // Radii window: local layer k is global layer k0 + k.
    for (int k = 0; k <= l.N_r; ++k) {
      REQUIRE(l.radii[k] == g.radii[l.k0 + k]);
    }
  }
}

TEST_CASE("ptc_mesh_local: host push through identity local mesh is "
          "bit-exact with the global path",
          "[prismatic][ptc_mesh_local]") {
  fixture fx;
  auto mp_b = fx.bundle(prismatic_partition::single_rank(TL, TN_r),
                        halo_depth::solver);
  prismatic_vertex_recovery recovery;
  recovery.build(fx.mesh);
  prismatic_ptc_mesh_local lm;
  lm.build(fx.mesh, mp_b, &recovery);
  auto lmp = lm.host_ptrs();
  auto gmp = fx.mesh.host_ptrs();

  // Deterministic fields.
  std::vector<Scalar> E(fx.mesh.m_N_edges), B(fx.mesh.m_N_faces);
  for (int e = 0; e < fx.mesh.m_N_edges; ++e)
    E[e] = Scalar(0.01) * std::sin(Scalar(0.013) * e);
  for (int f = 0; f < fx.mesh.m_N_faces; ++f)
    B[f] = Scalar(0.5) + Scalar(0.1) * std::cos(Scalar(0.007) * f);

  // Recovery Bv: global path vs local owned-slot path.
  auto rp = recovery.host_ptrs();
  for (int vi = 0; vi < fx.mesh.m_N_verts; ++vi) {
    rp.compute_vertex_B(gmp, B.data(), vi);
  }
  std::vector<Scalar> Bv_loc(size_t(3) * lmp.N_verts, Scalar(0));
  for (int k = lmp.shell_own_lo; k < lmp.shell_own_hi; ++k) {
    for (int s = 0; s < lmp.n_vert_s_own; ++s) {
      compute_vertex_B_local(lmp, B.data(), Bv_loc.data(), k, s);
    }
  }
  // The two fit implementations are separate template instantiations,
  // so FP contraction (FMA) can differ per term — allow a few ULP.
  for (size_t i = 0; i < Bv_loc.size(); ++i) {
    REQUIRE(double(Bv_loc[i]) ==
            Catch::Approx(double(rp.Bv[i])).margin(1e-5).epsilon(1e-5));
  }

  // Seed identical particles; push through both paths.
  const int n_ptc = 500;
  prismatic_particles_t ptc_g(n_ptc, MemType::host_only);
  prismatic_particles_t ptc_l(n_ptc, MemType::host_only);
  int n_seed = 0;
  for (int t = 0; t < fx.mesh.m_N_tri && n_seed < n_ptc; t += 3) {
    for (int k = 0; k < fx.mesh.m_N_r && n_seed < n_ptc; k += 2) {
      auto seed = [&](prismatic_particles_t& p) {
        auto h = p.get_host_ptrs();
        h.x1[n_seed] = Scalar(0.3);
        h.x2[n_seed] = Scalar(0.3);
        h.x3[n_seed] = Scalar(0.5);
        h.p1[n_seed] = Scalar(0.4) * std::sin(Scalar(0.3) * n_seed);
        h.p2[n_seed] = Scalar(0.4) * std::cos(Scalar(0.5) * n_seed);
        h.p3[n_seed] = Scalar(0.2);
        h.E[n_seed] = Scalar(1.2);
        h.weight[n_seed] = Scalar(1);
        h.cell[n_seed] = uint32_t(k * fx.mesh.m_N_tri + t);
        h.flag[n_seed] = (n_seed % 2) ? set_ptc_type_flag(0u, PtcType::positron)
                                      : 0u;
      };
      seed(ptc_g);
      seed(ptc_l);
      n_seed++;
    }
  }
  ptc_g.set_num(n_seed);
  ptc_l.set_num(n_seed);

  std::vector<Scalar> Jg(fx.mesh.m_N_edges, 0), Jl(fx.mesh.m_N_edges, 0);
  std::vector<Scalar> rg(fx.mesh.m_N_verts, 0), rl(fx.mesh.m_N_verts, 0);
  auto hg = ptc_g.get_host_ptrs();
  auto hl = ptc_l.get_host_ptrs();
  const Scalar dt = Scalar(0.01);
  for (int step = 0; step < 5; ++step) {
    for (int n = 0; n < n_seed; ++n) {
      if (hg.cell[n] != empty_cell) {
        Scalar q = (get_ptc_type(hg.flag[n]) == (int)PtcType::positron)
                       ? Scalar(1)
                       : Scalar(-1);
        update_single_particle(gmp, gmp.N_tri, hg, n, E.data(), B.data(),
                               Jg.data(), rg.data(), q, Scalar(1), dt,
                               false, false, rp.Bv);
      }
      if (hl.cell[n] != empty_cell) {
        Scalar q = (get_ptc_type(hl.flag[n]) == (int)PtcType::positron)
                       ? Scalar(1)
                       : Scalar(-1);
        // Same Bv input on both paths (identity layout): this isolates
        // the push/gather/deposit/walk conversion itself, which must be
        // bit-exact.
        update_single_particle(lmp, lmp.N_tri, hl, n, E.data(), B.data(),
                               Jl.data(), rl.data(), q, Scalar(1), dt,
                               false, false, rp.Bv);
      }
    }
  }

  for (int n = 0; n < n_seed; ++n) {
    REQUIRE(hg.cell[n] == hl.cell[n]);
    REQUIRE(hg.x1[n] == hl.x1[n]);
    REQUIRE(hg.x2[n] == hl.x2[n]);
    REQUIRE(hg.x3[n] == hl.x3[n]);
    REQUIRE(hg.p1[n] == hl.p1[n]);
    REQUIRE(hg.p2[n] == hl.p2[n]);
    REQUIRE(hg.p3[n] == hl.p3[n]);
    REQUIRE(hg.E[n] == hl.E[n]);
  }
  for (int e = 0; e < fx.mesh.m_N_edges; ++e) REQUIRE(Jg[e] == Jl[e]);
  for (int v = 0; v < fx.mesh.m_N_verts; ++v) REQUIRE(rg[v] == rl[v]);
}

TEST_CASE("ptc_mesh_local: cell ownership tiles and migrate_dest routes "
          "to the owner",
          "[prismatic][ptc_mesh_local]") {
  fixture fx;
  const int A = 8, K = 2;

  std::vector<std::unique_ptr<prismatic_ptc_mesh_local>> lms;
  std::vector<prismatic_mesh_partition> bundles;
  bundles.reserve(A * K);
  for (int w = 0; w < A * K; ++w) {
    bundles.push_back(fx.bundle(
        prismatic_partition::combined(TL, TN_r, A, K, w), halo_depth::pic));
    lms.push_back(std::make_unique<prismatic_ptc_mesh_local>());
    lms.back()->build(fx.mesh, bundles.back());
  }

  for (int gt = 0; gt < fx.mesh.m_N_tri; ++gt) {
    for (int gk = 0; gk < fx.mesh.m_N_r; ++gk) {
      int owner = -1;
      for (int w = 0; w < A * K; ++w) {
        const auto& lm = *lms[w];
        const int lt = lm.tri_g2l()[gt];
        const int lk = gk - lm.k0();
        auto l = lm.host_ptrs();
        const bool in_range =
            lt >= 0 && lk >= 0 && lk < l.N_r;
        const bool owns = in_range && l.owns_cell(lt, lk);
        if (owns) {
          REQUIRE(owner == -1);  // exactly one owner
          owner = w;
        }
      }
      REQUIRE(owner >= 0);

      // Every rank holding the cell as NON-owned routes it to `owner`.
      for (int w = 0; w < A * K; ++w) {
        const auto& lm = *lms[w];
        const int lt = lm.tri_g2l()[gt];
        const int lk = gk - lm.k0();
        auto l = lm.host_ptrs();
        if (lt < 0 || lk < 0 || lk >= l.N_r) continue;
        const uint32_t cell = uint32_t(lk * l.N_tri + lt);
        const int dest = l.migrate_dest(cell);
        if (w == owner) {
          REQUIRE(dest == -1);
        } else {
          REQUIRE(dest == owner);
        }
        // Wire round-trip: global encode → owner's local decode (the
        // wire is uint64 — global cells pass 2^32 near L9).
        const uint64_t wc = l.wire_cell(cell);
        const int glay = int(wc / uint64_t(fx.mesh.m_N_tri));
        const int gtri = int(wc % uint64_t(fx.mesh.m_N_tri));
        REQUIRE(glay == gk);
        REQUIRE(gtri == gt);
      }
    }
  }
}

TEST_CASE("ptc_mesh_local: wire_cell is 64-bit above the uint32 wall",
          "[prismatic][ptc_mesh_local]") {
  // Synthetic L10-scale geometry (no mesh build): the global cell
  // k * N_tri_global + tri passes 2^32 near L9 — the wire must carry
  // it exactly (checkpoint plan appendix item 1).
  prismatic_ptc_mesh_ptrs p{};
  const int l2g[2] = {20971519, 3};  // local tri 0 -> the last L10 tri
  p.N_tri = 2;
  p.k0 = 3000;
  p.N_tri_global = 20971520;  // 20 * 4^10
  p.tri_l2g = l2g;

  const uint32_t cell = uint32_t(1) * 2 + 0;  // local layer 1, local tri 0
  const uint64_t wc = p.wire_cell(cell);
  REQUIRE(wc == uint64_t(3001) * 20971520ull + 20971519ull);
  REQUIRE(wc > (uint64_t(1) << 32));
  // Decode round-trip at 64 bits.
  REQUIRE(int(wc / uint64_t(p.N_tri_global)) == 3001);
  REQUIRE(int(wc % uint64_t(p.N_tri_global)) == 20971519);
}
