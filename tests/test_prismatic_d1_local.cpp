#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_d1_local.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <map>
#include <memory>

using namespace Aperture;
using Catch::Matchers::WithinAbs;

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L, int N_r) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, N_r, 1.0, 2.0);
  return mesh;
}

// Reconstruct the global d1 row for face g_face as a {edge -> sign} map.
std::map<int, Scalar> global_d1_row(const prismatic_mesh& mesh, int g_face) {
  const int* rp = mesh.d1_row_ptr.host_ptr();
  const int* ci = mesh.d1_col_idx.host_ptr();
  const Scalar* v = mesh.d1_val.host_ptr();
  std::map<int, Scalar> out;
  for (int j = rp[g_face]; j < rp[g_face + 1]; ++j) {
    out[ci[j]] = v[j];
  }
  return out;
}

// Reconstruct the global d1^T row for edge g_edge as a {face -> sign} map.
std::map<int, Scalar> global_d1t_row(const prismatic_mesh& mesh, int g_edge) {
  const int* rp = mesh.d1t_row_ptr.host_ptr();
  const int* ci = mesh.d1t_col_idx.host_ptr();
  const Scalar* v = mesh.d1t_val.host_ptr();
  std::map<int, Scalar> out;
  for (int j = rp[g_edge]; j < rp[g_edge + 1]; ++j) {
    out[ci[j]] = v[j];
  }
  return out;
}

}  // namespace

// =========================================================================
// d1 row degrees match the topology: 3 for tri faces, 4 for rect (2 h + 2 v).
// =========================================================================
TEST_CASE("d1_local: tri face d1 row has degree 3 in d1_tri_h",
          "[prismatic][d1_local]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto d1 = prismatic_d1_local::build(*mesh, mp);

  for (int l = 0; l < d1.d1_tri_h.n_rows(); ++l) {
    int row_len = d1.d1_tri_h.row_ptr[l + 1] - d1.d1_tri_h.row_ptr[l];
    REQUIRE(row_len == 3);
  }
}

TEST_CASE("d1_local: rect face has 2 h-edges + 2 v-edges",
          "[prismatic][d1_local]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto d1 = prismatic_d1_local::build(*mesh, mp);

  for (int l = 0; l < d1.d1_rect_h.n_rows(); ++l) {
    int h_len = d1.d1_rect_h.row_ptr[l + 1] - d1.d1_rect_h.row_ptr[l];
    int v_len = d1.d1_rect_v.row_ptr[l + 1] - d1.d1_rect_v.row_ptr[l];
    REQUIRE(h_len == 2);
    REQUIRE(v_len == 2);
  }
}

// =========================================================================
// Single-rank: local row reconstruction equals the global d1 row.
// =========================================================================
TEST_CASE("d1_local single-rank: reconstructed rows equal global d1",
          "[prismatic][d1_local]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto d1 = prismatic_d1_local::build(*mesh, mp);

  const int N_h_edges = (mesh->m_N_r + 1) * mesh->m_N_edge_s;
  const int N_tri_faces = (mesh->m_N_r + 1) * mesh->m_N_tri;

  auto const& L_tri = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he = mp.layout(cochain_type::h_edge);
  auto const& L_ve = mp.layout(cochain_type::v_edge);

  // Tri faces.
  for (int l = 0; l < L_tri.owned_size(); ++l) {
    int g_face = L_tri.to_global(l);
    auto global_row = global_d1_row(*mesh, g_face);

    std::map<int, Scalar> reconstructed;
    for (int j = d1.d1_tri_h.row_ptr[l]; j < d1.d1_tri_h.row_ptr[l + 1]; ++j) {
      int local_he = d1.d1_tri_h.col_idx[j];
      int g_he = L_he.to_global(local_he);  // global h_edge idx
      reconstructed[g_he] = d1.d1_tri_h.val[j];
    }
    REQUIRE(reconstructed == global_row);
  }

  // Rect faces.
  for (int l = 0; l < L_rect.owned_size(); ++l) {
    int g_rect = L_rect.to_global(l);
    int g_face = N_tri_faces + g_rect;
    auto global_row = global_d1_row(*mesh, g_face);

    std::map<int, Scalar> reconstructed;
    for (int j = d1.d1_rect_h.row_ptr[l]; j < d1.d1_rect_h.row_ptr[l + 1]; ++j) {
      int local_he = d1.d1_rect_h.col_idx[j];
      int g_he = L_he.to_global(local_he);
      reconstructed[g_he] = d1.d1_rect_h.val[j];
    }
    for (int j = d1.d1_rect_v.row_ptr[l]; j < d1.d1_rect_v.row_ptr[l + 1]; ++j) {
      int local_ve = d1.d1_rect_v.col_idx[j];
      int g_ve = L_ve.to_global(local_ve);
      int g_edge_combined = N_h_edges + g_ve;
      reconstructed[g_edge_combined] = d1.d1_rect_v.val[j];
    }
    REQUIRE(reconstructed == global_row);
  }
  (void)L_ve;
}

// =========================================================================
// d1^T transposes d1: every (face, edge) pair appears with the same sign.
// =========================================================================
TEST_CASE("d1_local: d1 and d1^T are consistent transposes",
          "[prismatic][d1_local]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto d1 = prismatic_d1_local::build(*mesh, mp);

  auto const& L_tri = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he = mp.layout(cochain_type::h_edge);
  auto const& L_ve = mp.layout(cochain_type::v_edge);
  (void)L_tri;

  // For every owned h_edge, every face in d1t_h_tri / d1t_h_rect must
  // have an entry in the corresponding d1 block referring back to this
  // edge with the same sign.
  for (int l_he = 0; l_he < L_he.owned_size(); ++l_he) {
    // Walk d1t_h_tri row.
    for (int j = d1.d1t_h_tri.row_ptr[l_he];
         j < d1.d1t_h_tri.row_ptr[l_he + 1]; ++j) {
      int local_tri = d1.d1t_h_tri.col_idx[j];
      Scalar sign = d1.d1t_h_tri.val[j];
      // The tri face's d1 row should contain this h_edge with the same sign.
      // (This works only if local_tri is in the OWNED tri range — i.e. < L_tri.owned_size())
      if (local_tri >= L_tri.owned_size()) continue;  // ghost — skip
      bool found = false;
      for (int k = d1.d1_tri_h.row_ptr[local_tri];
           k < d1.d1_tri_h.row_ptr[local_tri + 1]; ++k) {
        if (d1.d1_tri_h.col_idx[k] == l_he) {
          REQUIRE(d1.d1_tri_h.val[k] == sign);
          found = true;
          break;
        }
      }
      REQUIRE(found);
    }
    // Walk d1t_h_rect row.
    for (int j = d1.d1t_h_rect.row_ptr[l_he];
         j < d1.d1t_h_rect.row_ptr[l_he + 1]; ++j) {
      int local_rect = d1.d1t_h_rect.col_idx[j];
      Scalar sign = d1.d1t_h_rect.val[j];
      if (local_rect >= L_rect.owned_size()) continue;
      bool found = false;
      for (int k = d1.d1_rect_h.row_ptr[local_rect];
           k < d1.d1_rect_h.row_ptr[local_rect + 1]; ++k) {
        if (d1.d1_rect_h.col_idx[k] == l_he) {
          REQUIRE(d1.d1_rect_h.val[k] == sign);
          found = true;
          break;
        }
      }
      REQUIRE(found);
    }
  }

  // For every owned v_edge, faces in d1t_v_rect must reference back via
  // d1_rect_v with the same sign.
  for (int l_ve = 0; l_ve < L_ve.owned_size(); ++l_ve) {
    for (int j = d1.d1t_v_rect.row_ptr[l_ve];
         j < d1.d1t_v_rect.row_ptr[l_ve + 1]; ++j) {
      int local_rect = d1.d1t_v_rect.col_idx[j];
      Scalar sign = d1.d1t_v_rect.val[j];
      if (local_rect >= L_rect.owned_size()) continue;
      bool found = false;
      for (int k = d1.d1_rect_v.row_ptr[local_rect];
           k < d1.d1_rect_v.row_ptr[local_rect + 1]; ++k) {
        if (d1.d1_rect_v.col_idx[k] == l_ve) {
          REQUIRE(d1.d1_rect_v.val[k] == sign);
          found = true;
          break;
        }
      }
      REQUIRE(found);
    }
  }
}

// =========================================================================
// Combined partition: every owned face / edge's reconstructed row matches
// the global mesh's d1 / d1^T row.
// =========================================================================
TEST_CASE("d1_local combined partition: rows match global d1",
          "[prismatic][d1_local]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  const int N_tri_faces = (mesh->m_N_r + 1) * mesh->m_N_tri;
  const int N_h_edges = (mesh->m_N_r + 1) * mesh->m_N_edge_s;

  // Spot-check a few interior + boundary partitions.
  for (auto rk : std::vector<std::pair<int, int>>{
           {0, 0}, {1, 5}, {2, 11}, {3, 19}}) {
    auto part = prismatic_partition::combined(L, N_r, K, rk.first, rk.second);
    part.set_topology(&topo);
    auto mp = prismatic_mesh_partition::build(part, topo);
    auto d1 = prismatic_d1_local::build(*mesh, mp);

    auto const& L_tri = mp.layout(cochain_type::tri_face);
    auto const& L_rect = mp.layout(cochain_type::rect_face);
    auto const& L_he = mp.layout(cochain_type::h_edge);
    auto const& L_ve = mp.layout(cochain_type::v_edge);

    // Tri-face rows.
    for (int l = 0; l < L_tri.owned_size(); ++l) {
      int g_face = L_tri.to_global(l);
      auto global_row = global_d1_row(*mesh, g_face);
      std::map<int, Scalar> reconstructed;
      for (int j = d1.d1_tri_h.row_ptr[l]; j < d1.d1_tri_h.row_ptr[l + 1]; ++j) {
        int g_he = L_he.to_global(d1.d1_tri_h.col_idx[j]);
        reconstructed[g_he] = d1.d1_tri_h.val[j];
      }
      REQUIRE(reconstructed == global_row);
    }

    // Rect-face rows: need to merge h and v blocks before comparing.
    for (int l = 0; l < L_rect.owned_size(); ++l) {
      int g_rect = L_rect.to_global(l);
      int g_face = N_tri_faces + g_rect;
      auto global_row = global_d1_row(*mesh, g_face);
      std::map<int, Scalar> reconstructed;
      for (int j = d1.d1_rect_h.row_ptr[l]; j < d1.d1_rect_h.row_ptr[l + 1]; ++j) {
        int g_he = L_he.to_global(d1.d1_rect_h.col_idx[j]);
        reconstructed[g_he] = d1.d1_rect_h.val[j];
      }
      for (int j = d1.d1_rect_v.row_ptr[l]; j < d1.d1_rect_v.row_ptr[l + 1]; ++j) {
        int g_ve = L_ve.to_global(d1.d1_rect_v.col_idx[j]);
        reconstructed[N_h_edges + g_ve] = d1.d1_rect_v.val[j];
      }
      REQUIRE(reconstructed == global_row);
    }

    // d1^T rows: h-edge.
    for (int l = 0; l < L_he.owned_size(); ++l) {
      int g_he = L_he.to_global(l);
      auto global_row = global_d1t_row(*mesh, g_he);
      std::map<int, Scalar> reconstructed;
      for (int j = d1.d1t_h_tri.row_ptr[l]; j < d1.d1t_h_tri.row_ptr[l + 1]; ++j) {
        int g_tri = L_tri.to_global(d1.d1t_h_tri.col_idx[j]);
        reconstructed[g_tri] = d1.d1t_h_tri.val[j];
      }
      for (int j = d1.d1t_h_rect.row_ptr[l]; j < d1.d1t_h_rect.row_ptr[l + 1]; ++j) {
        int g_rect = L_rect.to_global(d1.d1t_h_rect.col_idx[j]);
        reconstructed[N_tri_faces + g_rect] = d1.d1t_h_rect.val[j];
      }
      REQUIRE(reconstructed == global_row);
    }

    // d1^T rows: v-edge.
    for (int l = 0; l < L_ve.owned_size(); ++l) {
      int g_ve = L_ve.to_global(l);
      int g_edge = N_h_edges + g_ve;
      auto global_row = global_d1t_row(*mesh, g_edge);
      std::map<int, Scalar> reconstructed;
      for (int j = d1.d1t_v_rect.row_ptr[l]; j < d1.d1t_v_rect.row_ptr[l + 1]; ++j) {
        int g_rect = L_rect.to_global(d1.d1t_v_rect.col_idx[j]);
        reconstructed[N_tri_faces + g_rect] = d1.d1t_v_rect.val[j];
      }
      REQUIRE(reconstructed == global_row);
    }
  }
}


// =========================================================================
// B0.2 operator-level test: applying the local d1 / d1^T blocks to a
// HALOED local field must reproduce the global SpMV on every owned row.
// This exercises exactly what the 4.1b solver conversion relies on:
// every column a local row touches is resolvable in the local index
// space (owned or halo) and carries the right value and sign.
// =========================================================================
static void check_operator_equivalence(const prismatic_mesh& mesh,
                                       const prismatic_mesh_partition& mp,
                                       const prismatic_d1_local& d1) {
  const int N_tri_faces = (mesh.m_N_r + 1) * mesh.m_N_tri;
  const int N_h_edges = (mesh.m_N_r + 1) * mesh.m_N_edge_s;
  const int N_edges = mesh.m_N_edges;
  const int N_faces = mesh.m_N_faces;

  // Deterministic synthetic global cochains.
  std::vector<double> E_g(N_edges), B_g(N_faces);
  for (int e = 0; e < N_edges; e++) E_g[e] = std::sin(0.013 * e) + 0.37;
  for (int f = 0; f < N_faces; f++) B_g[f] = std::cos(0.007 * f) - 0.21;

  auto const& L_tri  = mp.layout(cochain_type::tri_face);
  auto const& L_rect = mp.layout(cochain_type::rect_face);
  auto const& L_he   = mp.layout(cochain_type::h_edge);
  auto const& L_ve   = mp.layout(cochain_type::v_edge);

  // Haloed local fields, filled straight from the global arrays (halo
  // exchange correctness is tested elsewhere; this isolates the
  // operator).
  std::vector<double> E_h(L_he.local_size()), E_v(L_ve.local_size());
  std::vector<double> B_tri(L_tri.local_size()), B_rect(L_rect.local_size());
  for (int l = 0; l < L_he.local_size(); l++)
    E_h[l] = E_g[L_he.to_global(l)];
  for (int l = 0; l < L_ve.local_size(); l++)
    E_v[l] = E_g[N_h_edges + L_ve.to_global(l)];
  for (int l = 0; l < L_tri.local_size(); l++)
    B_tri[l] = B_g[L_tri.to_global(l)];
  for (int l = 0; l < L_rect.local_size(); l++)
    B_rect[l] = B_g[N_tri_faces + L_rect.to_global(l)];

  const int* g_row = mesh.d1_row_ptr.host_ptr();
  const int* g_col = mesh.d1_col_idx.host_ptr();
  const Scalar* g_val = mesh.d1_val.host_ptr();
  const int* gt_row = mesh.d1t_row_ptr.host_ptr();
  const int* gt_col = mesh.d1t_col_idx.host_ptr();
  const Scalar* gt_val = mesh.d1t_val.host_ptr();

  // ---- d1: owned tri faces ----
  for (int l = 0; l < L_tri.owned_size(); l++) {
    double loc = 0;
    for (int j = d1.d1_tri_h.row_ptr[l]; j < d1.d1_tri_h.row_ptr[l + 1]; j++)
      loc += d1.d1_tri_h.val[j] * E_h[d1.d1_tri_h.col_idx[j]];
    const int g_face = L_tri.to_global(l);
    double glob = 0;
    for (int j = g_row[g_face]; j < g_row[g_face + 1]; j++)
      glob += g_val[j] * E_g[g_col[j]];
    REQUIRE_THAT(loc, WithinAbs(glob, 1e-10));
  }
  // ---- d1: owned rect faces (h + v blocks) ----
  for (int l = 0; l < L_rect.owned_size(); l++) {
    double loc = 0;
    for (int j = d1.d1_rect_h.row_ptr[l]; j < d1.d1_rect_h.row_ptr[l + 1]; j++)
      loc += d1.d1_rect_h.val[j] * E_h[d1.d1_rect_h.col_idx[j]];
    for (int j = d1.d1_rect_v.row_ptr[l]; j < d1.d1_rect_v.row_ptr[l + 1]; j++)
      loc += d1.d1_rect_v.val[j] * E_v[d1.d1_rect_v.col_idx[j]];
    const int g_face = N_tri_faces + L_rect.to_global(l);
    double glob = 0;
    for (int j = g_row[g_face]; j < g_row[g_face + 1]; j++)
      glob += g_val[j] * E_g[g_col[j]];
    REQUIRE_THAT(loc, WithinAbs(glob, 1e-10));
  }
  // ---- d1^T: owned h edges ----
  for (int l = 0; l < L_he.owned_size(); l++) {
    double loc = 0;
    for (int j = d1.d1t_h_tri.row_ptr[l]; j < d1.d1t_h_tri.row_ptr[l + 1]; j++)
      loc += d1.d1t_h_tri.val[j] * B_tri[d1.d1t_h_tri.col_idx[j]];
    for (int j = d1.d1t_h_rect.row_ptr[l]; j < d1.d1t_h_rect.row_ptr[l + 1]; j++)
      loc += d1.d1t_h_rect.val[j] * B_rect[d1.d1t_h_rect.col_idx[j]];
    const int g_edge = L_he.to_global(l);
    double glob = 0;
    for (int j = gt_row[g_edge]; j < gt_row[g_edge + 1]; j++)
      glob += gt_val[j] * B_g[gt_col[j]];
    REQUIRE_THAT(loc, WithinAbs(glob, 1e-10));
  }
  // ---- d1^T: owned v edges ----
  for (int l = 0; l < L_ve.owned_size(); l++) {
    double loc = 0;
    for (int j = d1.d1t_v_rect.row_ptr[l]; j < d1.d1t_v_rect.row_ptr[l + 1]; j++)
      loc += d1.d1t_v_rect.val[j] * B_rect[d1.d1t_v_rect.col_idx[j]];
    const int g_edge = N_h_edges + L_ve.to_global(l);
    double glob = 0;
    for (int j = gt_row[g_edge]; j < gt_row[g_edge + 1]; j++)
      glob += gt_val[j] * B_g[gt_col[j]];
    REQUIRE_THAT(loc, WithinAbs(glob, 1e-10));
  }
}

TEST_CASE("d1_local operator: haloed local SpMV equals global on all "
          "partition types", "[prismatic][d1_local][operator]") {
  const int L = 2;
  const int N_r = 8;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  SECTION("20-rank angular") {
    for (int f = 0; f < 20; f++) {
      auto part = prismatic_partition::ico_face_angular(L, N_r, f);
      part.set_topology(&topo);
      auto mp = prismatic_mesh_partition::build(part, topo);
      auto d1 = prismatic_d1_local::build(*mesh, mp);
      check_operator_equivalence(*mesh, mp, d1);
    }
  }
  SECTION("radial slabs, K = 3") {
    for (int r = 0; r < 3; r++) {
      auto part = prismatic_partition::radial_slab(L, N_r, 3, r);
      part.set_topology(&topo);
      auto mp = prismatic_mesh_partition::build(part, topo);
      auto d1 = prismatic_d1_local::build(*mesh, mp);
      check_operator_equivalence(*mesh, mp, d1);
    }
  }
  SECTION("combined 20 x 4") {
    for (auto rk : std::vector<std::pair<int, int>>{
             {0, 0}, {0, 7}, {1, 3}, {2, 11}, {3, 19}, {3, 0}}) {
      auto part = prismatic_partition::combined(L, N_r, 4, rk.first,
                                                rk.second);
      part.set_topology(&topo);
      auto mp = prismatic_mesh_partition::build(part, topo);
      auto d1 = prismatic_d1_local::build(*mesh, mp);
      check_operator_equivalence(*mesh, mp, d1);
    }
  }
}
