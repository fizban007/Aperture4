#include "catch2/catch_all.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_local.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <memory>
#include <vector>

using namespace Aperture;

namespace {

std::unique_ptr<prismatic_mesh> make_mesh(int L, int N_r) {
  auto mesh = std::make_unique<prismatic_mesh>();
  mesh->build(L, N_r, 1.0, 2.0);
  return mesh;
}

// Assert a local buffer matches the corresponding slice of a global
// buffer at every local index, using layout.to_global(l) + base offset.
template <typename LocalVec, typename GlobalPtr>
void check_local_matches_global(const LocalVec& local, GlobalPtr global,
                                 int global_offset,
                                 const distributed_cochain_layout& layout) {
  REQUIRE(int(local.size()) == layout.local_size());
  for (int l = 0; l < layout.local_size(); ++l) {
    int g = layout.to_global(l) + global_offset;
    REQUIRE(local[l] == global[g]);
  }
}

}  // namespace

// =========================================================================
// Single-rank: local buffers equal the global mesh buffers entry by entry.
// =========================================================================
TEST_CASE("mesh_local single-rank: local buffers match global mesh",
          "[prismatic][mesh_local]") {
  const int L = 2;
  const int N_r = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  auto local = prismatic_mesh_local::build(*mesh, mp);

  const int N_tri_faces = (mesh->m_N_r + 1) * mesh->m_N_tri;
  const int N_h_edges   = (mesh->m_N_r + 1) * mesh->m_N_edge_s;

  // Tri-face buffers.
  check_local_matches_global(local.tri_face_area,  mesh->face_area.host_ptr(),
                              0, mp.layout(cochain_type::tri_face));
  check_local_matches_global(local.tri_face_hodge2, mesh->hodge2.host_ptr(),
                              0, mp.layout(cochain_type::tri_face));
  check_local_matches_global(local.tri_face_boundary,
                              mesh->face_boundary.host_ptr(), 0,
                              mp.layout(cochain_type::tri_face));
  check_local_matches_global(local.tri_face_radial_layer,
                              mesh->face_radial_layer.host_ptr(), 0,
                              mp.layout(cochain_type::tri_face));

  // Rect-face buffers (offset by N_tri_faces).
  check_local_matches_global(local.rect_face_area,  mesh->face_area.host_ptr(),
                              N_tri_faces, mp.layout(cochain_type::rect_face));
  check_local_matches_global(local.rect_face_hodge2, mesh->hodge2.host_ptr(),
                              N_tri_faces, mp.layout(cochain_type::rect_face));

  // H-edge buffers.
  check_local_matches_global(local.h_edge_length, mesh->edge_length.host_ptr(),
                              0, mp.layout(cochain_type::h_edge));
  check_local_matches_global(local.h_edge_hodge1_inv,
                              mesh->hodge1_inv.host_ptr(), 0,
                              mp.layout(cochain_type::h_edge));
  check_local_matches_global(local.h_edge_v0, mesh->edge_v0.host_ptr(), 0,
                              mp.layout(cochain_type::h_edge));
  check_local_matches_global(local.h_edge_v1, mesh->edge_v1.host_ptr(), 0,
                              mp.layout(cochain_type::h_edge));

  // V-edge buffers (offset by N_h_edges).
  check_local_matches_global(local.v_edge_length, mesh->edge_length.host_ptr(),
                              N_h_edges, mp.layout(cochain_type::v_edge));
  check_local_matches_global(local.v_edge_hodge1_inv,
                              mesh->hodge1_inv.host_ptr(), N_h_edges,
                              mp.layout(cochain_type::v_edge));
  check_local_matches_global(local.v_edge_v0, mesh->edge_v0.host_ptr(),
                              N_h_edges, mp.layout(cochain_type::v_edge));
  check_local_matches_global(local.v_edge_v1, mesh->edge_v1.host_ptr(),
                              N_h_edges, mp.layout(cochain_type::v_edge));

  // Vertex buffers.
  check_local_matches_global(local.vert_r,     mesh->vert_r.host_ptr(), 0,
                              mp.layout(cochain_type::vertex));
  check_local_matches_global(local.vert_theta, mesh->vert_theta.host_ptr(), 0,
                              mp.layout(cochain_type::vertex));
  check_local_matches_global(local.vert_phi,   mesh->vert_phi.host_ptr(), 0,
                              mp.layout(cochain_type::vertex));
}

// =========================================================================
// Combined partition: every rank's local buffers equal the corresponding
// slice of the global mesh, and sizes are strictly smaller than global.
// =========================================================================
TEST_CASE("mesh_local combined partition: sizes shrink + values match global",
          "[prismatic][mesh_local]") {
  const int L = 2;
  const int N_r = 8;
  const int K = 4;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  const int N_tri_faces_global = (mesh->m_N_r + 1) * mesh->m_N_tri;
  const int N_h_edges_global   = (mesh->m_N_r + 1) * mesh->m_N_edge_s;

  for (int r = 0; r < K; ++r) {
    for (int f = 0; f < 20; ++f) {
      auto part = prismatic_partition::combined(L, N_r, K, r, f);
      part.set_topology(&topo);
      auto mp = prismatic_mesh_partition::build(part, topo);
      auto local = prismatic_mesh_local::build(*mesh, mp);

      // Sizes shrink — each per-rank buffer is smaller than the global.
      REQUIRE(int(local.tri_face_area.size()) <
              (mesh->m_N_r + 1) * mesh->m_N_tri);
      REQUIRE(int(local.vert_r.size()) <
              (mesh->m_N_r + 1) * mesh->m_N_vert_s);

      // Spot-check value correctness at every local slot.
      check_local_matches_global(local.tri_face_area,
                                  mesh->face_area.host_ptr(), 0,
                                  mp.layout(cochain_type::tri_face));
      check_local_matches_global(local.rect_face_hodge2,
                                  mesh->hodge2.host_ptr(), N_tri_faces_global,
                                  mp.layout(cochain_type::rect_face));
      check_local_matches_global(local.h_edge_length,
                                  mesh->edge_length.host_ptr(), 0,
                                  mp.layout(cochain_type::h_edge));
      check_local_matches_global(local.v_edge_hodge1_inv,
                                  mesh->hodge1_inv.host_ptr(),
                                  N_h_edges_global,
                                  mp.layout(cochain_type::v_edge));
      check_local_matches_global(local.vert_r, mesh->vert_r.host_ptr(), 0,
                                  mp.layout(cochain_type::vertex));
    }
  }
}

// =========================================================================
// Sum of owned counts across all ranks equals the global count — the
// mesh is partitioned without omission.
// =========================================================================
TEST_CASE("mesh_local: owned counts sum to global across all ranks",
          "[prismatic][mesh_local]") {
  const int L = 2;
  const int N_r = 6;
  const int K = 3;
  auto mesh = make_mesh(L, N_r);
  auto topo = icosphere_topology::build_from_mesh(*mesh);

  const int N_tri_faces_global = (mesh->m_N_r + 1) * mesh->m_N_tri;
  const int N_rect_faces_global = mesh->m_N_r * mesh->m_N_edge_s;
  const int N_h_edges_global = (mesh->m_N_r + 1) * mesh->m_N_edge_s;
  const int N_v_edges_global = mesh->m_N_r * mesh->m_N_vert_s;
  const int N_verts_global = (mesh->m_N_r + 1) * mesh->m_N_vert_s;

  int sum_tri = 0, sum_rect = 0, sum_he = 0, sum_ve = 0, sum_vt = 0;
  for (int r = 0; r < K; ++r) {
    for (int f = 0; f < 20; ++f) {
      auto part = prismatic_partition::combined(L, N_r, K, r, f);
      part.set_topology(&topo);
      auto mp = prismatic_mesh_partition::build(part, topo);
      sum_tri  += mp.layout(cochain_type::tri_face).owned_size();
      sum_rect += mp.layout(cochain_type::rect_face).owned_size();
      sum_he   += mp.layout(cochain_type::h_edge).owned_size();
      sum_ve   += mp.layout(cochain_type::v_edge).owned_size();
      sum_vt   += mp.layout(cochain_type::vertex).owned_size();
    }
  }
  REQUIRE(sum_tri  == N_tri_faces_global);
  REQUIRE(sum_rect == N_rect_faces_global);
  REQUIRE(sum_he   == N_h_edges_global);
  REQUIRE(sum_ve   == N_v_edges_global);
  REQUIRE(sum_vt   == N_verts_global);
}
