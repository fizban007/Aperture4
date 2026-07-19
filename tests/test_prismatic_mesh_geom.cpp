// Phase 7D.1 — per-element geometry helpers (prismatic_mesh_geom.h)
// must reproduce the full build's 3D per-cochain arrays BIT-EXACTLY,
// on every element, including the boundary shells and the float
// round-trip conventions.  This is the contract that lets local
// builders compute geometry from the sphere stage alone.
#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_geom.h"

using namespace Aperture;

namespace {

void check_all(const prismatic_mesh& m) {
  namespace pg = prismatic_geom;
  const int N_h = (m.m_N_r + 1) * m.m_N_edge_s;
  const int N_tri_faces = (m.m_N_r + 1) * m.m_N_tri;

  // Vertex coordinates + dual volumes.
  for (int k = 0; k <= m.m_N_r; ++k) {
    for (int s = 0; s < m.m_N_vert_s; ++s) {
      const int vi = m.vert_idx(k, s);
      REQUIRE(pg::vert_r(m, k) == m.vert_r[vi]);
      REQUIRE(pg::vert_theta(m, s) == m.vert_theta[vi]);
      REQUIRE(pg::vert_phi(m, s) == m.vert_phi[vi]);
      REQUIRE(pg::vert_dual_vol(m, k, s) == m.vert_dual_vol[vi]);
    }
  }

  // h edges.
  for (int k = 0; k <= m.m_N_r; ++k) {
    for (int e = 0; e < m.m_N_edge_s; ++e) {
      const int ei = m.h_edge_idx(k, e);
      REQUIRE(pg::h_edge_length(m, k, e) == m.edge_length[ei]);
      REQUIRE(pg::hodge1_inv_h(m, k, e) == m.hodge1_inv[ei]);
      REQUIRE(pg::h_edge_gv0(m, k, e) == m.edge_v0[ei]);
      REQUIRE(pg::h_edge_gv1(m, k, e) == m.edge_v1[ei]);
      REQUIRE(pg::h_edge_boundary(m, k) == m.edge_boundary[ei]);
      REQUIRE(k == m.edge_radial_layer[ei]);
    }
  }

  // v edges.
  for (int k = 0; k < m.m_N_r; ++k) {
    for (int s = 0; s < m.m_N_vert_s; ++s) {
      const int ei = m.v_edge_idx(k, s);
      REQUIRE(pg::v_edge_length(m, k) == m.edge_length[ei]);
      REQUIRE(pg::hodge1_inv_v(m, k, s) == m.hodge1_inv[ei]);
      REQUIRE(pg::v_edge_gv0(m, k, s) == m.edge_v0[ei]);
      REQUIRE(pg::v_edge_gv1(m, k, s) == m.edge_v1[ei]);
      REQUIRE(pg::v_edge_boundary(m, k) == m.edge_boundary[ei]);
      REQUIRE(k == m.edge_radial_layer[ei]);
    }
  }
  (void)N_h;

  // Tri faces.
  for (int k = 0; k <= m.m_N_r; ++k) {
    for (int t = 0; t < m.m_N_tri; ++t) {
      const int fi = m.tri_face_idx(k, t);
      REQUIRE(pg::tri_area(m, k, t) == m.face_area[fi]);
      REQUIRE(pg::hodge2_tri(m, k, t) == m.hodge2[fi]);
      REQUIRE(pg::tri_face_boundary(m, k) == m.face_boundary[fi]);
      REQUIRE(k == m.face_radial_layer[fi]);
    }
  }
  (void)N_tri_faces;

  // Rect faces.
  for (int k = 0; k < m.m_N_r; ++k) {
    for (int e = 0; e < m.m_N_edge_s; ++e) {
      const int fi = m.rect_face_idx(k, e);
      REQUIRE(pg::rect_area(m, k, e) == m.face_area[fi]);
      REQUIRE(pg::hodge2_rect(m, k, e) == m.hodge2[fi]);
      REQUIRE(pg::rect_face_boundary(m, k) == m.face_boundary[fi]);
      REQUIRE(k == m.face_radial_layer[fi]);
    }
  }
}

}  // namespace

TEST_CASE("mesh geometry helpers reproduce the full build bit-exactly",
          "[prismatic][mesh_geom]") {
  prismatic_mesh m;
  m.build(2, 8, 1.0, 2.0);
  check_all(m);
}

TEST_CASE("mesh geometry helpers: different radial extent + ghost layers",
          "[prismatic][mesh_geom]") {
  prismatic_mesh m;
  m.build(1, 12, 1.0, 45.0, 2, 1);
  check_all(m);
}

TEST_CASE("sphere-only build allocates no 3D per-cochain arrays",
          "[prismatic][mesh_geom]") {
  prismatic_mesh m;
  m.build_sphere_only(2, 8, 1.0, 2.0);
  REQUIRE_FALSE(m.has_3d());
  // Counts and sphere data present.
  REQUIRE(m.m_N_tri == 320);
  REQUIRE(m.m_N_faces == 9 * 320 + 8 * 480);
  REQUIRE(m.radii.size() == 9);
  REQUIRE(m.sphere_vx.size() >= size_t(m.m_N_vert_s));
  REQUIRE(m.sphere_edge_v0.size() >= size_t(m.m_N_edge_s));
  REQUIRE(m.sph_tri_omega.size() == size_t(m.m_N_tri));
  // 3D arrays never allocated.
  REQUIRE(m.vert_r.size() == 0);
  REQUIRE(m.edge_length.size() == 0);
  REQUIRE(m.face_area.size() == 0);
  REQUIRE(m.hodge1_inv.size() == 0);
  REQUIRE(m.hodge2.size() == 0);
  REQUIRE(m.vert_dual_vol.size() == 0);
  REQUIRE(m.d1_row_ptr.size() == 0);
  REQUIRE(m.d1t_row_ptr.size() == 0);
  REQUIRE(m.edge_v0.size() == 0);
  REQUIRE(m.tri_face_v0.size() == 0);

  // The sphere stage matches a full build's sphere stage exactly.
  prismatic_mesh f;
  f.build(2, 8, 1.0, 2.0);
  for (int s = 0; s < m.m_N_vert_s; ++s) {
    REQUIRE(m.sphere_vx[s] == f.sphere_vx[s]);
    REQUIRE(m.sphere_theta[s] == f.sphere_theta[s]);
  }
  for (int i = 0; i < m.m_N_tri; ++i) {
    REQUIRE(m.sph_tri_omega[i] == f.sph_tri_omega[i]);
  }
  for (int k = 0; k <= m.m_N_r; ++k) REQUIRE(m.radii[k] == f.radii[k]);
}
