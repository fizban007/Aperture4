#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "systems/prismatic/prismatic_mesh_gr_ks.h"
#include "systems/physics/metric_ks_cartesian.hpp"
#include <cmath>
#include <memory>

using namespace Aperture;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static std::unique_ptr<prismatic_mesh_gr_ks> make_test_mesh(
    Scalar a = 0.9, int L = 2, int N_r = 5,
    double r_min = 1.0, double r_max = 10.0) {
  auto mesh = std::make_unique<prismatic_mesh_gr_ks>();
  mesh->build(L, N_r, r_min, r_max);
  mesh->compute_metric(a);
  return mesh;
}

TEST_CASE("GR KS mesh: metric arrays allocated", "[prismatic_gr_ks]") {
  auto m = make_test_mesh();

  REQUIRE(m->edge_f.size() == (size_t)m->m_N_edges);
  REQUIRE(m->edge_lx.size() == (size_t)m->m_N_edges);
  REQUIRE(m->edge_alpha.size() == (size_t)m->m_N_edges);
  REQUIRE(m->edge_r.size() == (size_t)m->m_N_edges);
  REQUIRE(m->face_f.size() == (size_t)m->m_N_faces);
  REQUIRE(m->face_lx.size() == (size_t)m->m_N_faces);
  REQUIRE(m->face_alpha.size() == (size_t)m->m_N_faces);
  REQUIRE(m->face_r.size() == (size_t)m->m_N_faces);
}

TEST_CASE("GR KS mesh: edge metric matches direct computation",
          "[prismatic_gr_ks]") {
  Scalar a = 0.9;
  auto m = make_test_mesh(a);

  // Spot-check a handful of edges
  for (int e = 0; e < m->m_N_edges; e += m->m_N_edges / 10) {
    int v0 = m->edge_v0[e], v1 = m->edge_v1[e];
    Scalar mx = Scalar(0.5) * (m->vert_x[v0] + m->vert_x[v1]);
    Scalar my = Scalar(0.5) * (m->vert_y[v0] + m->vert_y[v1]);
    Scalar mz = Scalar(0.5) * (m->vert_z[v0] + m->vert_z[v1]);

    Scalar r_exp = Metric_KS_Cart::radius(mx, my, mz, a);
    Scalar f_exp = Metric_KS_Cart::f_ks(r_exp, mz, a);
    Scalar alpha_exp = Metric_KS_Cart::alpha(f_exp);

    CHECK_THAT(m->edge_r[e], WithinRel(r_exp, Scalar(1e-6)));
    CHECK_THAT(m->edge_f[e], WithinRel(f_exp, Scalar(1e-6)));
    CHECK_THAT(m->edge_alpha[e], WithinRel(alpha_exp, Scalar(1e-6)));

    // null vector norm should be 1
    Scalar l2 = m->edge_lx[e] * m->edge_lx[e] +
                m->edge_ly[e] * m->edge_ly[e] +
                m->edge_lz[e] * m->edge_lz[e];
    CHECK_THAT(l2, WithinAbs(Scalar(1.0), Scalar(1e-5)));
  }
}

TEST_CASE("GR KS mesh: face metric matches direct computation",
          "[prismatic_gr_ks]") {
  Scalar a = 0.9;
  auto m = make_test_mesh(a);

  // Check some triangular faces
  int n_tri = m->m_N_tri * (m->m_N_r + 1);
  for (int fi = 0; fi < n_tri; fi += n_tri / 10) {
    int va = m->tri_face_v0[fi];
    int vb = m->tri_face_v1[fi];
    int vc = m->tri_face_v2[fi];
    Scalar cx = (m->vert_x[va] + m->vert_x[vb] + m->vert_x[vc]) / Scalar(3.0);
    Scalar cy = (m->vert_y[va] + m->vert_y[vb] + m->vert_y[vc]) / Scalar(3.0);
    Scalar cz = (m->vert_z[va] + m->vert_z[vb] + m->vert_z[vc]) / Scalar(3.0);

    Scalar r_exp = Metric_KS_Cart::radius(cx, cy, cz, a);
    CHECK_THAT(m->face_r[fi], WithinRel(r_exp, Scalar(1e-6)));

    Scalar l2 = m->face_lx[fi] * m->face_lx[fi] +
                m->face_ly[fi] * m->face_ly[fi] +
                m->face_lz[fi] * m->face_lz[fi];
    CHECK_THAT(l2, WithinAbs(Scalar(1.0), Scalar(1e-5)));
  }

  // Check some rectangular faces
  int n_rect = m->m_N_edge_s * m->m_N_r;
  for (int ri = 0; ri < n_rect; ri += n_rect / 10) {
    int fi = n_tri + ri;
    int va = m->rect_face_v0[ri];
    int vb = m->rect_face_v1[ri];
    int vc = m->rect_face_v2[ri];
    int vd = m->rect_face_v3[ri];
    Scalar cx = Scalar(0.25) * (m->vert_x[va] + m->vert_x[vb] +
                                m->vert_x[vc] + m->vert_x[vd]);
    Scalar cy = Scalar(0.25) * (m->vert_y[va] + m->vert_y[vb] +
                                m->vert_y[vc] + m->vert_y[vd]);
    Scalar cz = Scalar(0.25) * (m->vert_z[va] + m->vert_z[vb] +
                                m->vert_z[vc] + m->vert_z[vd]);

    Scalar r_exp = Metric_KS_Cart::radius(cx, cy, cz, a);
    CHECK_THAT(m->face_r[fi], WithinRel(r_exp, Scalar(1e-6)));
  }
}

TEST_CASE("GR KS mesh: Schwarzschild limit is isotropic",
          "[prismatic_gr_ks]") {
  Scalar a = 0.0;
  auto m = make_test_mesh(a);

  // For a=0, the null vector l should be purely radial: l = r_hat = (x,y,z)/r
  for (int e = 0; e < m->m_N_edges; e += m->m_N_edges / 10) {
    int v0 = m->edge_v0[e], v1 = m->edge_v1[e];
    Scalar mx = Scalar(0.5) * (m->vert_x[v0] + m->vert_x[v1]);
    Scalar my = Scalar(0.5) * (m->vert_y[v0] + m->vert_y[v1]);
    Scalar mz = Scalar(0.5) * (m->vert_z[v0] + m->vert_z[v1]);

    Scalar r = m->edge_r[e];
    Scalar lx_exp = mx / r, ly_exp = my / r, lz_exp = mz / r;

    CHECK_THAT(m->edge_lx[e], WithinAbs(lx_exp, Scalar(1e-5)));
    CHECK_THAT(m->edge_ly[e], WithinAbs(ly_exp, Scalar(1e-5)));
    CHECK_THAT(m->edge_lz[e], WithinAbs(lz_exp, Scalar(1e-5)));
  }
}

TEST_CASE("GR KS mesh: sgb = sqrt(gamma) * beta consistency",
          "[prismatic_gr_ks]") {
  Scalar a = 0.9;
  auto m = make_test_mesh(a);

  // sgb^i should equal sqrt(1+f) * f/(1+f) * l_i = f/sqrt(1+f) * l_i
  for (int e = 0; e < m->m_N_edges; e += m->m_N_edges / 10) {
    Scalar fv = m->edge_f[e];
    Scalar coeff = fv / math::sqrt(1.0f + fv);
    CHECK_THAT(m->edge_sgb_x[e],
               WithinRel(coeff * m->edge_lx[e], Scalar(1e-5)));
    CHECK_THAT(m->edge_sgb_y[e],
               WithinRel(coeff * m->edge_ly[e], Scalar(1e-5)));
    CHECK_THAT(m->edge_sgb_z[e],
               WithinRel(coeff * m->edge_lz[e], Scalar(1e-5)));
  }
}

TEST_CASE("GR KS mesh: ptrs struct has correct pointers",
          "[prismatic_gr_ks]") {
  auto m = make_test_mesh();
  auto p = m->host_ptrs_gr();

  // Base class fields should be populated
  REQUIRE(p.N_edges == m->m_N_edges);
  REQUIRE(p.N_faces == m->m_N_faces);
  REQUIRE(p.hodge1_inv != nullptr);
  REQUIRE(p.hodge2 != nullptr);

  // GR fields should point into the mesh buffers
  REQUIRE(p.edge_f == m->edge_f.host_ptr());
  REQUIRE(p.face_f == m->face_f.host_ptr());
  REQUIRE(p.edge_alpha == m->edge_alpha.host_ptr());
  REQUIRE(p.face_alpha == m->face_alpha.host_ptr());

  // Values through ptrs should match direct buffer access
  for (int e = 0; e < m->m_N_edges; e += m->m_N_edges / 5) {
    CHECK(p.edge_f[e] == m->edge_f[e]);
    CHECK(p.edge_alpha[e] == m->edge_alpha[e]);
  }
}

TEST_CASE("GR KS mesh: ptrs lower/raise helpers", "[prismatic_gr_ks]") {
  auto m = make_test_mesh(0.9);
  auto p = m->host_ptrs_gr();

  int e = m->m_N_edges / 3;  // arbitrary edge
  Scalar vx = 1.3, vy = -0.7, vz = 2.1;
  Scalar wx, wy, wz;
  Scalar rx, ry, rz;

  p.lower_at_edge(e, vx, vy, vz, wx, wy, wz);
  p.raise_at_edge(e, wx, wy, wz, rx, ry, rz);

  CHECK_THAT(rx, WithinRel(vx, Scalar(1e-5)));
  CHECK_THAT(ry, WithinRel(vy, Scalar(1e-5)));
  CHECK_THAT(rz, WithinRel(vz, Scalar(1e-5)));
}
