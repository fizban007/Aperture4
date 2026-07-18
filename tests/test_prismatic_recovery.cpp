// Tests for the vertex-recovery B-gather (prismatic_vertex_recovery.h)
// and the gnomonic point-location predicate.  Mirrors the validated
// Python prototype (python/prismatic_recovery.py, roadmap A1).

#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_ptrs.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include "systems/prismatic/prismatic_particles.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"

#include "catch2/catch_all.hpp"
#include <cmath>
#include <functional>
#include <random>
#include <vector>

using namespace Aperture;

namespace {

struct dv3 {
  double x, y, z;
};
dv3 operator+(dv3 a, dv3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
dv3 operator-(dv3 a, dv3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
dv3 operator*(double s, dv3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(dv3 a, dv3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
dv3 cross(dv3 a, dv3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(dv3 a) { return std::sqrt(dot(a, a)); }

using field_fn = std::function<dv3(dv3)>;

// Face-flux cochain of an analytic field by Gauss quadrature over the
// curved faces (same parametrization as the recovery weight build).
std::vector<Scalar> flux_cochain(const prismatic_mesh& mesh,
                                 const field_fn& B) {
  constexpr int NQ = 6;
  static const double GX[NQ] = {0.03376524289842399, 0.16939530676686776,
                                0.3806904069584015,  0.6193095930415985,
                                0.8306046932331322,  0.966234757101576};
  static const double GW[NQ] = {0.08566224618958517, 0.18038078652406930,
                                0.23395696728634552, 0.23395696728634552,
                                0.18038078652406930, 0.08566224618958517};
  std::vector<Scalar> B_f(mesh.m_N_faces, 0);
  auto sv = [&](int s) -> dv3 {
    return {mesh.sphere_vx[s], mesh.sphere_vy[s], mesh.sphere_vz[s]};
  };
  // shell triangles
  for (int k = 0; k <= mesh.m_N_r; k++) {
    double r = mesh.radii[k];
    for (int t = 0; t < mesh.m_N_tri; t++) {
      dv3 u0 = sv(mesh.tri_verts[t * 3 + 0]);
      dv3 u1 = sv(mesh.tri_verts[t * 3 + 1]);
      dv3 u2 = sv(mesh.tri_verts[t * 3 + 2]);
      double acc = 0;
      for (int a = 0; a < NQ; a++)
        for (int b = 0; b < NQ; b++) {
          double l1 = GX[a], l2 = GX[b] * (1 - l1);
          double w = GW[a] * GW[b] * (1 - l1);
          dv3 P = (1 - l1 - l2) * u0 + l1 * u1 + l2 * u2;
          double Pn = norm(P);
          dv3 Nh = (1 / Pn) * P;
          dv3 d1 = u1 - u0, d2 = u2 - u0;
          dv3 dN1 = (1 / Pn) * (d1 - dot(Nh, d1) * Nh);
          dv3 dN2 = (1 / Pn) * (d2 - dot(Nh, d2) * Nh);
          dv3 dA = cross(r * dN1, r * dN2);
          acc += w * dot(B(r * Nh), dA);
        }
      B_f[k * mesh.m_N_tri + t] = Scalar(acc);
    }
  }
  // rect faces
  int n_tri_faces = (mesh.m_N_r + 1) * mesh.m_N_tri;
  for (int k = 0; k < mesh.m_N_r; k++) {
    double ra = mesh.radii[k], rb = mesh.radii[k + 1];
    for (int e = 0; e < mesh.m_N_edge_s; e++) {
      dv3 u0 = sv(mesh.edge_v0[e]);
      dv3 u1 = sv(mesh.edge_v1[e]);
      double acc = 0;
      for (int a = 0; a < NQ; a++)
        for (int b = 0; b < NQ; b++) {
          double u = GX[a], z = GX[b], w = GW[a] * GW[b];
          dv3 P = (1 - u) * u0 + u * u1;
          double Pn = norm(P);
          dv3 Nh = (1 / Pn) * P;
          dv3 dPu = u1 - u0;
          dv3 dNu = (1 / Pn) * (dPu - dot(Nh, dPu) * Nh);
          double r = ra + (rb - ra) * z;
          dv3 dA = cross(r * dNu, (rb - ra) * Nh);
          acc += w * dot(B(r * Nh), dA);
        }
      B_f[n_tri_faces + k * mesh.m_N_edge_s + e] = Scalar(acc);
    }
  }
  return B_f;
}

// Run the vertex-field kernel through the host exec policy (same code
// path as the GPU launch).
void compute_Bv(prismatic_vertex_recovery& rec, const prismatic_mesh& mesh,
                const std::vector<Scalar>& B_f) {
  auto mp = mesh.host_ptrs();
  auto rp = rec.host_ptrs();
  const Scalar* bf = B_f.data();
  prismatic_exec_policy_host::launch([mp, rp, bf]() {
    prismatic_exec_policy_host::loop(0, rp.N_verts, [&](int vi) {
      rp.compute_vertex_B(mp, bf, vi);
    });
  });
}

struct located {
  int tri, layer;
  Scalar l[3], zeta;
};
located locate(const prismatic_mesh_ptrs& mp, dv3 p, int hint = 0) {
  located out;
  double r = norm(p);
  out.layer = mp.find_radial_layer(Scalar(r));
  REQUIRE(out.layer >= 0);
  out.zeta = mp.compute_zeta(out.layer, Scalar(r));
  dv3 ph = (1 / r) * p;
  out.tri = mp.find_triangle(Scalar(ph.x), Scalar(ph.y), Scalar(ph.z), hint);
  mp.compute_barycentric(out.tri, Scalar(ph.x), Scalar(ph.y), Scalar(ph.z),
                         out.l[0], out.l[1], out.l[2]);
  return out;
}

}  // namespace

TEST_CASE("Gnomonic locator: exact tiling, no orphan points",
          "[prismatic][recovery]") {
  prismatic_mesh mesh;
  mesh.build(2, 4, 1.0, 2.0);
  auto mp = mesh.host_ptrs();

  std::mt19937 rng(42);
  std::normal_distribution<double> gauss;
  for (int trial = 0; trial < 2000; trial++) {
    dv3 u{gauss(rng), gauss(rng), gauss(rng)};
    u = (1 / norm(u)) * u;
    // start the walk from a random (often far) hint
    int hint = trial % mesh.m_N_tri;
    int t = mp.find_triangle(Scalar(u.x), Scalar(u.y), Scalar(u.z), hint);
    Scalar l1, l2, l3;
    mp.compute_barycentric(t, Scalar(u.x), Scalar(u.y), Scalar(u.z), l1, l2, l3);
    // containing triangle: all coords nonnegative (float roundoff slack)
    REQUIRE(l1 >= Scalar(-1e-5));
    REQUIRE(l2 >= Scalar(-1e-5));
    REQUIRE(l3 >= Scalar(-1e-5));
    REQUIRE(std::abs(l1 + l2 + l3 - Scalar(1)) < Scalar(1e-5));
  }
}

TEST_CASE("Vertex recovery: constant field reproduced exactly",
          "[prismatic][recovery]") {
  prismatic_mesh mesh;
  mesh.build(2, 6, 1.0, 2.0);
  prismatic_vertex_recovery rec;
  rec.build(mesh);
  REQUIRE(rec.max_condition() < 100.0);

  dv3 B0{0.3, -1.1, 0.7};
  auto B_f = flux_cochain(mesh, [&](dv3) { return B0; });
  compute_Bv(rec, mesh, B_f);

  double max_err = 0;
  for (int vi = 0; vi < mesh.m_N_verts; vi++) {
    max_err = std::max(max_err, std::abs(rec.Bv[0 * mesh.m_N_verts + vi] - B0.x));
    max_err = std::max(max_err, std::abs(rec.Bv[1 * mesh.m_N_verts + vi] - B0.y));
    max_err = std::max(max_err, std::abs(rec.Bv[2 * mesh.m_N_verts + vi] - B0.z));
  }
  REQUIRE(max_err < 2e-4);  // float32 fluxes + weights

  // gather at random interior points
  auto mp = mesh.host_ptrs();
  std::mt19937 rng(7);
  std::normal_distribution<double> gauss;
  std::uniform_real_distribution<double> ur(1.01, 1.99);
  for (int trial = 0; trial < 200; trial++) {
    dv3 u{gauss(rng), gauss(rng), gauss(rng)};
    u = (ur(rng) / norm(u)) * u;
    auto lc = locate(mp, u);
    Scalar Bx, By, Bz;
    interpolate_B_recovery(mp, rec.Bv.host_ptr(), lc.tri, lc.layer, lc.l,
                           lc.zeta, Bx, By, Bz);
    REQUIRE(std::abs(Bx - B0.x) < 2e-4);
    REQUIRE(std::abs(By - B0.y) < 2e-4);
    REQUIRE(std::abs(Bz - B0.z) < 2e-4);
  }
}

TEST_CASE("Vertex recovery: linear solenoidal field vertex values",
          "[prismatic][recovery]") {
  prismatic_mesh mesh;
  mesh.build(2, 6, 1.0, 2.0);
  prismatic_vertex_recovery rec;
  rec.build(mesh);

  // B = B0 + G x with trace-free G
  dv3 B0{0.3, -1.1, 0.7};
  double G[3][3] = {{0.2, 0.5, -0.3}, {0.1, -0.4, 0.6}, {-0.2, 0.3, 0.2}};
  auto lin = [&](dv3 p) -> dv3 {
    return {B0.x + G[0][0] * p.x + G[0][1] * p.y + G[0][2] * p.z,
            B0.y + G[1][0] * p.x + G[1][1] * p.y + G[1][2] * p.z,
            B0.z + G[2][0] * p.x + G[2][1] * p.y + G[2][2] * p.z};
  };
  auto B_f = flux_cochain(mesh, lin);
  compute_Bv(rec, mesh, B_f);

  double max_err = 0;
  for (int vi = 0; vi < mesh.m_N_verts; vi++) {
    int k = vi / mesh.m_N_vert_s, s = vi % mesh.m_N_vert_s;
    double r = mesh.radii[k];
    dv3 xv{r * mesh.sphere_vx[s], r * mesh.sphere_vy[s], r * mesh.sphere_vz[s]};
    dv3 Bex = lin(xv);
    max_err = std::max(max_err, std::abs(rec.Bv[0 * mesh.m_N_verts + vi] - Bex.x));
    max_err = std::max(max_err, std::abs(rec.Bv[1 * mesh.m_N_verts + vi] - Bex.y));
    max_err = std::max(max_err, std::abs(rec.Bv[2 * mesh.m_N_verts + vi] - Bex.z));
  }
  REQUIRE(max_err < 1e-3);  // linear-reproducing up to float32 roundoff
}

TEST_CASE("Vertex recovery: gather is C0 across faces (primal is not)",
          "[prismatic][recovery]") {
  prismatic_mesh mesh;
  mesh.build(2, 6, 1.0, 2.0);
  prismatic_vertex_recovery rec;
  rec.build(mesh);
  auto mp = mesh.host_ptrs();

  // dipole field: B = (3 (m.rh) rh - m)/r^3, m = z_hat
  auto dip = [](dv3 p) -> dv3 {
    double r = norm(p);
    dv3 rh = (1 / r) * p;
    double mr = rh.z;
    double ir3 = 1.0 / (r * r * r);
    return {ir3 * 3 * mr * rh.x, ir3 * 3 * mr * rh.y,
            ir3 * (3 * mr * rh.z - 1.0)};
  };
  auto B_f = flux_cochain(mesh, dip);
  compute_Bv(rec, mesh, B_f);

  std::mt19937 rng(3);
  std::uniform_int_distribution<int> epick(0, mesh.m_N_edge_s - 1);
  std::uniform_real_distribution<double> umid(0.3, 0.7), urad(1.2, 1.8);
  double max_jump = 0;
  const double eps = 1e-5;
  for (int trial = 0; trial < 100; trial++) {
    int e = epick(rng);
    int s0 = mesh.edge_v0[e], s1 = mesh.edge_v1[e];
    dv3 u0{mesh.sphere_vx[s0], mesh.sphere_vy[s0], mesh.sphere_vz[s0]};
    dv3 u1{mesh.sphere_vx[s1], mesh.sphere_vy[s1], mesh.sphere_vz[s1]};
    double tm = umid(rng);
    dv3 pm = (1 - tm) * u0 + tm * u1;
    pm = (1 / norm(pm)) * pm;
    dv3 nt = cross(pm, u1 - u0);
    nt = (1 / norm(nt)) * nt;
    double r = urad(rng);
    dv3 pa = pm + eps * nt, pb = pm - eps * nt;
    pa = (r / norm(pa)) * pa;
    pb = (r / norm(pb)) * pb;
    auto la = locate(mp, pa);
    auto lb = locate(mp, pb, la.tri);
    Scalar Bax, Bay, Baz, Bbx, Bby, Bbz;
    interpolate_B_recovery(mp, rec.Bv.host_ptr(), la.tri, la.layer, la.l,
                           la.zeta, Bax, Bay, Baz);
    interpolate_B_recovery(mp, rec.Bv.host_ptr(), lb.tri, lb.layer, lb.l,
                           lb.zeta, Bbx, Bby, Bbz);
    double jump = std::sqrt(double(Bax - Bbx) * (Bax - Bbx) +
                            double(Bay - Bby) * (Bay - Bby) +
                            double(Baz - Bbz) * (Baz - Bbz));
    dv3 Bex = dip(r * pm);
    max_jump = std::max(max_jump, jump / norm(Bex));
  }
  // primal jumps at this resolution are O(10%); recovery must be at
  // float32-roundoff scale
  REQUIRE(max_jump < 1e-3);
}


TEST_CASE("GCA: ExB drift with recovery gather matches analytic",
          "[prismatic][recovery][gca]") {
  // L3: the flattened-prism Whitney E-basis reproduces constants only
  // to O(h^2) (~8% at L2, ~2% at L3); the drift error tracks that
  // truncation, not the GCA integrator.
  prismatic_mesh mesh;
  mesh.build(3, 12, 1.0, 2.0);
  prismatic_vertex_recovery rec;
  rec.build(mesh);

  // Uniform crossed fields: B = 50 z_hat, E = 5 x_hat ->
  // v_d = E x B / B^2 = -0.1 y_hat, independent of charge.
  dv3 B0{0, 0, 50.0};
  auto B_f = flux_cochain(mesh, [&](dv3) { return B0; });
  compute_Bv(rec, mesh, B_f);
  dv3 E0{5.0, 0, 0};
  std::vector<Scalar> E_e(mesh.m_N_edges);
  auto vpos = [&](int v) {
    double r = mesh.vert_r[v], th = mesh.vert_theta[v], ph = mesh.vert_phi[v];
    return dv3{r * std::sin(th) * std::cos(ph),
               r * std::sin(th) * std::sin(ph), r * std::cos(th)};
  };
  for (int e = 0; e < mesh.m_N_edges; e++) {
    dv3 a = vpos(mesh.edge_v0[e]), b = vpos(mesh.edge_v1[e]);
    E_e[e] = Scalar(E0.x * (b.x - a.x) + E0.y * (b.y - a.y) +
                    E0.z * (b.z - a.z));
  }

  prismatic_particles_t ptc(4, MemType::host_only);
  ptc.init();
  auto mp = mesh.host_ptrs();
  auto p = ptc.get_host_ptrs();
  located lc = locate(mp, {1.5, 0, 0});
  p.x1[0] = lc.l[0]; p.x2[0] = lc.l[1]; p.x3[0] = lc.zeta;
  p.p1[0] = 0; p.p2[0] = 0; p.p3[0] = 0;   // u_par = 0, mu = 0 (cold)
  p.E[0] = 1; p.weight[0] = 1;
  p.cell[0] = prism_cell_encode(lc.tri, lc.layer, mesh.m_N_tri);
  p.flag[0] = gen_ptc_type_flag(PtcType::electron);
  p.id[0] = 0;
  ptc.set_num(1);

  auto rp = rec.host_ptrs();
  const Scalar dt = 0.05;
  const int nsteps = 60;  // t = 3 -> drift displacement -0.3 y_hat
  for (int s = 0; s < nsteps; s++) {
    update_single_particle(mp, mesh.m_N_tri, p, 0, E_e.data(), B_f.data(),
                           nullptr, nullptr, Scalar(-1), Scalar(1), dt,
                           true, false, rp.Bv);
    REQUIRE(p.cell[0] != empty_cell);
  }

  int tri, layer;
  prism_cell_decode(p.cell[0], mesh.m_N_tri, tri, layer);
  Scalar x, y, z;
  local_to_cartesian_impl(mp, tri, layer, p.x1[0], p.x2[0], p.x3[0], x, y, z);

  // Drift accuracy: interpolation is exact for constants, so the error
  // budget is the fixed-point iteration + float round-off.
  REQUIRE(std::abs(x - 1.5) < 0.02);
  REQUIRE(std::abs(y + 0.3) < 0.02);
  REQUIRE(std::abs(z) < 0.01);
  // GCA invariants: mu conserved exactly, no parallel acceleration
  // (E.B = 0), Gamma ~ drift kappa.
  REQUIRE(p.p2[0] == Scalar(0));
  REQUIRE(std::abs(p.p1[0]) < 1e-5);
  REQUIRE(std::abs(p.E[0] - 1.0 / std::sqrt(1.0 - 0.01)) < 5e-4);
}
