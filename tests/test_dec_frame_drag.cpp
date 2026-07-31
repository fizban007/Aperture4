/*
 * Copyright (c) 2026 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

// ===========================================================================
// Frame-drag ("fake GR") EMF term in the DEC solver core.
//
// The discrete W_e = ∮_e (v_LT x B)·dl is a per-edge linear functional of
// adjacent face fluxes.  Measured accuracy structure (see the
// build_frame_drag header note): v-edges exact to round-off for uniform
// B; h-edges O(h²) — the edge-parallel component of the circulation is
// unrepresentable by the adjacent-face stencil (all its normals are ⊥
// the edge tangent to O(h)) and is itself O(h²) on the curved arc.
// These tests therefore assert (a) machine-exactness on v-edges,
// (b) SECOND-ORDER CONVERGENCE of the h-edge and face-curl residuals
// under mesh refinement, and (c) the analytic face-curl target through
// the production faraday() operator — which exercises probe conventions,
// the min-norm solve, CSR wiring, and edge orientations end to end.
// A sign error anywhere fails (c) at the 200% level, not the few-e-4
// truncation level.
// ===========================================================================

#include "catch2/catch_all.hpp"
#include "systems/prismatic/dec_solver_dist.h"
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cmath>
#include <vector>

using namespace Aperture;

namespace {

using policy = prismatic_exec_policy_host;
using core_t = dec_solver_dist<policy>;

constexpr int TN_r = 8;

struct fd_fixture {
  prismatic_mesh mesh;
  icosphere_topology topo;
  prismatic_partition part;
  prismatic_mesh_partition mp_b;
  core_t core;
  std::vector<double> Ntri, Nrect;  // converged face probe vectors

  explicit fd_fixture(int L) : part(prismatic_partition::single_rank(L, TN_r)) {
    mesh.build(L, TN_r, 1.0, 2.0);
    topo = icosphere_topology::build_from_mesh(mesh);
    part.set_topology(&topo);
    mp_b = prismatic_mesh_partition::build(part, topo);
    core.build(mesh, mp_b);
    compute_probes();
  }

  void compute_probes() {
    auto lp = core.get_lp(exec_tags::host{});
    auto mp = mesh.host_ptrs();
    Ntri.assign(size_t(3) * lp.n_local_tri, 0.0);
    for (int f = 0; f < lp.n_local_tri; f++) {
      gidx_t g = lp.tri_face_l2g[f];
      gidx_t v0, v1, v2;
      tri_face_vertex_ids(mp, g, v0, v1, v2);
      Scalar r0, ax, ay, az, r1, bx, by, bz, r2, cx, cy, cz;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx, by, bz);
      vertex_unit(mp, v2, r2, cx, cy, cz);
      for (int c = 0; c < 3; c++) {
        Ntri[3 * f + c] = gauss_quad(
            [&](double u) -> double {
              return gauss_quad(
                  [&](double t) -> double {
                    Scalar x, y, z, n[3];
                    tri_sphere_sample(r0, ax, ay, az, bx, by, bz, cx, cy,
                                      cz, Scalar(u), Scalar(t), x, y, z,
                                      n[0], n[1], n[2]);
                    return n[c];
                  },
                  0.0, 1.0);
            },
            0.0, 1.0);
      }
    }
    Nrect.assign(size_t(3) * lp.n_local_rect, 0.0);
    for (int f = 0; f < lp.n_local_rect; f++) {
      gidx_t g = lp.rect_face_l2g[f];
      gidx_t v0, v1, v3;
      rect_face_vertex_ids(mp, g, v0, v1, v3);
      Scalar rlo, ax, ay, az, rt, bx, by, bz, rhi, u3x, u3y, u3z;
      vertex_unit(mp, v0, rlo, ax, ay, az);
      vertex_unit(mp, v1, rt, bx, by, bz);
      vertex_unit(mp, v3, rhi, u3x, u3y, u3z);
      (void)rt; (void)u3x; (void)u3y; (void)u3z;
      for (int c = 0; c < 3; c++) {
        Nrect[3 * f + c] = gauss_quad(
            [&](double u) -> double {
              return gauss_quad(
                  [&](double v) -> double {
                    Scalar x, y, z, n[3];
                    rect_sphere_sample(rlo, rhi, ax, ay, az, bx, by, bz,
                                       Scalar(u), Scalar(v), x, y, z,
                                       n[0], n[1], n[2]);
                    return n[c];
                  },
                  0.0, 1.0);
            },
            0.0, 1.0);
      }
    }
  }

  void fill_uniform_B(const double Bv[3], buffer<Scalar>& Bbuf) {
    auto lp = core.get_lp(exec_tags::host{});
    const int bs = core.b_split();
    for (int f = 0; f < lp.n_local_tri; f++) {
      Bbuf[f] = Scalar(Bv[0]*Ntri[3*f] + Bv[1]*Ntri[3*f+1] +
                       Bv[2]*Ntri[3*f+2]);
    }
    for (int f = 0; f < lp.n_local_rect; f++) {
      Bbuf[bs + f] = Scalar(Bv[0]*Nrect[3*f] + Bv[1]*Nrect[3*f+1] +
                            Bv[2]*Nrect[3*f+2]);
    }
  }

  // Max |W_e - ∮(v x B)·dl| over owned edges, split by edge kind, for a
  // uniform B and the given drag profile.
  void edge_residual(Scalar omega0, int p, const double Bv[3],
                     double& err_h, double& err_v) {
    auto lp = core.get_lp(exec_tags::host{});
    auto mp = mesh.host_ptrs();
    core.build_frame_drag(omega0, Scalar(1.0), p);
    const int ne = core.n_edges_local(), nfl = core.n_faces_local();
    const int es = core.e_split();
    buffer<Scalar> E0b, Bb, B0b, W;
    for (auto* b : {&E0b, &W}) { b->set_memtype(MemType::host_only); b->resize(ne); }
    for (auto* b : {&Bb, &B0b}) { b->set_memtype(MemType::host_only); b->resize(nfl); }
    E0b.assign(0);
    B0b.assign(0);
    fill_uniform_B(Bv, Bb);
    core.frame_drag_eff_E(E0b, Bb, B0b, W);

    const int NQ = 200;
    err_h = err_v = 0;
    for (int e = 0; e < lp.n_owned_he; e++) {
      gidx_t g = lp.h_edge_l2g[e];
      gidx_t v0, v1;
      h_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, bx, by, bz;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx, by, bz);
      double ref = 0;
      for (int i = 0; i < NQ; i++) {
        Scalar t = (i + Scalar(0.5)) / NQ;
        Scalar x, y, z, dlx, dly, dlz;
        h_edge_sphere_sample(r0, ax, ay, az, bx, by, bz, t, x, y, z, dlx,
                             dly, dlz);
        Scalar vx, vy, vz;
        frame_drag_velocity(x, y, z, omega0, Scalar(1.0), p, vx, vy, vz);
        ref += ((vy*Bv[2] - vz*Bv[1]) * dlx + (vz*Bv[0] - vx*Bv[2]) * dly +
                (vx*Bv[1] - vy*Bv[0]) * dlz) / NQ;
      }
      err_h = std::max(err_h, std::abs(double(W[e]) - ref));
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      gidx_t g = lp.v_edge_l2g[e];
      gidx_t v0, v1;
      v_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, a1x, a1y, a1z;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, a1x, a1y, a1z);
      (void)a1x; (void)a1y; (void)a1z;
      double ref = 0;
      for (int i = 0; i < NQ; i++) {
        double t = (i + 0.5) / NQ;
        double rt = (1.0 - t) * r0 + t * r1;
        Scalar x = Scalar(rt*ax), y = Scalar(rt*ay), z = Scalar(rt*az);
        double dl[3] = {double(r1 - r0) * ax, double(r1 - r0) * ay,
                        double(r1 - r0) * az};
        Scalar vx, vy, vz;
        frame_drag_velocity(x, y, z, omega0, Scalar(1.0), p, vx, vy, vz);
        ref += ((vy*Bv[2] - vz*Bv[1]) * dl[0] +
                (vz*Bv[0] - vx*Bv[2]) * dl[1] +
                (vx*Bv[1] - vy*Bv[0]) * dl[2]) / NQ;
      }
      err_v = std::max(err_v, std::abs(double(W[es + e]) - ref));
    }
  }

  // Max |d1(W)[f] - (omega ẑ x B)·N_f| over owned faces (uniform drag
  // p = 0, uniform B — the analytic-curl Stokes check through the
  // production faraday() operator), plus the max analytic reference.
  void curl_residual(Scalar omega0, const double Bv[3], double& err,
                     double& ref_max) {
    auto lp = core.get_lp(exec_tags::host{});
    core.build_frame_drag(omega0, Scalar(1.0), 0);
    const int ne = core.n_edges_local(), nfl = core.n_faces_local();
    const int bs = core.b_split();
    buffer<Scalar> E0b, Bb, B0b, W, Bcurl;
    for (auto* b : {&E0b, &W}) { b->set_memtype(MemType::host_only); b->resize(ne); }
    for (auto* b : {&Bb, &B0b, &Bcurl}) { b->set_memtype(MemType::host_only); b->resize(nfl); }
    E0b.assign(0);
    B0b.assign(0);
    Bcurl.assign(0);
    fill_uniform_B(Bv, Bb);
    core.frame_drag_eff_E(E0b, Bb, B0b, W);
    // Bcurl[f] = +d1W[f] via the production Faraday with dt = -1.
    core.faraday(W, Bcurl, -1.0);

    const double curl[3] = {-double(omega0) * Bv[1],
                            double(omega0) * Bv[0], 0.0};
    err = ref_max = 0;
    for (int f = 0; f < lp.n_owned_tri; f++) {
      double ref = curl[0]*Ntri[3*f] + curl[1]*Ntri[3*f+1] +
                   curl[2]*Ntri[3*f+2];
      err = std::max(err, std::abs(double(Bcurl[f]) - ref));
      ref_max = std::max(ref_max, std::abs(ref));
    }
    for (int f = 0; f < lp.n_owned_rect; f++) {
      double ref = curl[0]*Nrect[3*f] + curl[1]*Nrect[3*f+1] +
                   curl[2]*Nrect[3*f+2];
      err = std::max(err, std::abs(double(Bcurl[bs + f]) - ref));
      ref_max = std::max(ref_max, std::abs(ref));
    }
  }
};

}  // namespace

TEST_CASE("Frame-drag EMF: v-edges exact, h-edges 2nd-order convergent",
          "[prismatic][framedrag]") {
  const Scalar omega0 = 0.25;
  const double Bv[3] = {0.4, 0.25, 0.8};

  fd_fixture f2(2), f3(3);
  double eh2, ev2, eh3, ev3;
  f2.edge_residual(omega0, 3, Bv, eh2, ev2);
  f3.edge_residual(omega0, 3, Bv, eh3, ev3);

  INFO("h-edge err L2 = " << eh2 << " -> L3 = " << eh3
                          << " (ratio " << eh2 / eh3 << ")");
  INFO("v-edge err L2 = " << ev2 << ", L3 = " << ev3);
  // v-edges: exact for uniform B (measured ~3e-9, float storage).
  REQUIRE(ev2 < 1e-6);
  REQUIRE(ev3 < 1e-6);
  // h-edges: the O(h²) unrepresentable-B_parallel truncation; measured
  // ratio 4.19 at L2->L3.  Require clearly better than 1st order.
  REQUIRE(eh2 < 2e-3);        // absolute sanity at L2
  REQUIRE(eh2 / eh3 > 3.0);   // 2nd-order convergence (4.0 ideal)
}

TEST_CASE("Frame-drag EMF: discrete curl converges to analytic (Stokes)",
          "[prismatic][framedrag]") {
  const Scalar omega0 = 0.3;
  const double Bvs[2][3] = {{1, 0, 0}, {0.3, -0.7, 0.55}};

  fd_fixture f2(2), f3(3);
  for (auto& Bv : Bvs) {
    double e2, r2, e3, r3;
    f2.curl_residual(omega0, Bv, e2, r2);
    f3.curl_residual(omega0, Bv, e3, r3);
    INFO("B = (" << Bv[0] << "," << Bv[1] << "," << Bv[2] << "): err L2 = "
                 << e2 << " (ref " << r2 << ") -> L3 = " << e3 << " (ref "
                 << r3 << "), ratio " << e2 / e3);
    // The analytic target is nonzero and the discrete curl approaches
    // it under refinement.  A sign/orientation error anywhere in the
    // chain gives err ~ 2*ref, not a converging few-percent residual.
    REQUIRE(r2 > 1e-3);
    REQUIRE(e2 < 0.15 * r2);
    REQUIRE(e2 / e3 > 2.0);
  }
}

TEST_CASE("Frame-drag EMF: zero drag is an exact no-op",
          "[prismatic][framedrag]") {
  fd_fixture fx(2);
  fx.core.build_frame_drag(Scalar(0), Scalar(1.0), 3);

  const int ne = fx.core.n_edges_local();
  const int nf = fx.core.n_faces_local();
  buffer<Scalar> Eb, Bb, B0b, Eeff;
  for (auto* b : {&Eb, &Eeff}) { b->set_memtype(MemType::host_only); b->resize(ne); }
  for (auto* b : {&Bb, &B0b}) { b->set_memtype(MemType::host_only); b->resize(nf); }
  for (int e = 0; e < ne; e++) Eb[e] = std::sin(Scalar(0.013) * e);
  for (int f = 0; f < nf; f++) Bb[f] = std::cos(Scalar(0.007) * f);
  B0b.assign(0);

  fx.core.frame_drag_eff_E(Eb, Bb, B0b, Eeff);
  for (int e = 0; e < ne; e++) {
    if (Eeff[e] != Eb[e]) {
      FAIL("Eeff differs from E at edge " << e);
    }
  }
  SUCCEED("bitwise identical");
}
