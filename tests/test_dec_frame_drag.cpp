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
        frame_drag_shift(x, y, z, omega0, Scalar(1.0), p, vx, vy, vz);
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
        frame_drag_shift(x, y, z, omega0, Scalar(1.0), p, vx, vy, vz);
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

    // curl(beta x B) for uniform drag (p = 0) and uniform B, with the
    // SHIFT beta = -v_LT: identities give curl(v x B) = (B.grad)v = omega
    // zhat x B for v = omega zhat x x, so beta contributes the negative.
    const double curl[3] = {double(omega0) * Bv[1],
                            -double(omega0) * Bv[0], 0.0};
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

  // -----------------------------------------------------------------------
  // SIGN / EQUILIBRIUM probe.  Everything above validates the DISCRETIZATION
  // of W_e; all of it passes identically under v_LT -> -v_LT.  This one
  // asks the physical question instead: which sign of the frame-drag term
  // makes the Muslimov-Tsygan state stationary?
  //
  // Build the analytic MT state -- aligned dipole B, plasma drifting at
  // omega_eff(r) = Omega - omega_lt(r), and the ideal-MHD field
  // E = -(u x B) integrated exactly along every edge.  Then:
  //
  //   E - W = -(omega_eff + omega_lt)(zhat x x) x B = -Omega (zhat x x) x B
  //           => RIGID rotation of an axisymmetric B  => curl = 0
  //   E + W = -(omega_eff - omega_lt)(zhat x x) x B
  //           = -(Omega - 2 omega_lt(r))(zhat x x) x B
  //           => DIFFERENTIAL rotation => curl != 0, and roughly 2x the
  //              curl of E alone (d/dr of Omega-2w_lt is -2 w_lt', vs -w_lt')
  //
  // d1 of an exactly-integrated edge cochain IS the face flux of curl by
  // Stokes, so the residual on the vanishing branch is purely the W stencil
  // truncation.  Returns the RMS of d1(.) over owned faces for E, E+W, E-W.
  //
  // NORM: rms, not max.  The two branches attain their maxima on DIFFERENT
  // faces, so a max-norm reports ~0.5*curl(E) on the cancelling branch even
  // when the cancellation is pointwise near-perfect.  (Measured: max-norm
  // gives 2.04 / 0.50 where rms gives 2.00 / ~0.)
  //
  // FIELD: uniform_B = true uses B = B0 zhat.  The equilibrium argument
  // needs only AXISYMMETRY ABOUT THE SPIN AXIS, which a uniform axial field
  // has, and the W stencil is built to be exact for uniform B (v-edges to
  // round-off) -- so the cancelling branch goes to zero at the truncation
  // floor instead of being buried under the r^-3 dipole stencil error.
  // uniform_B = false runs the same probe on the physical dipole, where the
  // stencil error is real and the thresholds must be looser.
  // -----------------------------------------------------------------------
  // curl_E    : rms d1(E)              -- the differential-rotation signal
  // curl_eff  : rms d1(frame_drag_eff_E(E,B))  -- THE SHIPPED OPERATOR
  // curl_flip : rms d1(E - W)          -- the opposite sign convention
  void mt_curl(Scalar Omega, Scalar omega_lt0, int p, bool uniform_B,
               double& curl_E, double& curl_eff, double& curl_flip) {
    auto lp = core.get_lp(exec_tags::host{});
    auto mp = mesh.host_ptrs();
    core.build_frame_drag(omega_lt0, Scalar(1.0), p);
    const int ne = core.n_edges_local(), nfl = core.n_faces_local();
    const int es = core.e_split(), bs = core.b_split();

    buffer<Scalar> Eb, Bb, B0b, W, Ep, Em, cE, cP, cM;
    for (auto* b : {&Eb, &W, &Ep, &Em}) {
      b->set_memtype(MemType::host_only); b->resize(ne); b->assign(0);
    }
    for (auto* b : {&Bb, &B0b, &cE, &cP, &cM}) {
      b->set_memtype(MemType::host_only); b->resize(nfl); b->assign(0);
    }
    const double Bu[3] = {0.0, 0.0, 1.0};
    const Scalar mz = Scalar(1);
    if (uniform_B) fill_uniform_B(Bu, Bb);
    else core.fill_dipole_B(Bb, Scalar(0), Scalar(0), mz);

    // E(x) = -(u x B) with u = (Omega - omega_lt(r)) zhat x x.
    auto E_at = [&](Scalar x, Scalar y, Scalar z, double& ex, double& ey,
                    double& ez) {
      Scalar r = std::sqrt(x * x + y * y + z * z);
      Scalar om = Omega - frame_drag_omega(r, omega_lt0, Scalar(1.0), p);
      double ux = -double(om) * y, uy = double(om) * x, uz = 0.0;
      double bx = Bu[0], by = Bu[1], bz = Bu[2];
      if (!uniform_B) {
        Scalar sx, sy, sz;
        dipole_B_impl(x, y, z, Scalar(0), Scalar(0), mz, sx, sy, sz);
        bx = sx; by = sy; bz = sz;
      }
      ex = -(uy * bz - uz * by);
      ey = -(uz * bx - ux * bz);
      ez = -(ux * by - uy * bx);
    };

    const int NQ = 200;
    for (int e = 0; e < lp.n_owned_he; e++) {
      gidx_t g = lp.h_edge_l2g[e];
      gidx_t v0, v1;
      h_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, bx, by, bz;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx, by, bz);
      double acc = 0;
      for (int i = 0; i < NQ; i++) {
        Scalar t = (i + Scalar(0.5)) / NQ;
        Scalar x, y, z, dlx, dly, dlz;
        h_edge_sphere_sample(r0, ax, ay, az, bx, by, bz, t, x, y, z, dlx,
                             dly, dlz);
        double ex, ey, ez;
        E_at(x, y, z, ex, ey, ez);
        acc += (ex * dlx + ey * dly + ez * dlz) / NQ;
      }
      Eb[e] = Scalar(acc);
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      gidx_t g = lp.v_edge_l2g[e];
      gidx_t v0, v1;
      v_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, a1x, a1y, a1z;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, a1x, a1y, a1z);
      (void)a1x; (void)a1y; (void)a1z;
      double dl[3] = {double(r1 - r0) * ax, double(r1 - r0) * ay,
                      double(r1 - r0) * az};
      double acc = 0;
      for (int i = 0; i < NQ; i++) {
        double t = (i + 0.5) / NQ;
        double rt = (1.0 - t) * r0 + t * r1;
        double ex, ey, ez;
        E_at(Scalar(rt * ax), Scalar(rt * ay), Scalar(rt * az), ex, ey, ez);
        acc += (ex * dl[0] + ey * dl[1] + ez * dl[2]) / NQ;
      }
      Eb[es + e] = Scalar(acc);
    }

    // Eeff exactly as the solver forms it -- no sign bookkeeping in the
    // test, so this stays valid whichever convention the code adopts.
    core.frame_drag_eff_E(Eb, Bb, B0b, W);
    for (int e = 0; e < ne; e++) {
      Ep[e] = W[e];                          // shipped Eeff
      Em[e] = Scalar(2) * Eb[e] - W[e];      // E - (Eeff - E): flipped sign
    }

    core.faraday(Eb, cE, -1.0);              // c* <- +d1(.)
    core.faraday(Ep, cP, -1.0);
    core.faraday(Em, cM, -1.0);

    auto rms = [&](buffer<Scalar>& c) {
      double s = 0;
      for (int f = 0; f < lp.n_owned_tri; f++) s += double(c[f]) * double(c[f]);
      for (int f = 0; f < lp.n_owned_rect; f++)
        s += double(c[bs + f]) * double(c[bs + f]);
      return std::sqrt(s / (lp.n_owned_tri + lp.n_owned_rect));
    };
    curl_E = rms(cE);
    curl_eff = rms(cP);
    curl_flip = rms(cM);
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

// ===========================================================================
// THE SIGN TEST.  Which sign of the frame-drag term admits the
// Muslimov-Tsygan equilibrium the scheme is built to produce?
//
// dec_solver_dist's header states the design goal: "The steady corotation
// state of a star spun at Omega with surface EMF (Omega - omega_lt) -- the
// Muslimov-Tsygan reduced-rho_GJ configuration -- is an exact equilibrium of
// this pair of modifications."  Stationarity means curl(E_eff) = 0.  With
// the plasma drifting at omega_eff(r) = Omega - omega_lt(r) and E = -(u x B)
// for an axisymmetric dipole, only ONE sign leaves a rigid rotation behind:
//
//     E - v_LT x B  =  -Omega (zhat x x) x B          -> curl = 0   OK
//     E + v_LT x B  =  -(Omega - 2 w_lt(r)) (...) x B -> curl != 0  NOT
//
// The far-field limit is the same statement: the '+' branch relaxes to
// Omega - 2 omega_lt(R*) at large r (0.6 Omega at the production
// compactness), not to Omega.
//
// This is the physics check the other four cases in this file cannot make:
// every one of them passes identically under v_LT -> -v_LT, because they
// validate the DISCRETIZATION of the term against its own analytic form.
// ===========================================================================
TEST_CASE("Frame-drag EMF: the MT state is stationary under the solver",
          "[prismatic][framedrag]") {
  const Scalar Omega = 0.25, omega_lt0 = 0.05;   // w_lt(R*)/Omega = 0.2
  const int p = 3;

  // MEASURED 2026-08-04 with the CURRENT sign (Eeff = E + v_LT x B),
  // rms d1 over owned faces, float storage, at L2 / L3:
  //
  //                     uniform axial B            aligned dipole B
  //   curl(E)           5.530e-4 / 2.809e-4        6.344e-4 / 3.222e-4
  //   curl(Eeff)        1.108e-3 / 5.620e-4        1.288e-3 / 6.489e-4
  //                     = 2.0031x / 2.0005x        = 2.0299x / 2.0137x
  //   curl(flipped)     3.557e-5 / 6.639e-6        1.794e-4 / 6.608e-5
  //                     = 0.064x / 0.024x (r 5.4)  = 0.283x / 0.205x (r 2.7)
  //
  // Eeff lands on EXACTLY twice the differential-rotation curl and stays
  // there under refinement -- a physical term, not truncation.  The
  // flipped convention cancels to a floor that CONVERGES, which is what
  // truncation does.  So the shipped sign makes the stationary state
  // omega(r) = Omega + w_lt(r): rotation ENHANCED near the star, the
  // opposite of the Muslimov-Tsygan reduction the scheme is built for.
  //
  // These assertions are on frame_drag_eff_E's OWN output, so they do not
  // encode a sign convention and stay meaningful after the fix.

  SECTION("uniform axial B (stencil-exact: the decisive case)") {
    fd_fixture f2(2), f3(3);
    double cE2, cEff2, cFl2, cE3, cEff3, cFl3;
    f2.mt_curl(Omega, omega_lt0, p, true, cE2, cEff2, cFl2);
    f3.mt_curl(Omega, omega_lt0, p, true, cE3, cEff3, cFl3);

    INFO("L2: curl(E) = " << cE2 << ", curl(Eeff) = " << cEff2 << " ("
         << cEff2 / cE2 << "x), flipped = " << cFl2 << " ("
         << cFl2 / cE2 << "x)");
    INFO("L3: curl(E) = " << cE3 << ", curl(Eeff) = " << cEff3 << " ("
         << cEff3 / cE3 << "x), flipped = " << cFl3 << " ("
         << cFl3 / cE3 << "x)");
    INFO("If curl(Eeff) ~ 2x and flipped ~ 0, the frame-drag term has the "
         "WRONG SIGN: swap it in frame_drag_eff_E / frame_drag_velocity.");

    // Guard against a trivially-zero comparison: the differential-rotation
    // curl of the MT state must be real and resolved.
    REQUIRE(cE2 > 1e-6);
    REQUIRE(cE3 > 1e-6);

    // THE INVARIANT.  With the plasma drifting at omega_eff(r) =
    // Omega - w_lt(r), the solver's own Eeff must be curl-free: that is
    // precisely the claim in dec_solver_dist.h's header.  What may remain
    // is the W stencil floor, so this is asserted against curl(E) and
    // required to CONVERGE (truncation shrinks; a physical term does not).
    REQUIRE(cEff2 < 0.10 * cE2);
    REQUIRE(cEff3 < 0.04 * cE3);
    REQUIRE(cEff2 / cEff3 > 3.0);
  }

  SECTION("aligned dipole B (physical field, looser stencil floor)") {
    fd_fixture f2(2), f3(3);
    double cE2, cEff2, cFl2, cE3, cEff3, cFl3;
    f2.mt_curl(Omega, omega_lt0, p, false, cE2, cEff2, cFl2);
    f3.mt_curl(Omega, omega_lt0, p, false, cE3, cEff3, cFl3);

    INFO("L2: curl(E) = " << cE2 << ", curl(Eeff) = " << cEff2 << " ("
         << cEff2 / cE2 << "x), flipped = " << cFl2 << " ("
         << cFl2 / cE2 << "x)");
    INFO("L3: curl(E) = " << cE3 << ", curl(Eeff) = " << cEff3 << " ("
         << cEff3 / cE3 << "x), flipped = " << cFl3 << " ("
         << cFl3 / cE3 << "x)");

    // Same invariant on the real field.  W is built exact for UNIFORM B
    // and the dipole varies as r^-3 across a cell, so the achievable floor
    // is ~20-30% rather than a few percent -- hence the looser bound, with
    // convergence still carrying the burden of proof.
    REQUIRE(cE2 > 1e-6);
    REQUIRE(cEff2 < 0.35 * cE2);
    REQUIRE(cEff3 / cE3 < cEff2 / cE2);   // relative residual must shrink
    REQUIRE(cEff2 / cEff3 > 2.0);
  }
}

// ===========================================================================
// THE NO-OP INVARIANT, particle side.
//
// Every flat-space result in the repo -- the L6 inclination scan, the
// convergence studies, the partition- and dt-invariance guarantees -- was
// produced by a pusher without any metric terms.  Turning the GR machinery
// ON at ZERO strength must therefore reproduce the flat push BIT FOR BIT,
// not merely to round-off: alpha = 1 and v_LT = 0 are exact, and every GR
// expression in gca_push / the Boris branch is written so that 1.0*x == x
// and x + 0.0 == x collapse it to the original token sequence.
//
// This is the test that catches a "harmless tidy-up" of those expressions
// (factoring out an alpha, grouping each trapezoid endpoint together)
// silently perturbing every flat run at the last bit.
// ===========================================================================
TEST_CASE("Lapse + shift: zero-strength GR is a bitwise no-op in the pusher",
          "[prismatic][framedrag][pusher]") {
  gr_metric_params off;                 // enabled = false
  gr_metric_params zero;                // machinery ON, strength ZERO
  zero.enabled = true;
  zero.omega_lt0 = Scalar(0);
  zero.compactness = Scalar(0);
  zero.r_star = Scalar(1);
  zero.lt_p = 3;

  // The metric evaluator itself must return exact identity values.
  for (double x : {0.3, 1.0, 2.7}) {
    for (double z : {-1.1, 0.0, 2.0}) {
      Scalar a, vx, vy, vz;
      gr_metric_at(zero, Scalar(x), Scalar(0.7), Scalar(z), a, vx, vy, vz);
      REQUIRE(a == Scalar(1));
      REQUIRE(vx == Scalar(0));
      REQUIRE(vy == Scalar(0));
      REQUIRE(vz == Scalar(0));
      Scalar a2, wx, wy, wz;
      gr_metric_at(off, Scalar(x), Scalar(0.7), Scalar(z), a2, wx, wy, wz);
      REQUIRE(a2 == a);
    }
  }

  // A nonzero compactness must NOT be identity -- guards against the test
  // above passing because gr_lapse always returns 1.
  REQUIRE(gr_lapse(Scalar(1.0), Scalar(0.5), Scalar(1.0)) <
          Scalar(0.71));
  REQUIRE(gr_lapse(Scalar(1.0), Scalar(0.5), Scalar(1.0)) >
          Scalar(0.70));
  // ... and a nonzero drag must produce a nonzero v_LT with the SHIFT
  // opposite to it (the one place the convention is decided).
  Scalar vx, vy, vz, bx, by, bz;
  frame_drag_velocity(Scalar(1), Scalar(0), Scalar(0), Scalar(0.05),
                      Scalar(1), 3, vx, vy, vz);
  frame_drag_shift(Scalar(1), Scalar(0), Scalar(0), Scalar(0.05),
                   Scalar(1), 3, bx, by, bz);
  REQUIRE(vy == Scalar(0.05));      // v_LT = omega zhat x xhat = +omega yhat
  REQUIRE(bx == -vx);
  REQUIRE(by == -vy);
  REQUIRE(bz == -vz);
}

// ===========================================================================
// THE BC <-> PUSHER CONSISTENCY TEST.
//
// The inner BC prescribes the FIDO electric field on the stellar surface.
// The pusher transports particles with dx/dt = alpha v - beta.  These are
// two halves of one statement and nothing previously compared them: the
// lapse was added to the solver and the pusher while the BC kept its
// flat-space form, which made the imposed surface EMF 29% too small at
// compactness 0.5 and diverged job 5162728 (see that run's ABORTED.md).
//
// The invariant, stated so it cannot drift: take the BC's E, read off the
// ExB drift it implies, push that through the pusher's OWN transport law,
// and the star's surface must come back rotating rigidly at exactly Omega
// -- because that is what "the star spins at Omega" means in coordinate
// terms.  Any lapse or shift factor missing from either side breaks it.
//
//   BC:       E = -(om_bc zhat x r) x B
//   drift:    v_FIDO = E x B / B^2 = om_bc (zhat x r)
//   pusher:   dx/dt  = alpha v_FIDO + v_LT
//   REQUIRE:  dx/dt  = Omega (zhat x r)
//
// This drives the REAL apply_inner_bc kernel, so it covers the quadrature
// and edge orientations too, not just the algebra.
// ===========================================================================
TEST_CASE("Inner BC and pusher transport agree on rigid corotation",
          "[prismatic][framedrag][pusher]") {
  const Scalar Omega = 0.25, Bp = 1.0, r_star = 1.0;
  const int p = 3;

  // (compactness, omega_lt(R*)/Omega): flat, shift-only, and the
  // production GR setting where the lapse actually bites.
  struct cfg { double C, ltf; const char* name; };
  const cfg cfgs[] = {{0.0, 0.0, "flat"},
                      {0.0, 0.2, "shift-only (alpha == 1)"},
                      {0.5, 0.2, "shift + lapse (production)"}};

  for (const auto& c : cfgs) {
    fd_fixture fx(2);
    const Scalar wlt0 = Scalar(c.ltf * Omega);
    const Scalar comp = Scalar(c.C);

    dec_inner_bc_params par;
    par.Bp = Bp;
    par.Omega = Omega;
    par.obliquity = 0;
    par.use_deutsch = false;
    par.overwrite_b = false;      // leave B alone; only E is under test
    par.omega_lt0 = wlt0;
    par.lt_r_star = r_star;
    par.lt_p = p;
    par.lapse_compactness = comp;

    const int ne = fx.core.n_edges_local(), nf = fx.core.n_faces_local();
    buffer<Scalar> E, B, B0;
    E.set_memtype(MemType::host_only); E.resize(ne); E.assign(0);
    for (auto* b : {&B, &B0}) {
      b->set_memtype(MemType::host_only); b->resize(nf); b->assign(0);
    }
    fx.core.apply_inner_bc(E, B, B0, par, 0.0, 0.0);

    auto lp = fx.core.get_lp(exec_tags::host{});
    auto mp = fx.mesh.host_ptrs();

    // The BC wrote the circulation of -(om_bc zhat x r) x B_dipole, which
    // is LINEAR in om_bc.  So build the same circulation at om = 1 and
    // scale it by the rate the PUSHER's transport law demands:
    //
    //   alpha * om_fido + omega_lt(r) = Omega   =>   om_fido =
    //       (Omega - omega_lt(r)) / alpha
    //
    // with alpha and omega_lt read from gr_metric_at -- the pusher's own
    // helper, not the BC's expression -- then compare cochain to cochain.
    //
    // Compared against the MAX circulation, not per-edge: edges nearly
    // perpendicular to the corotation direction carry a circulation near
    // zero, and E is stored as float, so a per-edge relative comparison
    // there is dominated by storage noise rather than by the physics.
    int n_checked = 0;
    double worst = 0.0, scale = 0.0;
    const int NQ = 64;
    for (int e = 0; e < lp.n_owned_he; e++) {
      if (lp.h_edge_boundary[e] != 1) continue;
      gidx_t g = lp.h_edge_l2g[e];
      gidx_t v0, v1;
      h_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, bx, by, bz;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx, by, bz);

      double unit_circ = 0;   // circulation at om = 1
      for (int i = 0; i < NQ; i++) {
        Scalar t = (i + Scalar(0.5)) / NQ;
        Scalar x, y, z, dlx, dly, dlz;
        h_edge_sphere_sample(r0, ax, ay, az, bx, by, bz, t, x, y, z,
                             dlx, dly, dlz);
        Scalar Bx, By, Bz;
        dipole_B_impl(x, y, z, Scalar(0), Scalar(0), Bp, Bx, By, Bz);
        // -(u x B) with u = 1 * (zhat x x)
        double ux = -double(y), uy = double(x);
        double ex = -(uy * Bz), ey = -(-ux * Bz),
               ez = -(ux * By - uy * Bx);
        unit_circ += (ex * dlx + ey * dly + ez * dlz) / NQ;
      }
      // The FIDO rate that makes the COORDINATE motion rigid at Omega,
      // from the pusher's transport law.
      gr_metric_params grp;
      grp.enabled = true;
      grp.omega_lt0 = wlt0;
      grp.r_star = r_star;
      grp.lt_p = p;
      grp.compactness = comp;
      Scalar alpha, vx, vy, vz;
      // On the x axis at this shell v_LT = omega_lt * r * yhat, so the
      // angular rate reads straight off the y component.
      gr_metric_at(grp, r0, Scalar(0), Scalar(0), alpha, vx, vy, vz);
      const double om_fido =
          (double(Omega) - double(vy) / double(r0)) / double(alpha);

      const double expect = om_fido * unit_circ;
      worst = std::max(worst, std::abs(double(E[e]) - expect));
      scale = std::max(scale, std::abs(expect));
      n_checked++;
    }
    INFO(c.name << ": checked " << n_checked << " boundary h-edges, worst |E "
                << "- E_required| = " << worst << " against scale " << scale
                << " (" << worst / scale << ")");
    REQUIRE(n_checked > 0);
    REQUIRE(scale > 1e-6);
    // Quadrature (the BC uses 5-point Gauss, the reference 64-point
    // midpoint) plus float storage set the floor.  The defect this guards
    // against is 29%, not 1e-3.
    REQUIRE(worst < 1e-3 * scale);
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

// ===========================================================================
// Partition invariance: the frame-drag weights and the resulting effective
// circulation must be independent of how the mesh is decomposed.
//
// This is the check that matters before a distributed production run.  The
// MPI exchange of Eeff itself is the generic exchange_edge path already
// validated for E; what is NEW and untested is the per-rank WEIGHT
// CONSTRUCTION on a partitioned mesh -- ghost-face indexing through the
// d1t blocks, l2g mapping, and owned/ghost boundaries.  Those are all
// exercised in-process here (no MPI needed): every rank of an A x K
// decomposition must reproduce the single-rank answer on its owned edges.
// ===========================================================================
TEST_CASE("Frame-drag EMF: weights are partition-invariant",
          "[prismatic][framedrag]") {
  constexpr int L = 2, N_r = 8;
  const Scalar omega0 = 0.25;
  const int p = 3;

  prismatic_mesh mesh;
  mesh.build(L, N_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  // ---- Reference: single-rank (identity layouts) ----
  auto part_g = prismatic_partition::single_rank(L, N_r);
  part_g.set_topology(&topo);
  auto mp_g = prismatic_mesh_partition::build(part_g, topo);
  core_t core_g;
  core_g.build(mesh, mp_g);
  core_g.build_frame_drag(omega0, Scalar(1.0), p);

  // Global field arrays: smooth but arbitrary (exercises all stencil
  // entries, unlike a uniform field which the scheme is exact on).
  std::vector<Scalar> Eg(mesh.m_N_edges), Bg(mesh.m_N_faces), B0g(mesh.m_N_faces);
  for (int e = 0; e < mesh.m_N_edges; e++)
    Eg[e] = std::sin(Scalar(0.013) * e) + Scalar(0.21);
  for (int f = 0; f < mesh.m_N_faces; f++) {
    Bg[f] = std::cos(Scalar(0.007) * f) - Scalar(0.13);
    B0g[f] = Scalar(0.4) * std::sin(Scalar(0.004) * f);
  }

  auto run_core = [&](core_t& c, buffer<Scalar>& W) {
    const int ne = c.n_edges_local(), nf = c.n_faces_local();
    buffer<Scalar> E, B, B0;
    for (auto* b : {&E, &W}) { b->set_memtype(MemType::host_only); b->resize(ne); }
    for (auto* b : {&B, &B0}) { b->set_memtype(MemType::host_only); b->resize(nf); }
    c.edge_from_global(Eg.data(), E);
    c.face_from_global(Bg.data(), B);
    c.face_from_global(B0g.data(), B0);
    c.frame_drag_eff_E(E, B, B0, W);
  };

  buffer<Scalar> Wg;
  run_core(core_g, Wg);
  auto lp_g = core_g.get_lp(exec_tags::host{});
  const int es_g = core_g.e_split();

  // ---- Every rank of a 4 x 2 decomposition must match on owned edges ----
  const int A = 4, K = 2;
  double max_diff = 0.0;
  int n_checked = 0;
  for (int rank = 0; rank < A * K; rank++) {
    auto part_l = prismatic_partition::combined(L, N_r, A, K, rank);
    part_l.set_topology(&topo);
    auto mp_l = prismatic_mesh_partition::build(part_l, topo);
    core_t core_l;
    core_l.build(mesh, mp_l);
    core_l.build_frame_drag(omega0, Scalar(1.0), p);

    buffer<Scalar> Wl;
    run_core(core_l, Wl);
    auto lp_l = core_l.get_lp(exec_tags::host{});
    const int es_l = core_l.e_split();

    for (int e = 0; e < lp_l.n_owned_he; e++) {
      const gidx_t g = lp_l.h_edge_l2g[e];
      max_diff = std::max(max_diff,
                          std::abs(double(Wl[e]) - double(Wg[int(g)])));
      n_checked++;
    }
    for (int e = 0; e < lp_l.n_owned_ve; e++) {
      const gidx_t g = lp_l.v_edge_l2g[e];
      max_diff = std::max(
          max_diff, std::abs(double(Wl[es_l + e]) - double(Wg[es_g + int(g)])));
      n_checked++;
    }
  }
  // Every owned edge of the decomposition must have been visited exactly
  // once in aggregate (no gaps, no double coverage of owned rows).
  INFO("edges checked: " << n_checked << ", max |W_local - W_global| = "
                         << max_diff);
  REQUIRE(n_checked == lp_g.n_owned_he + lp_g.n_owned_ve);
  // Weight construction is per-edge deterministic; the only slack is
  // float summation order in the Gram/quadrature, so this is tight.
  REQUIRE(max_diff < 1e-5);
}

// ===========================================================================
// Adjoint pairing of the frame-drag coupling (the beta x E Ampere term).
//
// The one-sided Faraday W coupling is a grid-scale numerical instability:
// sym(h2 d1 W) has O(1) eigenvalues localized in the first shells above
// the star (measured +-1.4 at L3), growing in VACUUM at gamma ~ 0.04 (L4)
// to ~0.2 (L6), sign-independent -- the near-surface tangential-E layer
// seen in every fake-GR production run.  The cure is the exact energy
// transpose: H_aux = h2 B + F with F = W^T h1inv^-1 E, which makes
// U_full = 1/2 E'h1inv^-1 E + 1/2 B'h2 B + E'h1inv^-1 W B an exact
// semi-discrete invariant, provided BOTH couplings are midpoint-centred
// under leapfrog.  These cases pin (a) the transpose construction,
// (b) the actual stability of the centred update, (c) partition
// invariance of the distributed partial-transpose assembly.
// ===========================================================================

TEST_CASE("Frame-drag adjoint: F is the exact energy transpose of W",
          "[prismatic][framedrag][adjoint]") {
  fd_fixture fx(3);
  fx.core.build_frame_drag(Scalar(0.25), Scalar(1.0), 3);
  auto lp = fx.core.get_lp(exec_tags::host{});
  const int ne = fx.core.n_edges_local(), nf = fx.core.n_faces_local();
  const int es = fx.core.e_split(), bs = fx.core.b_split();

  buffer<Scalar> E, Ez, B, B0, WB, F;
  for (auto* b : {&E, &Ez, &WB}) { b->set_memtype(MemType::host_only); b->resize(ne); }
  for (auto* b : {&B, &B0, &F}) { b->set_memtype(MemType::host_only); b->resize(nf); }
  for (int e = 0; e < ne; e++) E[e] = std::sin(Scalar(0.013) * e) + Scalar(0.21);
  for (int f = 0; f < nf; f++) B[f] = std::cos(Scalar(0.007) * f) - Scalar(0.13);
  Ez.assign(0);
  B0.assign(0);

  // (W B)_e from the shipped operator with E = 0.
  fx.core.frame_drag_eff_E(Ez, B, B0, WB);
  // F from the shipped transpose.
  fx.core.frame_drag_aux_F(E, F);

  double lhs = 0.0, rhs = 0.0, scale = 0.0;
  for (int e = 0; e < lp.n_owned_he; e++) {
    lhs += double(E[e]) * double(WB[e]) / double(lp.h_edge_hodge1_inv[e]);
    scale += std::abs(double(E[e]) * double(WB[e]) /
                      double(lp.h_edge_hodge1_inv[e]));
  }
  for (int e = 0; e < lp.n_owned_ve; e++) {
    lhs += double(E[es + e]) * double(WB[es + e]) /
           double(lp.v_edge_hodge1_inv[e]);
    scale += std::abs(double(E[es + e]) * double(WB[es + e]) /
                      double(lp.v_edge_hodge1_inv[e]));
  }
  for (int f = 0; f < nf; f++) rhs += double(F[f]) * double(B[f]);
  INFO("E.h1inv^-1.(W B) = " << lhs << ", F.B = " << rhs
                             << ", |terms| = " << scale);
  REQUIRE(std::abs(lhs - rhs) < 1e-5 * scale);
}

TEST_CASE("Frame-drag adjoint: the centred pair is stable where the "
          "one-sided coupling grows",
          "[prismatic][framedrag][adjoint]") {
  // Vacuum leapfrog from noise, no BC, no damping, J = 0.  The drag is
  // set strong (w0 = 0.4, still subluminal) so the one-sided instability
  // is fast at this small size; the drift ratio is the assertion, so the
  // case does not depend on absolute rate calibration.
  // Drag strength and dt are chosen so BOTH arms sit in their asymptotic
  // regimes: w0 = 0.2 (4x production) keeps the one-sided growth fast but
  // finite in float over the run, and dt = 0.15 h keeps the 2-sweep
  // Picard midpoint well inside its convergence radius (lambda dt / 2
  // ~ 0.1 here vs ~3e-3 at production parameters).
  constexpr int L = 2, N_r = 8;
  const double dt = 0.15 * std::sqrt(4.0 * M_PI / 162.0);
  const int n_steps = 4000;

  auto run = [&](bool adjoint) -> double {
    fd_fixture fx(L);
    fx.core.build_frame_drag(Scalar(0.2), Scalar(1.0), 3);
    auto lp = fx.core.get_lp(exec_tags::host{});
    const int ne = fx.core.n_edges_local(), nf = fx.core.n_faces_local();
    const int es = fx.core.e_split(), bs = fx.core.b_split();

    buffer<Scalar> E, Eeff, Eold, Emid, B, B0, Bold, Bmid, F, J;
    for (auto* b : {&E, &Eeff, &Eold, &Emid, &J}) {
      b->set_memtype(MemType::host_only); b->resize(ne);
    }
    for (auto* b : {&B, &B0, &Bold, &Bmid, &F}) {
      b->set_memtype(MemType::host_only); b->resize(nf);
    }
    J.assign(0);
    B0.assign(0);
    for (int e = 0; e < ne; e++)
      E[e] = Scalar(1e-3) * std::sin(Scalar(0.917) * e + Scalar(0.3));
    for (int f = 0; f < nf; f++)
      B[f] = Scalar(1e-3) * std::cos(Scalar(1.331) * f);

    auto energy = [&]() -> double {
      double U = 0;
      for (int e = 0; e < lp.n_owned_he; e++)
        U += 0.5 * double(E[e]) * double(E[e]) /
             double(lp.h_edge_hodge1_inv[e]);
      for (int e = 0; e < lp.n_owned_ve; e++)
        U += 0.5 * double(E[es + e]) * double(E[es + e]) /
             double(lp.v_edge_hodge1_inv[e]);
      for (int f = 0; f < lp.n_owned_tri; f++)
        U += 0.5 * double(lp.tri_face_hodge2[f]) * double(B[f]) * double(B[f]);
      for (int f = 0; f < lp.n_owned_rect; f++)
        U += 0.5 * double(lp.rect_face_hodge2[f]) * double(B[bs + f]) *
             double(B[bs + f]);
      return U;
    };
    const double U0 = energy();

    for (int s = 0; s < n_steps; s++) {
      if (adjoint) {
        for (int f = 0; f < nf; f++) Bold[f] = B[f];
        fx.core.frame_drag_eff_E(E, B, B0, Eeff);
        fx.core.faraday(Eeff, B, dt);
        for (int f = 0; f < nf; f++)
          Bmid[f] = Scalar(0.5) * (Bold[f] + B[f]);
        fx.core.frame_drag_eff_E(E, Bmid, B0, Eeff);
        for (int f = 0; f < nf; f++) B[f] = Bold[f];
        fx.core.faraday(Eeff, B, dt);
        for (int e = 0; e < ne; e++) Eold[e] = E[e];
        fx.core.frame_drag_aux_F(E, F);
        fx.core.ampere_fd(E, B, F, J, dt);
        for (int e = 0; e < ne; e++)
          Emid[e] = Scalar(0.5) * (Eold[e] + E[e]);
        fx.core.frame_drag_aux_F(Emid, F);
        for (int e = 0; e < ne; e++) E[e] = Eold[e];
        fx.core.ampere_fd(E, B, F, J, dt);
      } else {
        fx.core.frame_drag_eff_E(E, B, B0, Eeff);
        fx.core.faraday(Eeff, B, dt);
        fx.core.ampere(E, B, J, dt);
      }
    }
    return energy() / U0;
  };

  const double growth_onesided = run(false);
  const double growth_adjoint = run(true);
  INFO("U(T)/U(0): one-sided = " << growth_onesided
                                 << ", adjoint centred = " << growth_adjoint);
  // The one-sided coupling must exhibit its instability at this drag
  // strength (an overflow to NaN counts -- that IS the instability), and
  // the centred adjoint pair must hold energy to leapfrog wobble plus the
  // Picard-2 residual at this exaggerated coupling.
  REQUIRE((std::isnan(growth_onesided) || growth_onesided > 5.0));
  REQUIRE(growth_adjoint < 1.25);
}

TEST_CASE("Frame-drag adjoint: partial-transpose F assembly is "
          "partition-invariant",
          "[prismatic][framedrag][adjoint]") {
  constexpr int L = 2, N_r = 8;
  const Scalar omega0 = 0.25;
  const int p = 3;

  prismatic_mesh mesh;
  mesh.build(L, N_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  std::vector<Scalar> Eg(mesh.m_N_edges);
  for (int e = 0; e < mesh.m_N_edges; e++)
    Eg[e] = std::sin(Scalar(0.013) * e) + Scalar(0.21);

  // ---- Reference: single-rank F (complete: every edge owned) ----
  auto part_g = prismatic_partition::single_rank(L, N_r);
  part_g.set_topology(&topo);
  auto mp_g = prismatic_mesh_partition::build(part_g, topo);
  core_t core_g;
  core_g.build(mesh, mp_g);
  core_g.build_frame_drag(omega0, Scalar(1.0), p);
  buffer<Scalar> E_ref, F_ref;
  E_ref.set_memtype(MemType::host_only);
  E_ref.resize(core_g.n_edges_local());
  F_ref.set_memtype(MemType::host_only);
  F_ref.resize(core_g.n_faces_local());
  core_g.edge_from_global(Eg.data(), E_ref);
  core_g.frame_drag_aux_F(E_ref, F_ref);
  auto lp_g = core_g.get_lp(exec_tags::host{});
  const int bs_g = core_g.b_split();

  // ---- 4 x 2 decomposition: per-rank PARTIAL F, summed globally via
  // face l2g -- exactly what reduce_face computes across ranks ----
  std::vector<double> F_sum(mesh.m_N_faces, 0.0);
  const int A = 4, K = 2;
  for (int rank = 0; rank < A * K; rank++) {
    auto part_l = prismatic_partition::combined(L, N_r, A, K, rank);
    part_l.set_topology(&topo);
    auto mp_l = prismatic_mesh_partition::build(part_l, topo);
    core_t core_l;
    core_l.build(mesh, mp_l);
    core_l.build_frame_drag(omega0, Scalar(1.0), p);
    buffer<Scalar> E_l, F_l;
    E_l.set_memtype(MemType::host_only);
    E_l.resize(core_l.n_edges_local());
    F_l.set_memtype(MemType::host_only);
    F_l.resize(core_l.n_faces_local());
    core_l.edge_from_global(Eg.data(), E_l);
    core_l.frame_drag_aux_F(E_l, F_l);
    auto lp_l = core_l.get_lp(exec_tags::host{});
    const int bs_l = core_l.b_split();
    // All LOCAL faces carry contributions from this rank's owned edges.
    for (int f = 0; f < lp_l.n_local_tri; f++)
      F_sum[size_t(lp_l.tri_face_l2g[f])] += double(F_l[f]);
    for (int f = 0; f < lp_l.n_local_rect; f++)
      F_sum[size_t(mesh.m_N_r + 1) * size_t(mesh.m_N_tri) + size_t(lp_l.rect_face_l2g[f])] +=
          double(F_l[bs_l + f]);
  }

  double max_diff = 0.0, scale = 0.0;
  for (int f = 0; f < lp_g.n_owned_tri; f++) {
    max_diff = std::max(
        max_diff,
        std::abs(F_sum[size_t(lp_g.tri_face_l2g[f])] - double(F_ref[f])));
    scale = std::max(scale, std::abs(double(F_ref[f])));
  }
  for (int f = 0; f < lp_g.n_owned_rect; f++) {
    max_diff = std::max(
        max_diff,
        std::abs(F_sum[size_t(mesh.m_N_r + 1) * size_t(mesh.m_N_tri) +
                       size_t(lp_g.rect_face_l2g[f])] -
                 double(F_ref[bs_g + f])));
    scale = std::max(scale, std::abs(double(F_ref[bs_g + f])));
  }
  INFO("max |sum_ranks F_partial - F_single| = " << max_diff
       << " (scale " << scale << ")");
  REQUIRE(max_diff < 1e-5 * scale);
}
