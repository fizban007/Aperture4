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
// Stellar multipole extension of the inner boundary field (shifted
// dipole, quadrupole, shifted quadrupole).
//
// The contract under test is twofold:
//   1. THE DEFAULT IS THE OLD CODE.  With all-zero extras the stellar
//      evaluator and the cochain fill must be BITWISE identical to the
//      centered-dipole path — asserted with exact equality, not a
//      tolerance.
//   2. The new terms are the physics they claim: the quadrupole is
//      B = -grad[(1/2) x·Q·x / r^5] and divergence-free (checked by
//      finite differences), the body->lab rotation satisfies
//      B(x; phase) = Rz(phase) B0(Rz(-phase) x) (checks the tensor
//      rotation algebra), and the BC's B overwrite at t = 0 reproduces
//      the IC fill on boundary faces exactly (IC/BC phase-convention
//      consistency — the mismatch class that would pump a boundary
//      layer in production).
// Tolerances scale with Scalar epsilon so the suite passes under both
// fp32 and fp64 builds.
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
#include <limits>
#include <vector>

using namespace Aperture;

namespace {

using policy = prismatic_exec_policy_host;
using core_t = dec_solver_dist<policy>;

constexpr double EPS = std::numeric_limits<Scalar>::epsilon();

// A generic non-axisymmetric test configuration: oblique dipole shifted
// off-axis plus a shifted quadrupole with all five components populated.
stellar_extras test_extras() {
  stellar_extras e;
  e.dip_off[0] = Scalar(0.15);
  e.dip_off[1] = Scalar(-0.1);
  e.dip_off[2] = Scalar(0.2);
  e.quad_Q[0] = Scalar(0.7);    // Qxx
  e.quad_Q[1] = Scalar(-0.3);   // Qxy
  e.quad_Q[2] = Scalar(0.45);   // Qxz
  e.quad_Q[3] = Scalar(-0.25);  // Qyy
  e.quad_Q[4] = Scalar(0.6);    // Qyz
  e.quad_off[0] = Scalar(-0.05);
  e.quad_off[1] = Scalar(0.12);
  e.quad_off[2] = Scalar(-0.18);
  return e;
}

void rot_z(double phi, double x, double y, double z, double& rx, double& ry,
           double& rz) {
  double c = std::cos(phi), s = std::sin(phi);
  rx = c * x - s * y;
  ry = s * x + c * y;
  rz = z;
}

}  // namespace

TEST_CASE("stellar_B_impl: default extras are bitwise the centered dipole",
          "[prismatic][stellar_multipole]") {
  const Scalar Bp = Scalar(37.5), obl = Scalar(0.4);
  const double phases[] = {0.0, 0.37, 2.9, -1.2};
  // Include exact zeros and negative zeros among the coordinates: the
  // fallback guarantee hinges on zero offsets never flipping the sign of
  // a zero coordinate.
  const double pts[][3] = {{1.3, 0.2, -0.7},  {-1.1, -0.9, 0.4},
                           {0.0, 1.5, -0.3},  {-0.0, -1.5, 0.3},
                           {1.9, 0.0, 0.0},   {0.6, -0.0, 1.1}};
  stellar_extras none;  // all zero
  for (double ph : phases) {
    stellar_moments mom = stellar_moments_at(Bp, obl, none, ph);
    // The rotated moment must equal the legacy expressions bitwise.
    Scalar mx_ref = Bp * std::sin(obl) * std::cos(ph);
    Scalar my_ref = Bp * std::sin(obl) * std::sin(ph);
    Scalar mz_ref = Bp * std::cos(obl);
    REQUIRE(mom.mx == mx_ref);
    REQUIRE(mom.my == my_ref);
    REQUIRE(mom.mz == mz_ref);
    REQUIRE_FALSE(mom.has_quad);
    for (auto& p : pts) {
      Scalar x = Scalar(p[0]), y = Scalar(p[1]), z = Scalar(p[2]);
      Scalar bx1, by1, bz1, bx2, by2, bz2;
      dipole_B_impl(x, y, z, mom.mx, mom.my, mom.mz, bx1, by1, bz1);
      stellar_B_impl(x, y, z, mom, bx2, by2, bz2);
      REQUIRE(bx1 == bx2);
      REQUIRE(by1 == by2);
      REQUIRE(bz1 == bz2);
    }
  }
}

TEST_CASE("quadrupole_B_impl: B = -grad Phi and div B = 0",
          "[prismatic][stellar_multipole]") {
  const Scalar Qxx = Scalar(0.7), Qxy = Scalar(-0.3), Qxz = Scalar(0.45);
  const Scalar Qyy = Scalar(-0.25), Qyz = Scalar(0.6);
  const Scalar Qzz = -(Qxx + Qyy);

  auto phi = [&](Scalar x, Scalar y, Scalar z) -> Scalar {
    Scalar r2 = x * x + y * y + z * z;
    Scalar r = std::sqrt(r2);
    Scalar r5 = r2 * r2 * r;
    Scalar Qx = Qxx * x + Qxy * y + Qxz * z;
    Scalar Qy = Qxy * x + Qyy * y + Qyz * z;
    Scalar Qz = Qxz * x + Qyz * y + Qzz * z;
    return Scalar(0.5) * (x * Qx + y * Qy + z * Qz) / r5;
  };

  const double pts[][3] = {{1.2, 0.3, -0.5}, {-0.8, 1.1, 0.9},
                           {0.4, -1.3, -1.0}, {2.0, 0.1, 0.6}};
  // Central differences: error O(h^2) + O(eps/h); h = cbrt(eps)*scale
  // balances them at ~eps^(2/3).
  const Scalar h = Scalar(std::cbrt(EPS));
  const double tol = 100.0 * std::pow(EPS, 2.0 / 3.0);

  for (auto& p : pts) {
    Scalar x = Scalar(p[0]), y = Scalar(p[1]), z = Scalar(p[2]);
    Scalar bx, by, bz;
    quadrupole_B_impl(x, y, z, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, bx, by, bz);

    Scalar gx = (phi(x + h, y, z) - phi(x - h, y, z)) / (2 * h);
    Scalar gy = (phi(x, y + h, z) - phi(x, y - h, z)) / (2 * h);
    Scalar gz = (phi(x, y, z + h) - phi(x, y, z - h)) / (2 * h);
    double scale = std::max({std::abs(double(bx)), std::abs(double(by)),
                             std::abs(double(bz)), 1e-30});
    CHECK(std::abs(double(bx + gx)) / scale < tol);
    CHECK(std::abs(double(by + gy)) / scale < tol);
    CHECK(std::abs(double(bz + gz)) / scale < tol);

    // FD divergence of B itself.
    Scalar bxp, byp, bzp, bxm, bym, bzm, t1, t2;
    quadrupole_B_impl(x + h, y, z, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, bxp, t1, t2);
    quadrupole_B_impl(x - h, y, z, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, bxm, t1, t2);
    quadrupole_B_impl(x, y + h, z, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, t1, byp, t2);
    quadrupole_B_impl(x, y - h, z, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, t1, bym, t2);
    quadrupole_B_impl(x, y, z + h, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, t1, t2, bzp);
    quadrupole_B_impl(x, y, z - h, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, t1, t2, bzm);
    double div = (double(bxp) - double(bxm) + double(byp) - double(bym) +
                  double(bzp) - double(bzm)) /
                 (2.0 * double(h));
    CHECK(std::abs(div) / scale < tol);
  }
}

TEST_CASE("stellar_moments_at: lab field is the rigidly rotated body field",
          "[prismatic][stellar_multipole]") {
  const Scalar Bp = Scalar(12.0), obl = Scalar(0.6);
  stellar_extras e = test_extras();
  stellar_moments mom0 = stellar_moments_at(Bp, obl, e, 0.0);

  const double phases[] = {0.41, 1.7, -2.3, 3.05};
  const double pts[][3] = {{1.4, -0.3, 0.8}, {-0.9, 1.2, -0.6},
                           {0.5, 0.7, 1.5}};
  const double tol = 2e3 * EPS;

  for (double ph : phases) {
    stellar_moments mom = stellar_moments_at(Bp, obl, e, ph);
    REQUIRE(mom.has_quad);
    for (auto& p : pts) {
      // Lab-frame evaluation at x ...
      Scalar bx, by, bz;
      stellar_B_impl(Scalar(p[0]), Scalar(p[1]), Scalar(p[2]), mom, bx, by,
                     bz);
      // ... must equal the phase-0 field at the back-rotated point,
      // rotated forward.
      double xb, yb, zb;
      rot_z(-ph, p[0], p[1], p[2], xb, yb, zb);
      Scalar b0x, b0y, b0z;
      stellar_B_impl(Scalar(xb), Scalar(yb), Scalar(zb), mom0, b0x, b0y, b0z);
      double rbx, rby, rbz;
      rot_z(ph, double(b0x), double(b0y), double(b0z), rbx, rby, rbz);
      double scale = std::max({std::abs(rbx), std::abs(rby), std::abs(rbz)});
      CHECK(std::abs(double(bx) - rbx) / scale < tol);
      CHECK(std::abs(double(by) - rby) / scale < tol);
      CHECK(std::abs(double(bz) - rbz) / scale < tol);
    }
  }
}

// ===========================================================================
// Cochain-level tests on a real (single-rank) mesh.
// ===========================================================================

namespace {

struct mesh_fixture {
  prismatic_mesh mesh;
  icosphere_topology topo;
  prismatic_partition part;
  prismatic_mesh_partition mp_b;
  core_t core;

  explicit mesh_fixture(int L, int N_r)
      : part(prismatic_partition::single_rank(L, N_r)) {
    mesh.build(L, N_r, 1.0, 2.0);
    topo = icosphere_topology::build_from_mesh(mesh);
    part.set_topology(&topo);
    mp_b = prismatic_mesh_partition::build(part, topo);
    core.build(mesh, mp_b);
  }
};

}  // namespace

TEST_CASE("fill_stellar_B: closed-shell flux vanishes for the multipole",
          "[prismatic][stellar_multipole]") {
  mesh_fixture fx(3, 8);
  auto lp = fx.core.get_lp(exec_tags::host{});

  buffer<Scalar> B;
  B.set_memtype(MemType::host_only);
  B.resize(fx.core.n_faces_local());

  const Scalar Bp = Scalar(10.0), obl = Scalar(0.5);

  auto shell_residual = [&](const stellar_moments& mom) {
    fx.core.fill_stellar_B(B, mom);
    // Net flux through each closed sphere shell (all tri faces of shell
    // k share the radial orientation convention) must vanish for a
    // solenoidal field with its sources inside — quadrature error only.
    double worst = 0.0;
    for (int k : {0, 4, 8}) {
      double net = 0.0, tot = 0.0;
      for (int l = 0; l < lp.n_owned_tri; l++) {
        gidx_t g = lp.tri_face_l2g[l];
        if (int(g / fx.mesh.m_N_tri) != k) continue;
        net += double(B[l]);
        tot += std::abs(double(B[l]));
      }
      REQUIRE(tot > 0.0);
      worst = std::max(worst, std::abs(net) / tot);
    }
    return worst;
  };

  // Control: the centered dipole validates the shared orientation
  // assumption of this test.
  stellar_extras none;
  double res_dip = shell_residual(stellar_moments_at(Bp, obl, none, 0.0));
  // Full multipole at a nonzero phase.
  double res_multi =
      shell_residual(stellar_moments_at(Bp, obl, test_extras(), 0.8));

  const double tol = std::max(1e-10, 1e4 * EPS);
  INFO("dipole shell residual = " << res_dip
                                  << ", multipole = " << res_multi);
  CHECK(res_dip < tol);
  CHECK(res_multi < tol);
}

TEST_CASE("apply_inner_bc: B overwrite at t=0 reproduces the IC fill exactly",
          "[prismatic][stellar_multipole]") {
  mesh_fixture fx(2, 8);
  auto lp = fx.core.get_lp(exec_tags::host{});
  const int nf = fx.core.n_faces_local(), ne = fx.core.n_edges_local();
  const int bs = fx.core.b_split();

  const Scalar Bp = Scalar(25.0), obl = Scalar(0.3);
  stellar_extras e = test_extras();

  buffer<Scalar> Bic, Bbc, B0, E;
  for (auto* b : {&Bic, &Bbc, &B0}) {
    b->set_memtype(MemType::host_only);
    b->resize(nf);
  }
  E.set_memtype(MemType::host_only);
  E.resize(ne);
  B0.assign(0);
  Bbc.assign(0);
  E.assign(0);

  fx.core.fill_stellar_B(Bic, stellar_moments_at(Bp, obl, e, 0.0));

  dec_inner_bc_params par;
  par.Bp = Bp;
  par.Omega = Scalar(0.2);
  par.obliquity = obl;
  par.stellar = e;
  par.overwrite_b = true;
  fx.core.apply_inner_bc(E, Bbc, B0, par, 0.0, 0.0);

  // Boundary tri and rect faces: the BC quadrature is the same code over
  // the same moments (phase Omega*0 = 0), so equality is exact.
  int n_checked = 0;
  for (int l = 0; l < lp.n_owned_tri; l++) {
    if (lp.tri_face_boundary[l] != 1) continue;
    REQUIRE(Bbc[l] == Bic[l]);
    n_checked++;
  }
  for (int l = 0; l < lp.n_owned_rect; l++) {
    if (lp.rect_face_boundary[l] != 1) continue;
    REQUIRE(Bbc[bs + l] == Bic[bs + l]);
    n_checked++;
  }
  REQUIRE(n_checked > 0);
}
