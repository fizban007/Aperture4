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
// Synchrotron cooling on the prismatic hybrid pusher's Boris branch.
//
// These exercise the Landau-Lifshitz drag (systems/physics/
// radiation_reaction.hpp) composed with the prismatic boris_push, in
// uniform fields -- no mesh required, since both act pointwise on the
// gathered E and B.
//
// The headline property is PITCH ANGLE PRESERVATION at high gamma: the
// scheme this replaced damped the perpendicular momentum directly, which
// drives the pitch angle to zero at every gamma and is wrong.
// ===========================================================================

#include "catch2/catch_all.hpp"
#include "systems/physics/radiation_reaction.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_particles.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include <cmath>
#include <vector>

using namespace Aperture;

namespace {

// One cooled Boris step in uniform fields: Lorentz force, then the
// operator-split radiation reaction -- exactly the Boris branch of
// update_single_particle().
void cooled_step(Scalar& px, Scalar& py, Scalar& pz, Scalar& gamma,
                 Scalar Ex, Scalar Ey, Scalar Ez, Scalar Bx, Scalar By,
                 Scalar Bz, Scalar q, Scalar m, Scalar dt, Scalar coef) {
  gamma = boris_push(px, py, pz, Ex, Ey, Ez, Bx, By, Bz, q, m, dt);
  sync_drag_substep(px, py, pz, gamma, Ex, Ey, Ez, Bx, By, Bz, coef, dt);
}

Scalar norm3(Scalar x, Scalar y, Scalar z) {
  return std::sqrt(x * x + y * y + z * z);
}

// Pitch angle between p and B, in radians.
Scalar pitch_angle(Scalar px, Scalar py, Scalar pz, Scalar Bx, Scalar By,
                   Scalar Bz) {
  Scalar p = norm3(px, py, pz);
  Scalar B = norm3(Bx, By, Bz);
  Scalar c = (px * Bx + py * By + pz * Bz) / (p * B);
  c = std::max(Scalar(-1), std::min(Scalar(1), c));
  return std::acos(c);
}

}  // namespace

// ---------------------------------------------------------------------------
// The property the whole change exists for.  A high-gamma particle at 45 deg
// pitch must lose a large fraction of its energy while the pitch angle stays
// put.  A perpendicular-damping scheme fails this: it drives the pitch to 0.
// ---------------------------------------------------------------------------
TEST_CASE("Synchrotron drag preserves pitch angle at high gamma",
          "[pusher]") {
  const Scalar Bz = 10.0, q = -1.0, m = 1.0, dt = 1.0e-4;
  const Scalar coef = 2.0e-5;

  const Scalar gamma0 = 2000.0;
  const Scalar u0 = std::sqrt(gamma0 * gamma0 - 1.0);
  const Scalar theta0 = M_PI / 4;  // 45 degrees
  Scalar px = u0 * std::sin(theta0), py = 0.0, pz = u0 * std::cos(theta0);
  Scalar gamma = gamma0;

  const Scalar pitch0 = pitch_angle(px, py, pz, 0, 0, Bz);

  for (int i = 0; i < 20000; i++) {
    cooled_step(px, py, pz, gamma, 0, 0, 0, 0, 0, Bz, q, m, dt, coef);
  }

  // Substantial cooling actually happened...
  REQUIRE(gamma < 0.5 * gamma0);
  REQUIRE(gamma > 1.0);
  // ...and the pitch angle rode through it essentially unchanged.
  const Scalar pitch1 = pitch_angle(px, py, pz, 0, 0, Bz);
  REQUIRE(std::abs(pitch1 - pitch0) < 0.02);
}

// ---------------------------------------------------------------------------
// Quantitative rate check.  Ultrarelativistic, 90 deg pitch, E = 0:
//   dgamma/dt = -coef gamma^2 B^2   =>   1/gamma(t) = 1/gamma0 + coef B^2 t
// ---------------------------------------------------------------------------
TEST_CASE("Synchrotron drag energy loss rate", "[pusher]") {
  const Scalar Bz = 10.0, q = -1.0, m = 1.0, dt = 1.0e-4;
  const Scalar coef = 2.0e-5;

  const Scalar gamma0 = 1000.0;
  Scalar px = std::sqrt(gamma0 * gamma0 - 1.0), py = 0.0, pz = 0.0;
  Scalar gamma = gamma0;

  const int n_steps = 10000;
  for (int i = 0; i < n_steps; i++) {
    cooled_step(px, py, pz, gamma, 0, 0, 0, 0, 0, Bz, q, m, dt, coef);
  }

  const Scalar t = n_steps * dt;
  const Scalar expected = 1.0 / (1.0 / gamma0 + coef * Bz * Bz * t);
  REQUIRE(gamma == Catch::Approx(expected).epsilon(0.02));
}

// ---------------------------------------------------------------------------
// A particle moving exactly along B feels E' = E + v x B = 0, so the drag
// vanishes identically -- no spurious parallel friction.  (Its real energy
// loss channel is CURVATURE radiation, deliberately out of scope.)
// ---------------------------------------------------------------------------
TEST_CASE("Synchrotron drag leaves field-aligned motion untouched",
          "[pusher]") {
  const Scalar Bz = 10.0, q = -1.0, m = 1.0, dt = 1.0e-4;
  const Scalar coef = 1.0e-3;

  const Scalar gamma0 = 500.0;
  Scalar px = 0.0, py = 0.0, pz = std::sqrt(gamma0 * gamma0 - 1.0);
  Scalar gamma = gamma0;

  for (int i = 0; i < 5000; i++) {
    cooled_step(px, py, pz, gamma, 0, 0, 0, 0, 0, Bz, q, m, dt, coef);
  }

  REQUIRE(gamma == Catch::Approx(gamma0).epsilon(1.0e-6));
  REQUIRE(std::abs(px) < 1.0e-6 * gamma0);
  REQUIRE(std::abs(py) < 1.0e-6 * gamma0);
}

// ---------------------------------------------------------------------------
// THE consistency test (plan section 4.1).  In crossed E, B with E < B, a
// heavily-cooled particle must relax onto the drift solution
// v = v_ExB + v_par b -- NOT onto zero.  A drag that damped raw p_perp
// would put friction on the bulk drift flow and fail here; the LL force
// vanishes in the drift frame because E' -> 0 there.
// ---------------------------------------------------------------------------
TEST_CASE("Cooled Boris converges to the ExB drift, not to rest",
          "[pusher]") {
  // B along z, E along x  =>  v_ExB = E x B / B^2 along MINUS y, |v| = E/B
  const Scalar Bz = 10.0, Ex = 2.0;
  const Scalar q = -1.0, m = 1.0, dt = 1.0e-4;
  const Scalar coef = 1.0e-2;   // aggressive: t_cool << run length
  const Scalar vE = -Ex / Bz;   // = -0.2, well below c

  // Large gyration plus a parallel drift
  Scalar px = 30.0, py = -20.0, pz = 12.0;
  Scalar gamma = std::sqrt(1.0 + px * px + py * py + pz * pz);

  for (int i = 0; i < 200000; i++) {
    cooled_step(px, py, pz, gamma, Ex, 0, 0, 0, 0, Bz, q, m, dt, coef);
  }

  const Scalar vx = px / gamma, vy = py / gamma, vz = pz / gamma;

  // Perpendicular motion is the drift itself, not zero.  A drag that
  // damped raw p_perp would have eaten this.
  REQUIRE(vy == Catch::Approx(vE).epsilon(0.02));
  REQUIRE(std::abs(vx) < 1.0e-2);
  REQUIRE(norm3(vx, vy, vz) > 0.3);  // emphatically not at rest

  // Parallel motion survives: once the gyration is gone E' -> 0 and the
  // drag switches off, leaving v = v_ExB + v_par b.
  REQUIRE(vz > 0.3);

  // And the endpoint matches a mu = 0 GCA push of the same u_par --
  // cooled-Boris == locked-GCA in the overlap region, which is the whole
  // point of putting the cooling in.
  //   Gamma = kappa sqrt(1 + u_par^2),  kappa = 1/sqrt(1 - vE^2)
  const Scalar kappa = 1.0 / std::sqrt(1.0 - vE * vE);
  const Scalar u_par = pz;  // b = z_hat
  const Scalar gamma_gca = kappa * std::sqrt(1.0 + u_par * u_par);
  REQUIRE(gamma == Catch::Approx(gamma_gca).epsilon(1.0e-3));
}

// ---------------------------------------------------------------------------
// Regularity.  The LL drag has no drift frame and no 1/B, so the two cases
// that forced guards in the guiding-centre code -- B -> 0 and E >= B --
// need none here.  They must simply stay finite.
// ---------------------------------------------------------------------------
TEST_CASE("Synchrotron drag is regular at B -> 0 and E >= B", "[pusher]") {
  const Scalar q = -1.0, m = 1.0, dt = 1.0e-4, coef = 1.0e-4;

  SECTION("vanishing B") {
    Scalar px = 50.0, py = 10.0, pz = -3.0;
    Scalar gamma = std::sqrt(1.0 + px * px + py * py + pz * pz);
    for (int i = 0; i < 1000; i++) {
      cooled_step(px, py, pz, gamma, 0, 0, 0, 0, 0, 0, q, m, dt, coef);
    }
    REQUIRE(std::isfinite(gamma));
    REQUIRE(std::isfinite(px));
    // No fields at all: nothing should have changed.
    REQUIRE(px == Catch::Approx(50.0));
  }

  SECTION("E exceeds B (reconnection core)") {
    Scalar px = 5.0, py = 1.0, pz = 0.5;
    Scalar gamma = std::sqrt(1.0 + px * px + py * py + pz * pz);
    for (int i = 0; i < 1000; i++) {
      cooled_step(px, py, pz, gamma, 10.0, 0, 0, 0, 0, 1.0, q, m, dt, coef);
    }
    REQUIRE(std::isfinite(gamma));
    REQUIRE(std::isfinite(px));
    REQUIRE(std::isfinite(py));
    REQUIRE(std::isfinite(pz));
    REQUIRE(gamma >= 1.0);
  }
}

// ---------------------------------------------------------------------------
// Stiffness.  The locked limit runs at a cooling time of order the STEP,
// which is where a plain implicit-midpoint fixed point stops contracting and
// diverges to NaN.  The exponential update must stay bounded and physical
// however far past that it is pushed -- a NaN here is a dead production run.
// ---------------------------------------------------------------------------
TEST_CASE("Synchrotron drag stays physical when the cooling time is a step",
          "[pusher]") {
  const Scalar Bz = 10.0, q = -1.0, m = 1.0, dt = 1.0e-3;

  // nu dt at the initial gamma spans ~1e-1 (mild) to ~1e4 (absurd).
  for (Scalar coef : {Scalar(1.0e-2), Scalar(1.0), Scalar(100.0)}) {
    Scalar px = 100.0, py = 0.0, pz = 20.0;
    Scalar gamma = std::sqrt(1.0 + px * px + py * py + pz * pz);
    const Scalar gamma0 = gamma;
    for (int i = 0; i < 2000; i++) {
      cooled_step(px, py, pz, gamma, 0, 0, 0, 0, 0, Bz, q, m, dt, coef);
    }
    INFO("cooling coefficient " << coef);
    REQUIRE(std::isfinite(gamma));
    REQUIRE(std::isfinite(px));
    REQUIRE(std::isfinite(py));
    REQUIRE(std::isfinite(pz));
    // Bounded below by rest and above by the uncooled energy: the drag
    // only ever removes energy, and never overshoots through zero.
    REQUIRE(gamma >= 1.0);
    REQUIRE(gamma <= gamma0);
    // Stronger cooling must not cool LESS -- monotonic in the coefficient
    // would be nice but is not guaranteed at absurd stiffness; requiring
    // real cooling in every case is.
    REQUIRE(gamma < 0.5 * gamma0);
  }
}

// ---------------------------------------------------------------------------
// Cooling off (coef = 0) must reproduce the uncooled Boris push bit for bit,
// so existing configs and checkpoints are unaffected by this change.
// ---------------------------------------------------------------------------
TEST_CASE("Zero cooling coefficient is a no-op", "[pusher]") {
  const Scalar Bz = 10.0, Ex = 1.0, q = -1.0, m = 1.0, dt = 1.0e-3;

  Scalar ax = 7.0, ay = -2.0, az = 3.0;
  Scalar ga = std::sqrt(1.0 + ax * ax + ay * ay + az * az);
  Scalar bx = ax, by = ay, bz = az, gb = ga;

  for (int i = 0; i < 500; i++) {
    ga = boris_push(ax, ay, az, Ex, 0, 0, 0, 0, Bz, q, m, dt);
    cooled_step(bx, by, bz, gb, Ex, 0, 0, 0, 0, Bz, q, m, dt, Scalar(0));
  }

  REQUIRE(bx == ax);
  REQUIRE(by == ay);
  REQUIRE(bz == az);
  REQUIRE(gb == ga);
}

// ===========================================================================
// Hybrid-switch dispatch: dt-invariance (plan section 4.3).
//
// THIS IS THE REGRESSION TEST FOR THE ORIGINAL BUG CLASS.  The criterion
// used to be omega_c dt / gamma > const, so the physical switching surface
// was a function of the step: the same configured "0.1" put the surface at
// rate 10.2 / 20.4 / 40.7 at L5 / L6 / L7.  It is now the RATE
// omega_c / gamma > gca_switch_omegac, with dt absent, so the SAME particle
// in the SAME field must land on the same side of the switch at any dt.
//
// Run the identical particle set at dt, dt/2 and dt/4 and require the
// per-particle GCA/Boris decision to be bit-identical.  Reintroducing dt
// into the criterion fails this immediately.
// ===========================================================================
TEST_CASE("Hybrid switch dispatch is dt-invariant", "[pusher][prismatic]") {
  constexpr int TL = 2, TN_r = 8;
  prismatic_mesh mesh;
  mesh.build(TL, TN_r, 1.0, 2.0);
  auto mp = mesh.host_ptrs();

  std::vector<Scalar> E(mesh.m_N_edges), B(mesh.m_N_faces);
  for (int e = 0; e < mesh.m_N_edges; ++e)
    E[e] = Scalar(0.01) * std::sin(Scalar(0.013) * e);
  for (int f = 0; f < mesh.m_N_faces; ++f)
    B[f] = Scalar(0.5) + Scalar(0.1) * std::cos(Scalar(0.007) * f);

  // gamma is spread over four decades so omega_c/gamma = |q/m| B / gamma
  // straddles the threshold whatever the gathered |B| turns out to be --
  // without a straddle the comparison below would pass vacuously.
  const int n_ptc = 400;
  const Scalar switch_omegac = 0.2;

  auto seed_all = [&](prismatic_particles_t& p) {
    auto h = p.get_host_ptrs();
    int n = 0;
    for (int t = 0; t < mesh.m_N_tri && n < n_ptc; t += 3) {
      for (int k = 0; k < mesh.m_N_r && n < n_ptc; k += 2) {
        const Scalar gam =
            std::pow(Scalar(10), Scalar(4) * Scalar(n % 40) / Scalar(39));
        const Scalar u = std::sqrt(gam * gam - Scalar(1));
        // Direction varies per particle; magnitude consistent with gamma.
        const Scalar a = Scalar(0.3) * n, b = Scalar(0.5) * n;
        h.x1[n] = Scalar(0.3);
        h.x2[n] = Scalar(0.3);
        h.x3[n] = Scalar(0.5);
        h.p1[n] = u * std::sin(a) * std::cos(b);
        h.p2[n] = u * std::sin(a) * std::sin(b);
        h.p3[n] = u * std::cos(a);
        h.E[n] = gam;
        h.weight[n] = Scalar(1);
        h.cell[n] = uint32_t(k * mesh.m_N_tri + t);
        h.flag[n] = 0u;
        n++;
      }
    }
    p.set_num(n);
    return n;
  };

  // One step at a given dt; returns the per-particle GCA/Boris decision.
  auto partition_at = [&](Scalar dt) {
    prismatic_particles_t ptc(n_ptc, MemType::host_only);
    const int n_seed = seed_all(ptc);
    auto h = ptc.get_host_ptrs();
    for (int n = 0; n < n_seed; ++n) {
      update_single_particle(mp, mp.N_tri, h, n, E.data(), B.data(), nullptr,
                             nullptr, Scalar(-1), Scalar(1), dt,
                             /*use_gca=*/true, /*include_curvature=*/false,
                             /*Bv_rec=*/nullptr, /*absorb_r=*/Scalar(0),
                             nullptr, nullptr, switch_omegac,
                             /*zero_mu_on_capture=*/false,
                             /*sync_cool_coef=*/Scalar(0));
    }
    std::vector<char> gca(n_seed);
    for (int n = 0; n < n_seed; ++n) {
      gca[n] = check_flag(h.flag[n], PtcFlagEx::gca_state) ? 1 : 0;
    }
    return gca;
  };

  const auto at_dt = partition_at(Scalar(2.0e-3));
  const auto at_half = partition_at(Scalar(1.0e-3));
  const auto at_quarter = partition_at(Scalar(5.0e-4));

  REQUIRE(at_dt.size() == at_half.size());
  REQUIRE(at_dt.size() == at_quarter.size());

  // The partition must be non-trivial, or the comparison below proves
  // nothing.  This is also the assertion that catches a reintroduced dt:
  // scaling omega_c by dt shifts every particle to the same side of a
  // fixed threshold, collapsing the partition.  A failure here means the
  // criterion is no longer a pure rate -- check that the dispatch in
  // prismatic_ptc_update_kernel.hpp has no dt in it.
  int n_gca = 0;
  for (char c : at_dt) n_gca += c;
  INFO("GCA fraction at dt: " << double(n_gca) / at_dt.size()
                              << " (must be strictly between 0 and 1)");
  REQUIRE(n_gca > 0);
  REQUIRE(n_gca < int(at_dt.size()));

  // ...and identical at every dt.  This is the property the old
  // omega_c*dt criterion could not have.
  for (size_t n = 0; n < at_dt.size(); ++n) {
    INFO("particle " << n);
    REQUIRE(at_dt[n] == at_half[n]);
    REQUIRE(at_dt[n] == at_quarter[n]);
  }
}
