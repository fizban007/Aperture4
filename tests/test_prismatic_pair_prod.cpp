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
// Stage-1 pair production (gamma-threshold instant pairs) — kernel-level
// tests on host pointers, driving pair_produce_single directly (the
// codebase's kernel-testing style).  See PAIR_PRODUCTION_PLAN.md.
// ===========================================================================

#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_pair_producer.hpp"
#include <cmath>
#include <vector>

using namespace Aperture;

namespace {

struct pp_fixture {
  prismatic_mesh mesh;
  prismatic_particles_t ptc;
  int cursor = 0, overflow = 0, capped = 0;
  std::vector<Scalar> rho_abs;
  uint64_t id_counter = 0;

  pp_fixture(int cap = 64) : ptc(cap, MemType::host_only) {
    mesh.build(2, 8, 1.0, 2.0);
  }

  // Seed one particle; returns its index.
  size_t seed(Scalar px, Scalar py, Scalar pz, Scalar gamma, PtcType type,
              uint32_t extra_flag = 0, int layer = 2, int tri = 7) {
    auto h = ptc.get_host_ptrs();
    size_t n = ptc.number();
    h.x1[n] = 0.3; h.x2[n] = 0.3; h.x3[n] = 0.5;
    h.p1[n] = px; h.p2[n] = py; h.p3[n] = pz;
    h.E[n] = gamma;
    h.weight[n] = 1.5;
    h.cell[n] = prism_cell_encode(tri, layer, mesh.m_N_tri);
    h.flag[n] = set_ptc_type_flag(extra_flag, type);
    h.id[n] = n;
    ptc.set_num(n + 1);
    return n;
  }

  // Run the kernel over the current population; returns produced count
  // (already clamped and applied to the live number).
  int run(const pair_prod_params& par) {
    auto h = ptc.get_host_ptrs();
    auto mp = mesh.host_ptrs();
    const size_t num = ptc.number();
    const int capacity = int(ptc.size() - num);
    cursor = 0;
    overflow = 0;
    for (size_t n = 0; n < num; n++) {
      pair_produce_single(mp, mesh.m_N_tri, h, n, num, capacity, &cursor,
                          &overflow, &capped, &id_counter, uint64_t(7) << 32,
                          par, rho_abs.empty() ? nullptr : rho_abs.data());
    }
    const int produced = std::min(cursor, capacity & ~1);
    ptc.set_num(num + produced);
    return produced;
  }

  double total_gamma_weight() {
    auto h = ptc.get_host_ptrs();
    double s = 0;
    for (size_t n = 0; n < ptc.number(); n++) {
      if (h.cell[n] == empty_cell) continue;
      s += double(h.E[n]) * double(h.weight[n]);
    }
    return s;
  }
};

}  // namespace

TEST_CASE("Pair production: threshold partition and energy conservation",
          "[pairprod]") {
  pp_fixture fx;
  pair_prod_params par;
  par.gamma_thr = 100;
  par.gamma_s = 5;

  // One eligible electron, one below-threshold positron.
  const Scalar g_hot = 150;
  const Scalar p_hot = std::sqrt(g_hot * g_hot - 1);
  // direction (2, -1, 2)/3
  size_t hot = fx.seed(p_hot * 2 / 3, -p_hot / 3, p_hot * 2 / 3, g_hot,
                       PtcType::electron);
  size_t cold = fx.seed(3, 4, 0, std::sqrt(Scalar(26)), PtcType::positron);

  auto h = fx.ptc.get_host_ptrs();
  const Scalar cold_p1 = h.p1[cold], cold_E = h.E[cold];
  const uint32_t cold_flag = h.flag[cold];
  const double gw_before = fx.total_gamma_weight();

  const int produced = fx.run(par);
  REQUIRE(produced == 2);
  REQUIRE(fx.ptc.number() == 4);

  // Energy conservation, exactly (instant-pair stage radiates nothing).
  REQUIRE(fx.total_gamma_weight() ==
          Catch::Approx(gw_before).epsilon(1e-6));

  // Below-threshold particle bitwise untouched.
  REQUIRE(h.p1[cold] == cold_p1);
  REQUIRE(h.E[cold] == cold_E);
  REQUIRE(h.flag[cold] == cold_flag);

  // Parent: gamma reduced by 2*gamma_s, direction preserved.
  REQUIRE(h.E[hot] == Catch::Approx(g_hot - 10));
  const Scalar pn = std::sqrt(h.p1[hot]*h.p1[hot] + h.p2[hot]*h.p2[hot] +
                              h.p3[hot]*h.p3[hot]);
  REQUIRE(h.p1[hot] / pn == Catch::Approx(2.0 / 3).epsilon(1e-5));
  REQUIRE(h.p2[hot] / pn == Catch::Approx(-1.0 / 3).epsilon(1e-5));

  // Children: one electron + one positron, secondary-flagged, parent's
  // weight/cell/position, gamma_s along the parent direction.
  int n_e = 0, n_p = 0;
  for (size_t n = 2; n < 4; n++) {
    if (get_ptc_type(h.flag[n]) == (int)PtcType::electron) n_e++;
    if (get_ptc_type(h.flag[n]) == (int)PtcType::positron) n_p++;
    REQUIRE(check_flag(h.flag[n], PtcFlag::secondary));
    REQUIRE(h.weight[n] == Scalar(1.5));
    REQUIRE(h.cell[n] == h.cell[hot]);
    REQUIRE(h.x1[n] == h.x1[hot]);
    REQUIRE(h.E[n] == Catch::Approx(5.0));
    const Scalar ps = std::sqrt(h.p1[n]*h.p1[n] + h.p2[n]*h.p2[n] +
                                h.p3[n]*h.p3[n]);
    REQUIRE(ps == Catch::Approx(std::sqrt(24.0)).epsilon(1e-5));
    REQUIRE(h.p1[n] / ps == Catch::Approx(2.0 / 3).epsilon(1e-5));
    // id carries the rank tag in the high bits.
    REQUIRE((h.id[n] >> 32) == 7);
  }
  REQUIRE(n_e == 1);
  REQUIRE(n_p == 1);
}

TEST_CASE("Pair production: GCA parents make GCA children", "[pairprod]") {
  pp_fixture fx;
  pair_prod_params par;
  par.gamma_thr = 50;
  par.gamma_s = 4;

  // GCA particle: p1 = u_par (negative), p2 = mu, p3 = u_perp.
  const Scalar g = 80;
  const Scalar u_par = -std::sqrt(g * g - 1);
  size_t n0 = fx.seed(u_par, Scalar(1e-4), 0, g, PtcType::positron,
                      flag_or(PtcFlagEx::gca_state));

  const int produced = fx.run(par);
  REQUIRE(produced == 2);

  auto h = fx.ptc.get_host_ptrs();
  // Parent: locked-limit deduction, sign of u_par preserved, mu kept.
  REQUIRE(h.E[n0] == Catch::Approx(72.0));
  REQUIRE(h.p1[n0] < 0);
  REQUIRE(h.p1[n0] ==
          Catch::Approx(-std::sqrt(72.0 * 72.0 - 1)).epsilon(1e-5));
  REQUIRE(h.p2[n0] == Scalar(1e-4));

  for (size_t n = 1; n < 3; n++) {
    REQUIRE(check_flag(h.flag[n], PtcFlagEx::gca_state));
    REQUIRE(check_flag(h.flag[n], PtcFlag::secondary));
    REQUIRE(h.p1[n] ==
            Catch::Approx(-std::sqrt(16.0 - 1)).epsilon(1e-5));  // sign kept
    REQUIRE(h.p2[n] == Scalar(0));  // mu = 0 at birth
    REQUIRE(h.E[n] == Catch::Approx(4.0));
  }
}

TEST_CASE("Pair production: ions immune, r-gate respected", "[pairprod]") {
  pp_fixture fx;
  pair_prod_params par;
  par.gamma_thr = 50;
  par.gamma_s = 4;
  par.r_max = Scalar(1.2);  // radii span [1, 2] over 8 layers

  const Scalar g = 100;
  const Scalar p = std::sqrt(g * g - 1);
  fx.seed(p, 0, 0, g, PtcType::ion);                    // ion, inner layer
  fx.seed(p, 0, 0, g, PtcType::electron, 0, /*layer=*/7);  // e-, outer layer
  size_t ok = fx.seed(p, 0, 0, g, PtcType::electron, 0, /*layer=*/0);

  const int produced = fx.run(par);
  REQUIRE(produced == 2);  // only the inner electron fires

  auto h = fx.ptc.get_host_ptrs();
  REQUIRE(h.E[ok] == Catch::Approx(92.0));
  REQUIRE(h.E[0] == Catch::Approx(100.0));  // ion untouched
  REQUIRE(h.E[1] == Catch::Approx(100.0));  // gated e- untouched
}

TEST_CASE("Pair production: buffer overflow is clamped and safe",
          "[pairprod]") {
  pp_fixture fx(6);  // room for 4 seeds + ONE pair
  pair_prod_params par;
  par.gamma_thr = 50;
  par.gamma_s = 4;

  const Scalar g = 100;
  const Scalar p = std::sqrt(g * g - 1);
  for (int i = 0; i < 4; i++) fx.seed(p, 0, 0, g, PtcType::electron);

  const int produced = fx.run(par);
  REQUIRE(produced == 2);          // exactly one pair fit
  REQUIRE(fx.overflow == 3);       // three productions skipped
  REQUIRE(fx.ptc.number() == 6);

  // Exactly one parent lost energy; skipped parents untouched (retry
  // next step).
  auto h = fx.ptc.get_host_ptrs();
  int reduced = 0;
  for (size_t n = 0; n < 4; n++) {
    if (h.E[n] < g - 1) reduced++;
    else REQUIRE(h.E[n] == Catch::Approx(100.0));
  }
  REQUIRE(reduced == 1);
  // The two children are fully formed.
  REQUIRE(h.E[4] == Catch::Approx(4.0));
  REQUIRE(h.E[5] == Catch::Approx(4.0));
  REQUIRE(h.cell[4] != empty_cell);
  REQUIRE(h.cell[5] != empty_cell);
}

// ===========================================================================
// REGRESSION (2026-08-01): a rejection test that runs AFTER the slot
// reservation leaves an unwritten hole which add_num() still counts, so the
// buffer gains a "live" particle holding a STALE cell.  In production that
// particle reached migrate(), was routed on garbage, and aborted a 320-rank
// job with "arrival misrouted" (diagnosed from a core dump, because the
// rank-0-only logger swallowed the message).
//
// The trigger is a lepton above threshold whose momentum is degenerate, so
// the pair has no direction to be beamed along.  Every rejection must now
// happen before the reservation; this test pins that by poisoning the
// slots past the live range and requiring that nothing claims them.
// ===========================================================================
TEST_CASE("Pair production: rejected parents never reserve a slot",
          "[pairprod]") {
  pp_fixture fx(16);
  pair_prod_params par;
  par.gamma_thr = 50;
  par.gamma_s = 4;

  const Scalar g = 100;
  const Scalar p = std::sqrt(g * g - 1);
  // (a) degenerate: above threshold but zero momentum -> must be rejected
  fx.seed(0, 0, 0, g, PtcType::electron);
  // (b) healthy parent -> must produce, and must land in slot 0/1
  size_t ok = fx.seed(p, 0, 0, g, PtcType::electron);

  // Poison every slot past the live range with a cell that no rank could
  // own: if a rejected parent reserves a slot, add_num() exposes it and
  // the poison survives as a "live" particle.
  auto h = fx.ptc.get_host_ptrs();
  const size_t n_live = fx.ptc.number();
  for (size_t i = n_live; i < fx.ptc.size(); i++) {
    h.cell[i] = 0xDEADBEEFu;
    h.E[i] = -1.0f;
  }

  const int produced = fx.run(par);

  // Exactly one pair, from the healthy parent only.
  REQUIRE(produced == 2);
  REQUIRE(fx.overflow == 1);            // the degenerate one was counted
  REQUIRE(fx.ptc.number() == n_live + 2);

  // Every particle now counted as live must carry a real cell.  Before
  // the fix the degenerate parent reserved slots 0/1, the healthy parent
  // wrote 2/3, and add_num(4) exposed the poisoned 0/1.
  for (size_t i = 0; i < fx.ptc.number(); i++) {
    INFO("slot " << i << " cell=" << h.cell[i]);
    REQUIRE(h.cell[i] != 0xDEADBEEFu);
    REQUIRE(h.E[i] > 0.0f);
  }
  // The degenerate parent is untouched (not silently drained of energy).
  REQUIRE(h.E[0] == Catch::Approx(100.0));
  REQUIRE(h.p1[0] == Scalar(0));
  // The healthy parent produced normally.
  REQUIRE(h.E[ok] == Catch::Approx(92.0));
}

// ===========================================================================
// Multiplicity cap (PS18-style).  Threshold production is self-limiting per
// PARTICLE but not per REGION: where E_par stays unscreened, each generation
// re-accelerates past the threshold and the population doubles every few
// steps.  On 2026-08-01 that filled one rank's entire 4e8 buffer while a
// neighbouring rank sat at 6e5.  The cap must stop production in cells that
// already hold more than max_mult * n_GJ(r).
// ===========================================================================
TEST_CASE("Pair production: multiplicity cap halts a loaded cell",
          "[pairprod]") {
  pp_fixture fx;
  auto mp = fx.mesh.host_ptrs();
  pair_prod_params par;
  par.gamma_thr = 50;
  par.gamma_s = 4;
  par.max_mult = 10;
  par.n_ref = 1000;  // n_GJ(r=1); profile r^-3

  const Scalar g = 100;
  const Scalar p = std::sqrt(g * g - 1);
  const int layer = 2, tri = 7;
  size_t n0 = fx.seed(p, 0, 0, g, PtcType::electron, 0, layer, tri);

  // rho_abs cochain: start empty -> cap must NOT bite.
  fx.rho_abs.assign(fx.mesh.m_N_verts, Scalar(0));
  REQUIRE(fx.run(par) == 2);
  REQUIRE(fx.capped == 0);

  // Now load the parent's cell above the cap.  deposit is |q|w per vertex
  // and the kernel divides by vert_dual_vol, so load each of the 6 prism
  // vertices to (11 * n_GJ) * dual_vol -> density 11 n_GJ > 10 n_GJ.
  const Scalar r = Scalar(0.5) * (mp.radii[layer] + mp.radii[layer + 1]);
  const Scalar n_gj = par.n_ref / (r * r * r);
  for (int vi = 0; vi < 3; vi++) {
    const int sv = mp.tri_verts[tri * 3 + vi];
    for (int kk : {layer, layer + 1}) {
      const int v = mp.vertex_idx(kk, sv);
      fx.rho_abs[v] = Scalar(11.0) * n_gj * mp.vert_dual_vol[v];
    }
  }
  // Reset to a single fresh above-threshold parent in the same cell.
  fx.ptc.set_num(0);
  fx.seed(p, 0, 0, g, PtcType::electron, 0, layer, tri);
  const int produced = fx.run(par);

  INFO("capped = " << fx.capped << ", produced = " << produced);
  REQUIRE(produced == 0);        // cap bit
  REQUIRE(fx.capped == 1);
  auto h = fx.ptc.get_host_ptrs();
  REQUIRE(h.E[0] == Catch::Approx(100.0));  // parent untouched by a capped
                                            // rejection (no energy drained)
  // And max_mult <= 0 disables the cap entirely (legacy behaviour).
  par.max_mult = 0;
  fx.ptc.set_num(0);
  fx.seed(p, 0, 0, g, PtcType::electron, 0, layer, tri);
  REQUIRE(fx.run(par) == 2);
}
