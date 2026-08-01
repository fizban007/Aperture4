#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_particles.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "utils/logger.h"

namespace Aperture {

// =========================================================================
// Stage-1 pair production: gamma-threshold instant pairs
// (PAIR_PRODUCTION_PLAN.md).
//
// Any electron/positron whose Lorentz factor reaches pair_gamma_thr
// spawns an e+/e- pair at its own location.  Children each carry
// gamma_s = pair_gamma_secondary along the parent's direction with the
// parent's weight; the parent loses 2*gamma_s with its direction kept —
// total sum(gamma * w) is conserved exactly (the intermediate photon is
// instantaneous; stage 2 gives it a finite free path).
//
// Purely LOCAL physics: children are born in the parent's cell on the
// parent's rank — no migration, no halo traffic, no new checkpoint
// state.  Registered BETWEEN the injector and the updater so newborns
// are pushed and deposit in the same step (injector semantics).
//
// GCA parents (flag gca_state): the momentum slots hold (u_par, mu,
// u_perp).  The deduction uses Gamma ≈ sqrt(1 + u_par²) — the locked
// limit (mu ≈ 0, drift factor kappa ≈ 1), which is the regime where
// particles reach the threshold.  mu is left untouched.  Children of
// GCA parents are born in the GCA representation (u_par =
// sign * sqrt(gamma_s² - 1), mu = 0), mirroring inj_gca.
// =========================================================================

struct pair_prod_params {
  Scalar gamma_thr = 0;    // trigger Lorentz factor
  Scalar gamma_s = 5;      // child Lorentz factor
  Scalar r_max = 0;        // radial gate; <= 0 = everywhere
  // Multiplicity cap (PS18: "a limiter on the number of produced pairs
  // ... so that the multiplicity in every cell where pairs are produced
  // does not exceed 10").  Production stops in a cell whose |charge|
  // density already exceeds max_mult * n_ref / r^3, with n_ref the
  // Goldreich-Julian scale at the stellar surface.  <= 0 disables.
  //
  // WHY THIS IS NOT OPTIONAL: threshold production is self-limiting per
  // PARTICLE (each pair costs the parent 2*gamma_s) but NOT per REGION.
  // Where E_par stays unscreened, each generation re-accelerates past
  // the threshold and the population doubles every few steps.  On
  // 2026-08-01 this filled one rank's entire 4e8 particle buffer in
  // ~1990 steps while a neighbouring rank sat at 6e5 -- a 650x
  // imbalance, killing the job.  The cap is what makes the cascade
  // terminate on the plasma it has produced.
  Scalar max_mult = 0;
  Scalar n_ref = 0;        // n_GJ at r = 1 (= 4 Omega Bp in code units)
};

// Produce a pair from particle n if it qualifies.  `cursor` counts
// reserved child slots (2 per production) atomically; slots at or past
// `capacity` are not written and the parent is left untouched (counted
// in `overflow`), so production retries next step.  Returns nothing;
// the caller clamps the live count to min(cursor, capacity) (contiguity
// holds: a thread writes exactly its own two slots iff they fit).
template <typename MP, typename IdPtr>
HOST_DEVICE inline void pair_produce_single(
    const MP& mp, int N_tri, prism_ptc_ptrs& ptrs, size_t n, size_t num,
    int capacity, int* cursor, int* overflow, int* capped, IdPtr ptc_id,
    uint64_t track_rank, const pair_prod_params& par,
    const Scalar* rho_abs) {
  if (ptrs.cell[n] == empty_cell) return;
  const int sp = get_ptc_type(ptrs.flag[n]);
  if (sp != (int)PtcType::electron && sp != (int)PtcType::positron) return;

  const Scalar gamma = ptrs.E[n];
  if (gamma < par.gamma_thr) return;

  int tri_idx, layer_idx;
  prism_cell_decode(ptrs.cell[n], N_tri, tri_idx, layer_idx);
  // Layer midpoint radius (the gates below are coarse region switches,
  // not physics).
  const Scalar r = Scalar(0.5) * (mp.radii[layer_idx] +
                                  mp.radii[layer_idx + 1]);
  if (par.r_max > Scalar(0) && r > par.r_max) return;

  // Multiplicity cap: stop where the cell already holds more than
  // max_mult * n_GJ(r).  rho_abs is the PREVIOUS step's deposit (this
  // system runs before the updater) -- one step stale, which is
  // irrelevant against a cap that only has to act on a timescale of
  // several doublings.  Hat weights are all 1/6 at the prism centre,
  // matching the injector's own multiplicity read.
  if (par.max_mult > Scalar(0) && rho_abs != nullptr) {
    Scalar ra = 0;
    for (int vi = 0; vi < 3; vi++) {
      const int sv = mp.tri_verts[tri_idx * 3 + vi];
      const int vb = mp.vertex_idx(layer_idx, sv);
      const int vt = mp.vertex_idx(layer_idx + 1, sv);
      ra += Scalar(1.0 / 6) *
            (mp.vert_dual_vol[vb] > 0 ? rho_abs[vb] / mp.vert_dual_vol[vb]
                                      : Scalar(0));
      ra += Scalar(1.0 / 6) *
            (mp.vert_dual_vol[vt] > 0 ? rho_abs[vt] / mp.vert_dual_vol[vt]
                                      : Scalar(0));
    }
    const Scalar n_gj = par.n_ref / (r * r * r);
    if (ra > par.max_mult * n_gj) {
      atomic_add(capped, 1);
      return;
    }
  }

  // ---------------------------------------------------------------------
  // INVARIANT: every rejection test must run BEFORE the slot reservation.
  // A thread that reserves a slot and then bails leaves an unwritten hole
  // that add_num() still counts, so the buffer gains a "live" particle
  // holding a STALE cell — which migrate() then routes on garbage and the
  // receiver rejects as a misrouted arrival (an abort 40 nodes wide).
  // The capacity check below is the ONLY post-reservation bail, and it is
  // safe because slots are handed out in increasing order: the threads
  // that skip are exactly the highest-slot ones, and the caller counts
  // only min(reserved, capacity & ~1).
  // ---------------------------------------------------------------------
  const bool gca = check_flag(ptrs.flag[n], PtcFlagEx::gca_state);
  const Scalar gamma_new = gamma - Scalar(2) * par.gamma_s;
  const Scalar us = math::sqrt(par.gamma_s * par.gamma_s - Scalar(1));

  Scalar ce1, ce2, ce3;      // child momentum slots
  Scalar pp1, pp2, pp3;      // parent momentum after the deduction
  if (gca) {
    // (u_par, mu, u_perp) representation; locked-limit deduction.
    const Scalar sign = ptrs.p1[n] >= Scalar(0) ? Scalar(1) : Scalar(-1);
    const Scalar up2 = gamma_new * gamma_new - Scalar(1);
    pp1 = sign * math::sqrt(up2 > Scalar(0) ? up2 : Scalar(0));
    pp2 = ptrs.p2[n];  // mu untouched
    pp3 = ptrs.p3[n];
    ce1 = sign * us;
    ce2 = Scalar(0);  // mu = 0 (synchrotron-locked birth, as inj_gca)
    ce3 = Scalar(0);
  } else {
    const Scalar p = math::sqrt(ptrs.p1[n] * ptrs.p1[n] +
                                ptrs.p2[n] * ptrs.p2[n] +
                                ptrs.p3[n] * ptrs.p3[n]);
    // Degenerate: gamma above threshold but no momentum direction to
    // beam the pair along.  Reject BEFORE reserving (see invariant).
    if (!(p > Scalar(1e-20))) {  // also rejects NaN
      atomic_add(overflow, 1);
      return;
    }
    const Scalar d1 = ptrs.p1[n] / p, d2 = ptrs.p2[n] / p,
                 d3 = ptrs.p3[n] / p;
    const Scalar pn = math::sqrt(gamma_new * gamma_new - Scalar(1));
    pp1 = d1 * pn;
    pp2 = d2 * pn;
    pp3 = d3 * pn;
    ce1 = d1 * us;
    ce2 = d2 * us;
    ce3 = d3 * us;
  }

  const int slot = atomic_add(cursor, 2);
  if (slot + 1 >= capacity) {
    atomic_add(overflow, 1);
    return;  // parent untouched; retries next step
  }
  const size_t ie = num + slot;      // electron child
  const size_t ip = num + slot + 1;  // positron child

  // Commit the parent only now that the children's slots are secured.
  ptrs.p1[n] = pp1;
  ptrs.p2[n] = pp2;
  ptrs.p3[n] = pp3;
  ptrs.E[n] = gamma_new;

  const uint32_t base_flag =
      (gca ? flag_or(PtcFlag::secondary, PtcFlagEx::gca_state)
           : flag_or(PtcFlag::secondary));
  for (int c = 0; c < 2; c++) {
    const size_t i = (c == 0) ? ie : ip;
    ptrs.x1[i] = ptrs.x1[n];
    ptrs.x2[i] = ptrs.x2[n];
    ptrs.x3[i] = ptrs.x3[n];
    ptrs.p1[i] = ce1;
    ptrs.p2[i] = ce2;
    ptrs.p3[i] = ce3;
    ptrs.E[i] = par.gamma_s;
    ptrs.weight[i] = ptrs.weight[n];
    ptrs.cell[i] = ptrs.cell[n];
    ptrs.flag[i] = set_ptc_type_flag(
        base_flag, (c == 0) ? PtcType::electron : PtcType::positron);
    ptrs.id[i] = track_rank + atomic_add(ptc_id, 1);
  }
}

// -------------------------------------------------------------------------
// System wrapper.
// -------------------------------------------------------------------------
template <typename ExecPolicy>
class prismatic_pair_producer : public system_t {
 public:
  static std::string name() { return "prismatic_pair_producer"; }

  prismatic_pair_producer() = default;

  void register_data_components() override {}

  void init() override {
    sim_env().params().get_value("use_pair_production", m_enabled);
    if (!m_enabled) {
      Logger::print_info("Pair production: OFF");
      return;
    }
    double thr = 0, gs = 5, rmax = 0;
    sim_env().params().get_value("pair_gamma_thr", thr);
    sim_env().params().get_value("pair_gamma_secondary", gs);
    sim_env().params().get_value("pair_prod_r_max", rmax);
    if (thr < 2.0 * gs + 2.0) {
      Logger::print_err(
          "use_pair_production is on but pair_gamma_thr = {} < "
          "2*pair_gamma_secondary + 2 = {} (parent must remain "
          "relativistic after the deduction).  Set pair_gamma_thr and "
          "pair_gamma_secondary consistently.",
          thr, 2.0 * gs + 2.0);
      std::abort();
    }
    m_par.gamma_thr = Scalar(thr);
    m_par.gamma_s = Scalar(gs);
    m_par.r_max = Scalar(rmax);

    // Multiplicity cap.  Reference density is the GJ scale at the pole,
    // n_GJ(r) = 2 Omega B_pole / r^3 with B_pole = 2 Bp (dipole_B_impl
    // takes the moment, so the polar field is twice the config's Bp);
    // rationalized units, rho_GJ = 2 Omega . B, no 4pi.
    double max_mult = 0, Omega = 0, Bp = 0;
    sim_env().params().get_value("pair_max_multiplicity", max_mult);
    sim_env().params().get_value("Omega", Omega);
    sim_env().params().get_value("Bp", Bp);
    m_par.max_mult = Scalar(max_mult);
    m_par.n_ref = Scalar(4.0 * Omega * Bp);
    if (max_mult <= 0) {
      Logger::print_err(
          "pair_max_multiplicity is unset (<= 0): threshold pair "
          "production is UNCAPPED.  It is self-limiting per particle but "
          "not per region — an unscreened gap doubles its population "
          "every few steps and will fill the particle buffer (this "
          "happened on 2026-08-01).  Set pair_max_multiplicity (PS18 use "
          "10) unless you specifically want the uncapped behaviour.");
    }
    nonown_ptr<prismatic_vertex_field> rho_abs;
    sim_env().get_data_optional("rho_abs", rho_abs);
    if (rho_abs != nullptr) m_rho_abs = &(*rho_abs);
    if (max_mult > 0 && m_rho_abs == nullptr) {
      Logger::print_err(
          "pair_max_multiplicity is set but 'rho_abs' is unavailable "
          "(deposit_diagnostics off?) — the cap CANNOT act.  Enable "
          "deposit_diagnostics.");
      std::abort();
    }

    auto upd = sim_env().get_system("prismatic_ptc_updater");
    if (upd == nullptr) {
      Logger::print_err(
          "prismatic_pair_producer requires prismatic_ptc_updater");
      std::abort();
    }
    m_updater = &dynamic_cast<prismatic_ptc_updater<ExecPolicy>&>(*upd);

    nonown_ptr<prismatic_particle_data> ptc;
    sim_env().get_data("particles", ptc);
    m_ptc = &(*ptc);

    m_counters.set_memtype(ExecPolicy::data_mem_type());
    m_counters.resize(3);  // [0] cursor, [1] overflow, [2] capped

    // Same tracked-id convention as the injector: rank in the high bits.
    m_track_rank = static_cast<uint64_t>(sim_env().get_rank()) << 32;

    Logger::print_info(
        "Pair production ON (stage 1, instant): gamma_thr = {}, "
        "gamma_secondary = {}, r_max = {}, max_multiplicity = {} "
        "(n_GJ(1) = {:.4g}, profile r^-3)",
        thr, gs, rmax > 0 ? std::to_string(rmax) : std::string("(none)"),
        max_mult > 0 ? std::to_string(max_mult) : std::string("UNCAPPED"),
        4.0 * Omega * Bp);
  }

  void update(double dt, uint32_t step) override {
    if (!m_enabled) return;
    auto lmp = m_updater->ptc_mesh().get_ptrs(typename ExecPolicy::exec_tag{});
    const int N_tri = lmp.N_tri;
    const size_t num = m_ptc->number();
    const int capacity = int(m_ptc->size() - num);
    const auto par = m_par;
    const uint64_t track_rank = m_track_rank;

    m_counters.assign(0);
    const Scalar* rho_abs_p =
        m_rho_abs != nullptr
            ? (m_rho_abs->data().dev_ptr() != nullptr
                   ? m_rho_abs->data().dev_ptr()
                   : m_rho_abs->data().host_ptr())
            : nullptr;
    ExecPolicy::launch(
        [num, N_tri, capacity, par, lmp, track_rank, rho_abs_p]
        LAMBDA(auto ptc, auto counters, auto ptc_id) {
          ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
            pair_produce_single(lmp, N_tri, ptc, size_t(n), num, capacity,
                                &counters[0], &counters[1], &counters[2],
                                ptc_id, track_rank, par, rho_abs_p);
          });
        },
        *m_ptc, m_counters, m_ptc->ptc_id());
    ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_counters.copy_to_host();
#endif
    const int reserved = m_counters[0];
    const int produced = std::min(reserved, capacity & ~1);
    m_ptc->add_num(produced);
    m_total_pairs += produced / 2;
    m_total_skipped += m_counters[1];
    m_total_capped += m_counters[2];
    if (step % 100 == 0 && (m_total_pairs > 0 || m_total_skipped > 0)) {
      Logger::print_info(
          "pair_producer: {} pairs to date ({} capped by multiplicity, "
          "{} skipped on full buffer)",
          m_total_pairs, m_total_capped, m_total_skipped);
    }
    // Rank-LOCAL runaway alarm.  The census above prints on rank 0 only,
    // which is exactly how the 2026-08-01 blow-up hid: rank 0 sat at
    // 6e5 particles while rank 266 filled its whole 4e8 buffer.  Report
    // from ANY rank that crosses a fraction of its own buffer, and again
    // on each further decade of occupancy.
    const double frac = double(m_ptc->number()) / double(m_ptc->size());
    if (frac > 0.5 && m_ptc->number() > m_last_alarm * 2) {
      m_last_alarm = m_ptc->number();
      Logger::print_err_all(
          "pair_producer: particle buffer {:.1f}% full ({} of {}) at step "
          "{} — local cascade may be running away; {} pairs made here, {} "
          "capped by multiplicity",
          100.0 * frac, m_ptc->number(), m_ptc->size(), step, m_total_pairs,
          m_total_capped);
    }
  }

 private:
  bool m_enabled = false;
  pair_prod_params m_par;
  prismatic_ptc_updater<ExecPolicy>* m_updater = nullptr;
  prismatic_particle_data* m_ptc = nullptr;
  uint64_t m_track_rank = 0;
  uint64_t m_total_pairs = 0, m_total_skipped = 0, m_total_capped = 0;
  size_t m_last_alarm = 1;
  prismatic_vertex_field* m_rho_abs = nullptr;
  buffer<int> m_counters;
};

using prismatic_pair_producer_t =
    prismatic_pair_producer<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
