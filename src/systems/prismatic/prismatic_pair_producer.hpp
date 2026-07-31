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
    int capacity, int* cursor, int* overflow, IdPtr ptc_id,
    uint64_t track_rank, const pair_prod_params& par) {
  if (ptrs.cell[n] == empty_cell) return;
  const int sp = get_ptc_type(ptrs.flag[n]);
  if (sp != (int)PtcType::electron && sp != (int)PtcType::positron) return;

  const Scalar gamma = ptrs.E[n];
  if (gamma < par.gamma_thr) return;

  if (par.r_max > Scalar(0)) {
    int tri_idx, layer_idx;
    prism_cell_decode(ptrs.cell[n], N_tri, tri_idx, layer_idx);
    // Layer midpoint radius is a sufficient gate (the gate is a coarse
    // region switch, not physics).
    const Scalar r = Scalar(0.5) * (mp.radii[layer_idx] +
                                    mp.radii[layer_idx + 1]);
    if (r > par.r_max) return;
  }

  const int slot = atomic_add(cursor, 2);
  if (slot + 1 >= capacity) {
    atomic_add(overflow, 1);
    return;  // parent untouched; retries next step
  }
  const size_t ie = num + slot;      // electron child
  const size_t ip = num + slot + 1;  // positron child

  const bool gca = check_flag(ptrs.flag[n], PtcFlagEx::gca_state);
  const Scalar gamma_new = gamma - Scalar(2) * par.gamma_s;
  const Scalar us = math::sqrt(par.gamma_s * par.gamma_s - Scalar(1));

  Scalar ce1, ce2, ce3;  // child momentum slots
  if (gca) {
    // (u_par, mu, u_perp) representation; locked-limit deduction.
    const Scalar sign = ptrs.p1[n] >= Scalar(0) ? Scalar(1) : Scalar(-1);
    Scalar up2 = gamma_new * gamma_new - Scalar(1);
    ptrs.p1[n] = sign * math::sqrt(up2 > Scalar(0) ? up2 : Scalar(0));
    ptrs.E[n] = gamma_new;
    ce1 = sign * us;
    ce2 = Scalar(0);  // mu = 0 (synchrotron-locked birth, as inj_gca)
    ce3 = Scalar(0);
  } else {
    const Scalar p = math::sqrt(ptrs.p1[n] * ptrs.p1[n] +
                                ptrs.p2[n] * ptrs.p2[n] +
                                ptrs.p3[n] * ptrs.p3[n]);
    if (p < Scalar(1e-20)) {  // pathological zero-momentum "fast" particle
      atomic_add(overflow, 1);
      return;
    }
    const Scalar d1 = ptrs.p1[n] / p, d2 = ptrs.p2[n] / p,
                 d3 = ptrs.p3[n] / p;
    const Scalar pn = math::sqrt(gamma_new * gamma_new - Scalar(1));
    ptrs.p1[n] = d1 * pn;
    ptrs.p2[n] = d2 * pn;
    ptrs.p3[n] = d3 * pn;
    ptrs.E[n] = gamma_new;
    ce1 = d1 * us;
    ce2 = d2 * us;
    ce3 = d3 * us;
  }

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
    m_counters.resize(2);  // [0] = slot cursor, [1] = overflow count

    // Same tracked-id convention as the injector: rank in the high bits.
    m_track_rank = static_cast<uint64_t>(sim_env().get_rank()) << 32;

    Logger::print_info(
        "Pair production ON (stage 1, instant): gamma_thr = {}, "
        "gamma_secondary = {}, r_max = {}",
        thr, gs, rmax > 0 ? std::to_string(rmax) : std::string("(none)"));
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
    ExecPolicy::launch(
        [num, N_tri, capacity, par, lmp, track_rank]
        LAMBDA(auto ptc, auto counters, auto ptc_id) {
          ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
            pair_produce_single(lmp, N_tri, ptc, size_t(n), num, capacity,
                                &counters[0], &counters[1], ptc_id,
                                track_rank, par);
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
    if (step % 100 == 0 && (m_total_pairs > 0 || m_total_skipped > 0)) {
      Logger::print_info(
          "pair_producer: {} pairs to date ({} skipped on full buffer)",
          m_total_pairs, m_total_skipped);
    }
  }

 private:
  bool m_enabled = false;
  pair_prod_params m_par;
  prismatic_ptc_updater<ExecPolicy>* m_updater = nullptr;
  prismatic_particle_data* m_ptc = nullptr;
  uint64_t m_track_rank = 0;
  uint64_t m_total_pairs = 0, m_total_skipped = 0;
  buffer<int> m_counters;
};

using prismatic_pair_producer_t =
    prismatic_pair_producer<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
