#pragma once

#include "core/random.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_injector.hpp"
#include "utils/logger.h"
#include <memory>

namespace Aperture {

// Pair injection for magnetosphere runs: every inj_interval steps,
// inject inj_pairs_per_cell neutral e+/e- pairs, uniformly placed, in
// every eligible prism, with an isotropic Maxwell-Juttner momentum
// spread of temperature inj_kT and macro-weight inj_weight.
//
// Eligibility has two modes:
//   - surface (inj_r_max <= 0): the first inj_shells radial layers;
//   - volumetric (inj_r_max > 0): every cell whose center radius is
//     below inj_r_max.
// In both modes a positive inj_eb_threshold additionally requires the
// unscreened parallel field at the cell center to satisfy
// |E.B|/|B|^2 > inj_eb_threshold, so injection tracks local demand and
// quenches itself where the plasma has shorted out E_par.
//
// Surface-only thermal injection is KNOWN NOT to reach the corotating
// force-free solution — it stalls in a charge-separated state ("dead"
// electrosphere) with large unscreened E_par regions.  The volumetric
// E.B-triggered mode is the simplest scheme that does converge
// (standing in for the self-consistent pair production of
// Chen & Beloborodov 2014); note a uniform macro-weight already gives
// an injected number density per event ~ 1/V_cell ~ r^-3, i.e. the GJ
// radial scaling.
// Additionally throttled by total buffer occupancy (inj_buffer_frac).
//
// Register AFTER the field solver and BEFORE prismatic_ptc_updater, so
// freshly injected particles are pushed (and deposit current) in the
// same step they appear.
template <typename ExecPolicy>
class prismatic_surface_injector : public system_t {
 public:
  static std::string name() { return "prismatic_surface_injector"; }

  explicit prismatic_surface_injector(const prismatic_mesh& mesh)
      : m_mesh(mesh) {}

  void register_data_components() override {}

  void init() override {
    sim_env().params().get_value("inj_shells", m_inj_shells);
    sim_env().params().get_value("inj_pairs_per_cell", m_pairs_per_cell);
    sim_env().params().get_value("inj_interval", m_interval);
    sim_env().params().get_value("inj_weight", m_weight);
    sim_env().params().get_value("inj_kT", m_kT);
    sim_env().params().get_value("inj_buffer_frac", m_buffer_frac);
    sim_env().params().get_value("inj_eb_threshold", m_eb_threshold);
    sim_env().params().get_value("inj_r_max", m_inj_r_max);

    // Shell count eligible for injection (for the occupancy estimate
    // and the log line).
    m_n_shells_eligible = m_inj_shells;
    if (m_inj_r_max > Scalar(0)) {
      m_n_shells_eligible = 0;
      for (int k = 0; k < m_mesh.m_N_r; k++) {
        Scalar r_c = Scalar(0.5) * (m_mesh.radii[k] + m_mesh.radii[k + 1]);
        if (r_c < m_inj_r_max) m_n_shells_eligible++;
      }
    }

    // The injector fetches "particles" and "rng_states", registered by
    // prismatic_ptc_updater — construct here, after all systems have
    // registered their data.
    m_injector =
        std::make_unique<prismatic_ptc_injector<ExecPolicy>>(m_mesh);
    sim_env().get_data("particles", m_ptc);
    sim_env().get_data("E", m_E);
    sim_env().get_data("B", m_B);

    if (m_inj_r_max > Scalar(0)) {
      Logger::print_info(
          "Volumetric injector: {} pairs/cell for r < {} ({} shells) every "
          "{} step(s), E.B threshold {}, kT = {}, weight = {}",
          m_pairs_per_cell, m_inj_r_max, m_n_shells_eligible, m_interval,
          m_eb_threshold, m_kT, m_weight);
    } else {
      Logger::print_info(
          "Surface injector: {} pairs/cell in {} shell(s) every {} step(s), "
          "kT = {}, weight = {}",
          m_pairs_per_cell, m_inj_shells, m_interval, m_kT, m_weight);
    }
  }

  void update(double dt, uint32_t step) override {
    if (m_interval <= 0 || step % m_interval != 0) return;

    // Occupancy throttle: stay clear of the buffer end so the updater
    // and sort always have room.
    size_t expected = size_t(2) * m_pairs_per_cell * m_mesh.m_N_tri *
                      m_n_shells_eligible;
    if (m_ptc->number() + expected >
        size_t(m_buffer_frac * m_ptc->size())) {
      if (!m_throttled) {
        Logger::print_info(
            "Surface injector throttled at {} particles (buffer frac {})",
            m_ptc->number(), m_buffer_frac);
        m_throttled = true;
      }
      return;
    }
    m_throttled = false;

    int inj_shells = m_inj_shells;
    int pairs = m_pairs_per_cell;
    Scalar kT = m_kT;
    Scalar weight = m_weight;
    Scalar eb_thr = m_eb_threshold;
    Scalar inj_r_max = m_inj_r_max;
    const Scalar* E_e = m_E->data().dev_ptr() != nullptr
                            ? m_E->data().dev_ptr()
                            : m_E->host_ptr();
    const Scalar* B_f = m_B->data().dev_ptr() != nullptr
                            ? m_B->data().dev_ptr()
                            : m_B->host_ptr();

    m_injector->inject_pairs(
        // criteria: surface mode (first inj_shells layers) or
        // volumetric mode (cell center below inj_r_max); with a
        // positive inj_eb_threshold additionally require unscreened
        // E_par at the prism center, |E.B| > threshold * |B|^2.
        [inj_shells, inj_r_max, eb_thr, E_e, B_f] LAMBDA(int tri, int k,
                                                         const auto& mp) {
          if (inj_r_max > Scalar(0)) {
            Scalar r_c = Scalar(0.5) * (mp.radii[k] + mp.radii[k + 1]);
            if (r_c >= inj_r_max) return false;
          } else {
            if (k >= inj_shells) return false;
          }
          if (eb_thr <= Scalar(0)) return true;
          Scalar l[3] = {Scalar(1.0 / 3), Scalar(1.0 / 3), Scalar(1.0 / 3)};
          Scalar Ex, Ey, Ez, Bx, By, Bz;
          interpolate_fields(mp, tri, k, l, Scalar(0.5), E_e, B_f,
                             Ex, Ey, Ez, Bx, By, Bz);
          Scalar EdotB = Ex * Bx + Ey * By + Ez * Bz;
          Scalar B2 = Bx * Bx + By * By + Bz * Bz;
          return math::abs(EdotB) > eb_thr * B2;
        },
        // number per cell (particles, not pairs)
        [pairs] LAMBDA(int tri, int k, const auto& mp) {
          return 2 * pairs;
        },
        // momentum: isotropic Maxwell-Juttner at kT
        [kT] LAMBDA(auto& x_global, auto& state, PtcType type) {
          return rng_maxwell_juttner_3d<Scalar>(state, kT);
        },
        // weight: uniform
        [weight] LAMBDA(auto& x_global, PtcType type) { return weight; });
  }

 private:
  const prismatic_mesh& m_mesh;
  std::unique_ptr<prismatic_ptc_injector<ExecPolicy>> m_injector;
  nonown_ptr<prismatic_particle_data> m_ptc;
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;

  int m_inj_shells = 1;
  int m_n_shells_eligible = 1;
  int m_pairs_per_cell = 1;
  int m_interval = 1;
  Scalar m_weight = Scalar(1);
  Scalar m_kT = Scalar(0.1);
  Scalar m_buffer_frac = Scalar(0.9);
  Scalar m_eb_threshold = Scalar(0);
  Scalar m_inj_r_max = Scalar(0);
  bool m_throttled = false;
};

using prismatic_surface_injector_t =
    prismatic_surface_injector<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
