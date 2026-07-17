#pragma once

#include "core/random.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_injector.hpp"
#include "utils/logger.h"
#include <memory>

namespace Aperture {

// Surface pair injection for magnetosphere runs: every inj_interval
// steps, inject inj_pairs_per_cell neutral e+/e- pairs, uniformly
// placed, in every prism of the first inj_shells radial layers, with an
// isotropic Maxwell-Juttner momentum spread of temperature inj_kT and
// macro-weight inj_weight.
//
// This is the "star surface" scheme of aligned-rotator PIC studies: the
// injected plasma is neutral; the corotation E field set by the inner
// BC separates charge and fills the magnetosphere toward the GJ state.
// Throttling is by total buffer occupancy only (injection stops above
// inj_buffer_frac of max_ptc_num) — density- or sigma-based criteria
// can replace the f_criteria functor later without touching the driver.
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

    // The injector fetches "particles" and "rng_states", registered by
    // prismatic_ptc_updater — construct here, after all systems have
    // registered their data.
    m_injector =
        std::make_unique<prismatic_ptc_injector<ExecPolicy>>(m_mesh);
    sim_env().get_data("particles", m_ptc);

    Logger::print_info(
        "Surface injector: {} pairs/cell in {} shell(s) every {} step(s), "
        "kT = {}, weight = {}",
        m_pairs_per_cell, m_inj_shells, m_interval, m_kT, m_weight);
  }

  void update(double dt, uint32_t step) override {
    if (m_interval <= 0 || step % m_interval != 0) return;

    // Occupancy throttle: stay clear of the buffer end so the updater
    // and sort always have room.
    size_t expected = size_t(2) * m_pairs_per_cell * m_mesh.m_N_tri *
                      m_inj_shells;
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

    m_injector->inject_pairs(
        // criteria: the first inj_shells radial layers
        [inj_shells] LAMBDA(int tri, int k, const auto& mp) {
          return k < inj_shells;
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

  int m_inj_shells = 1;
  int m_pairs_per_cell = 1;
  int m_interval = 1;
  Scalar m_weight = Scalar(1);
  Scalar m_kT = Scalar(0.1);
  Scalar m_buffer_frac = Scalar(0.9);
  bool m_throttled = false;
};

using prismatic_surface_injector_t =
    prismatic_surface_injector<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
