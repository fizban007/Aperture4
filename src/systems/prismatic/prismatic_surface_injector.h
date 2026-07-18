#pragma once

#include "core/random.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_deposit.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_particles.h"
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
    sim_env().params().get_value("inj_gca", m_inj_gca);
    if (m_inj_gca) {
      Logger::print_info(
          "GCA-native injection: mu = 0 (synchrotron-locked), "
          "p1 = u_par ~ MJ({})", m_kT);
    }

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
    sim_env().params().get_value("inj_max_multiplicity", m_max_multiplicity);
    sim_env().params().get_value("inj_min_sigma", m_min_sigma);
    Scalar q_e = 1, m_e = 1;
    sim_env().params().get_value("q_e", q_e);
    sim_env().params().get_value("m_e", m_e);
    m_m_over_q = m_e / q_e;
    if (m_min_sigma > Scalar(0)) {
      Logger::print_info(
          "Injector cold-sigma floor: no injection below sigma = {}",
          m_min_sigma);
    }
    sim_env().get_data_optional("rho_abs", m_rho_abs);
    sim_env().get_data("J", m_Jf);
    if ((m_max_multiplicity > Scalar(0) || m_min_sigma > Scalar(0)) &&
        m_rho_abs != nullptr) {
      m_J_primal.set_memtype(ExecPolicy::data_mem_type());
      m_J_primal.resize(m_mesh.m_N_edges);
    }
    if (m_max_multiplicity > Scalar(0) && m_rho_abs != nullptr) {
      Logger::print_info(
          "Injector multiplicity cutoff: M = rho_abs c/|J| > {} stops "
          "injection", m_max_multiplicity);
    }

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

    // Multiplicity cutoff (J-guided, per the 2014-style demand logic):
    // M = rho_abs c / |J| compares the available charge carriers to the
    // minimum needed to carry the local current at c.  Small |J| with
    // even a modest density means a huge multiplicity -> plasma is
    // sufficient -> stop injecting there.  Uses last step's deposits.
    Scalar max_mult = (m_rho_abs != nullptr) ? m_max_multiplicity : Scalar(0);
    Scalar min_sigma = (m_rho_abs != nullptr) ? m_min_sigma : Scalar(0);
    Scalar m_over_q = m_m_over_q;
    const Scalar* J_p = nullptr;
    const Scalar* rho_abs = nullptr;
    if (max_mult > Scalar(0) || min_sigma > Scalar(0)) {
      auto mp_conv = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
      bool j_dual = (m_Jf->edge_kind() == EdgeCochainKind::dual_2);
      ExecPolicy::launch(
          [N_edges = m_mesh.m_N_edges, mp_conv, j_dual]
          LAMBDA(auto J_raw, auto J_out) {
            ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
              J_out[e] = j_dual ? mp_conv.hodge1_inv[e] * J_raw[e]
                                : J_raw[e];
            });
          },
          m_Jf->data(), m_J_primal);
      ExecPolicy::sync();
      J_p = (m_J_primal.dev_ptr() != nullptr) ? m_J_primal.dev_ptr()
                                              : m_J_primal.host_ptr();
      rho_abs = (m_rho_abs->data().dev_ptr() != nullptr)
                    ? m_rho_abs->data().dev_ptr()
                    : m_rho_abs->host_ptr();
    }

    m_injector->inject_pairs(
        // criteria: surface mode (first inj_shells layers) or
        // volumetric mode (cell center below inj_r_max); with a
        // positive inj_eb_threshold additionally require unscreened
        // E_par at the prism center, |E.B| > threshold * |B|^2.
        [inj_shells, inj_r_max, eb_thr, max_mult, min_sigma, m_over_q,
         E_e, B_f, J_p, rho_abs]
        LAMBDA(int tri, int k, const auto& mp) {
          if (inj_r_max > Scalar(0)) {
            Scalar r_c = Scalar(0.5) * (mp.radii[k] + mp.radii[k + 1]);
            if (r_c >= inj_r_max) return false;
          } else {
            if (k >= inj_shells) return false;
          }
          Scalar l[3] = {Scalar(1.0 / 3), Scalar(1.0 / 3), Scalar(1.0 / 3)};
          Scalar Ex, Ey, Ez, Bx, By, Bz, B2 = 0;
          bool need_fields = eb_thr > Scalar(0) || max_mult > Scalar(0) ||
                             min_sigma > Scalar(0);
          if (need_fields) {
            interpolate_fields(mp, tri, k, l, Scalar(0.5), E_e, B_f,
                               Ex, Ey, Ez, Bx, By, Bz);
            B2 = Bx * Bx + By * By + Bz * Bz;
          }
          if (eb_thr > Scalar(0)) {
            Scalar EdotB = Ex * Bx + Ey * By + Ez * Bz;
            if (math::abs(EdotB) <= eb_thr * B2) return false;
          }
          if (max_mult > Scalar(0) || min_sigma > Scalar(0)) {
            // rho_abs density at the prism center: hat weights are all
            // 1/6 at (1/3, 1/3, 1/3; zeta = 1/2).
            Scalar ra = 0;
            for (int vi = 0; vi < 3; vi++) {
              int sv = mp.tri_verts[tri * 3 + vi];
              int vb = k * mp.N_vert_s + sv;
              int vt = vb + mp.N_vert_s;
              ra += (Scalar(1.0 / 6)) *
                    (mp.vert_dual_vol[vb] > 0
                         ? rho_abs[vb] / mp.vert_dual_vol[vb] : Scalar(0));
              ra += (Scalar(1.0 / 6)) *
                    (mp.vert_dual_vol[vt] > 0
                         ? rho_abs[vt] / mp.vert_dual_vol[vt] : Scalar(0));
            }
            // Cold-magnetization floor: sigma = B^2 / (rho_abs m/q).
            // No injection into plasma that is already inertially
            // loaded — this is what the multiplicity criterion cannot
            // see in low-J regions filled by transport.
            if (min_sigma > Scalar(0) &&
                B2 < min_sigma * ra * m_over_q) return false;
            if (max_mult > Scalar(0)) {
              Scalar Jx, Jy, Jz, bx, by, bz;
              interpolate_fields(mp, tri, k, l, Scalar(0.5), J_p, B_f,
                                 Jx, Jy, Jz, bx, by, bz);
              Scalar Jmag = math::sqrt(Jx * Jx + Jy * Jy + Jz * Jz);
              if (ra > max_mult * Jmag) return false;
            }
          }
          return true;
        },
        // number per cell (particles, not pairs)
        [pairs] LAMBDA(int tri, int k, const auto& mp) {
          return 2 * pairs;
        },
        // momentum: isotropic Maxwell-Juttner at kT; in GCA-native
        // mode (inj_gca) the particle is born on the lowest Landau
        // level: mu = 0, u_par a signed 1D MJ magnitude.
        [kT, inj_gca = m_inj_gca] LAMBDA(auto& x_global, auto& state,
                                         PtcType type) {
          if (inj_gca) {
            Scalar u = rng_maxwell_juttner<Scalar>(state, kT);
            if (rng_uniform<Scalar>(state) < Scalar(0.5)) u = -u;
            return vec_t<Scalar, 3>(u, Scalar(0), Scalar(0));
          }
          return rng_maxwell_juttner_3d<Scalar>(state, kT);
        },
        // weight: uniform
        [weight] LAMBDA(auto& x_global, PtcType type) { return weight; },
        m_inj_gca ? [] { uint32_t f = 0;
                         set_flag(f, PtcFlagEx::gca_state);
                         return f; }()
                  : uint32_t(0));
  }

 private:
  const prismatic_mesh& m_mesh;
  std::unique_ptr<prismatic_ptc_injector<ExecPolicy>> m_injector;
  nonown_ptr<prismatic_particle_data> m_ptc;
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_Jf;
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  buffer<Scalar> m_J_primal;

  int m_inj_shells = 1;
  int m_n_shells_eligible = 1;
  int m_pairs_per_cell = 1;
  int m_interval = 1;
  Scalar m_weight = Scalar(1);
  Scalar m_kT = Scalar(0.1);
  Scalar m_buffer_frac = Scalar(0.9);
  Scalar m_eb_threshold = Scalar(0);
  Scalar m_inj_r_max = Scalar(0);
  // Stop injecting where rho_abs c / |J| exceeds this (0 disables).
  Scalar m_max_multiplicity = Scalar(0);
  // Cold-magnetization floor: no injection where B^2/(rho_abs m/q)
  // is already below this (0 disables).
  Scalar m_min_sigma = Scalar(0);
  Scalar m_m_over_q = Scalar(1);
  // Inject in the GCA representation (u_par, mu = 0) with the
  // gca_state flag set.
  bool m_inj_gca = false;
  bool m_throttled = false;
};

using prismatic_surface_injector_t =
    prismatic_surface_injector<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
