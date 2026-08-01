#pragma once

#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_ptc_update_kernel.hpp"
#include "core/particles_functions.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <chrono>
#include <cmath>

namespace Aperture {

template <typename ExecPolicy>
prismatic_ptc_updater<ExecPolicy>::prismatic_ptc_updater(
    prismatic_mesh& mesh, const prismatic_mesh_partition* mp,
    const prismatic_mpi_comm* comm)
    : m_mesh(mesh), m_mp(mp), m_comm(comm) {
  m_distributed = mp != nullptr && comm != nullptr && !comm->is_single_rank();
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  if (m_distributed) {
    // Local solver-shared fields: totals in, deposits out.  Idempotent
    // with the solver's registrations (same names, same bundle).
    m_E = sim_env().template register_data<prismatic_edge_field>(
        "E", *m_mp, mem);
    m_B = sim_env().template register_data<prismatic_face_field>(
        "B", *m_mp, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>(
        "J", *m_mp, mem);
    m_rho = sim_env().template register_data<prismatic_vertex_field>(
        "rho", *m_mp, mem);
    m_rho_abs = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs", *m_mp, mem);
    m_gamma_wsum = sim_env().template register_data<prismatic_vertex_field>(
        "gamma_wsum", *m_mp, mem);
  } else {
    m_E = sim_env().template register_data<prismatic_edge_field>("E", m_mesh, mem);
    m_B = sim_env().template register_data<prismatic_face_field>("B", m_mesh, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>("J", m_mesh, mem);
    m_rho = sim_env().template register_data<prismatic_vertex_field>("rho", m_mesh, mem);
    m_rho_abs = sim_env().template register_data<prismatic_vertex_field>(
        "rho_abs", m_mesh, mem);
    m_gamma_wsum = sim_env().template register_data<prismatic_vertex_field>(
        "gamma_wsum", m_mesh, mem);
  }
  m_J->set_edge_kind(EdgeCochainKind::dual_2);

  int max_ptc = 100000;
  sim_env().params().get_value("max_ptc_num", max_ptc);
  m_ptc = sim_env().template register_data<prismatic_particle_data>(
      "particles", max_ptc, mem);

  size_t seed = default_random_seed;
  sim_env().params().get_value("random_seed", seed);
  m_rng_states =
      sim_env()
          .template register_data<rng_states_t<typename ExecPolicy::exec_tag>>(
              "rng_states", seed);
  m_rng_states->skip_output(true);
  m_rng_states->include_in_snapshot(true);

  // -----------------------------------------------------------------------
  // The local particle mesh is built HERE (registration time), not in
  // init(): the injector systems run their init() BEFORE ours (they are
  // registered first so injection precedes the push in a step) and need
  // ptc_mesh() there.
  // Distributed: the shared pic-depth bundle (canonical comm required).
  // Single-rank: an identity bundle — identity tables and maps,
  // bit-exact with the old global path.
  // -----------------------------------------------------------------------
  sim_env().params().get_value("use_recovery_gather", m_use_recovery_gather);
  if (m_use_recovery_gather) {
    // Weights are per SPHERE vertex (replicated data) — built globally,
    // then remapped into the local mesh below.
    m_recovery.build(m_mesh);
  }
  const prismatic_mesh_partition* mp = m_mp;
  if (m_distributed) {
    if (!m_comm->canonical_rank_order()) {
      Logger::print_err(
          "prismatic_ptc_updater (7C) requires a canonical A*K comm "
          "(prismatic_mpi_comm::create(world, A, K))");
      std::abort();
    }
    if (m_mp->depth() != halo_depth::pic) {
      Logger::print_err(
          "prismatic_ptc_updater requires a pic-depth mesh_partition "
          "(prismatic_mesh_partition::build(part, topo, halo_depth::pic))");
      std::abort();
    }
  } else {
    m_topo_own = icosphere_topology::build_from_mesh(m_mesh);
    auto part = prismatic_partition::single_rank(m_mesh.m_L, m_mesh.m_N_r);
    part.set_topology(&m_topo_own);
    m_mp_own = prismatic_mesh_partition::build(part, m_topo_own);
    mp = &m_mp_own;
  }
  m_lmesh.build(m_mesh, *mp,
                m_use_recovery_gather ? &m_recovery : nullptr,
                ExecPolicy::data_mem_type());
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_lmesh.copy_to_device();
#endif
  if (m_use_recovery_gather) {
    m_Bv.set_memtype(ExecPolicy::data_mem_type());
    m_Bv.resize(size_t(3) * m_lmesh.host_ptrs().N_verts);
    m_Bv.assign(Scalar(0));
  }
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::init() {
  sim_env().params().get_value("q_e", m_charge_e);
  sim_env().params().get_value("m_e", m_mass_e);
  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("use_gca", m_use_gca);
  sim_env().params().get_value("include_curvature", m_include_curvature);
  sim_env().params().get_value("gca_zero_mu_on_capture", m_gca_zero_mu);
  sim_env().params().get_value("ptc_absorb_radius", m_absorb_radius);
  sim_env().params().get_value("deposit_diagnostics", m_deposit_diagnostics);
  sim_env().params().get_value("step_timer_interval", m_timer_interval);
  if (m_absorb_radius > Scalar(0)) {
    Logger::print_info("Particle absorption radius: {}", m_absorb_radius);
  }
  init_gca_switch();
  init_sync_cooling();

  if (m_distributed) {
    bool halo_device_direct = true;
    sim_env().params().get_value("halo_device_direct", halo_device_direct);
    m_ex.init(*m_mp, *m_comm, halo_device_direct);
    m_world_rank = m_comm->world_rank();
    m_world_size = m_comm->world_size();

    m_mig_count.set_memtype(ExecPolicy::data_mem_type());
    m_mig_cursor.set_memtype(ExecPolicy::data_mem_type());
    m_mig_count.resize(m_world_size);
    m_mig_cursor.resize(m_world_size);
    m_mig_bad.set_memtype(ExecPolicy::data_mem_type());
    m_mig_bad.resize(1);
    m_mig_bad.assign(0);
    sim_env().params().get_value("ptc_misroute_tolerance",
                                 m_misroute_tolerance);
    // Non-zero initial capacity so host_ptr() is valid even on steps
    // with nothing to send (MPI gets zero counts but a real pointer).
    for (auto& b : m_snd_s) {
      b.set_memtype(ExecPolicy::data_mem_type());
      b.resize(16);
    }
    m_snd_cell.set_memtype(ExecPolicy::data_mem_type());
    m_snd_cell.resize(16);
    m_snd_flag.set_memtype(ExecPolicy::data_mem_type());
    m_snd_flag.resize(16);
    m_snd_id.set_memtype(ExecPolicy::data_mem_type());
    m_snd_id.resize(16);
    Logger::print_info(
        "Distributed particle updater: rank {} owns {} tris (of {} local), "
        "layers [{}, {}) at k0 = {}",
        m_world_rank, m_lmesh.n_tri_own(), m_lmesh.n_tri_local(),
        m_lmesh.lay_own_lo(), m_lmesh.lay_own_hi(), m_lmesh.k0());

    // 7D memory audit: the per-rank footprint must scale ~ 1/(A*K) plus
    // the O(4^L) replicated angular-table constant (weak-scaling smoke
    // checks this across A*K shapes).
    auto lp_sz = m_lmesh.host_ptrs();
    const size_t local_3d_bytes =
        sizeof(Scalar) * (size_t(m_E->data().size()) + m_B->data().size() +
                          m_J->data().size() + 3 * m_rho->data().size() +
                          m_Bv.size()) +
        sizeof(int) * (size_t(lp_sz.N_r + 1) *
                           (lp_sz.N_edge_s + lp_sz.N_tri + lp_sz.N_vert_s) +
                       size_t(lp_sz.N_r) * (lp_sz.N_vert_s + lp_sz.N_edge_s));
    const size_t angular_bytes =
        size_t(m_mesh.m_N_tri) * 3 * sizeof(int) * 4 +
        size_t(m_mesh.m_N_vert_s) * 5 * sizeof(Scalar) +
        size_t(m_mesh.m_N_tri) * sizeof(double) +
        size_t(m_mesh.m_N_edge_s) * (2 * sizeof(double) + 4 * sizeof(int)) +
        size_t(m_mesh.m_N_vert_s) * sizeof(double);
    Logger::print_info(
        "Per-rank footprint: local 3D (fields+maps+Bv) ~ {:.1f} kB, "
        "replicated angular tables ~ {:.1f} kB, particle buffer {:.1f} MB",
        local_3d_bytes / 1.0e3, angular_bytes / 1.0e3,
        double(m_ptc->size()) * (8 * sizeof(Scalar) + 2 * 4 + 8) / 1.0e6);
  }

  Logger::print_info("Prismatic particle updater initialized: {} particles",
                     m_ptc->size());
}

// ===========================================================================
// Hybrid-switch setup.
//
// The switch fires on the gyro-frequency omega_c / gamma = |q/m| B / gamma
// against gca_switch_omegac, a RATE in inverse time units.  It replaces
// gca_switch_omegac_dt, which compared omega_c dt / gamma to a fixed number
// and so moved the PHYSICAL switching surface every time dt changed: the
// same "0.1" meant 10.2 at L5, 20.4 at L6 and 40.7 at L7, and the L7
// discrepancies (Y-point jitter 21.3 deg vs L6's 12.0, +22% open flux, a
// spurious P/2 shedding cycle) were largely that artifact.  The L7 gca005
// A/B branch fixed it by hand -- 0.05 at L7 dt is exactly 0.1 at L6 dt,
// i.e. rate 20.37 -- and recovered the L6 surface.  20.0 is that value,
// rounded; it is now the same physical surface at every level.
//
// Boris still has to be able to integrate the gyrations it is handed, and
// that constraint DOES involve dt: at the switch a Boris particle sees
// omega_c dt / gamma = gca_switch_omegac * dt radians per step.  The old
// units capped this implicitly; the rate form does not, so it is checked
// here instead.
// ===========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::init_gca_switch() {
  if (sim_env().params().has("gca_switch_omegac_dt")) {
    double legacy = 0.0, dt = 0.0;
    sim_env().params().get_value("gca_switch_omegac_dt", legacy);
    sim_env().params().get_value("dt", dt);
    Logger::print_err(
        "gca_switch_omegac_dt has been removed.  It compared omega_c dt / "
        "gamma to a fixed number, so the physical switching surface moved "
        "with dt -- the same 0.1 meant rate 10.2 at L5, 20.4 at L6 and "
        "40.7 at L7.  Use gca_switch_omegac, a RATE in inverse time units "
        "(the criterion is now omega_c / gamma > gca_switch_omegac, with "
        "no dt in it).  Migrate: gca_switch_omegac = old value / dt = {} "
        "for this config, which reproduces its current surface exactly.  "
        "The unified production value anchored on the L6 baseline (and on "
        "the L7 gca005 A/B branch that restored it) is 20.0.",
        dt > 0.0 ? legacy / dt : 0.0);
    std::abort();
  }

  if (!sim_env().params().has("gca_switch_omegac")) {
    Logger::print_err(
        "use_gca requires gca_switch_omegac (the hybrid switch rate, in "
        "inverse time units): a particle is pushed by GCA while "
        "omega_c / gamma > gca_switch_omegac and by Boris below it.  The "
        "production value is 20.0.");
    std::abort();
  }
  double omegac = 0.0;
  sim_env().params().get_value("gca_switch_omegac", omegac);
  if (omegac <= 0.0) {
    Logger::print_err("gca_switch_omegac must be positive, got {}", omegac);
    std::abort();
  }
  m_gca_switch_omegac = Scalar(omegac);

  if (!m_use_gca) {
    Logger::print_info("Hybrid GCA switch: OFF (use_gca = false)");
    return;
  }

  double dt = 0.0;
  sim_env().params().get_value("dt", dt);
  const double wc_at_switch = omegac * dt;
  Logger::print_info(
      "Hybrid GCA switch: omega_c/gamma > {:.4g} (rate; dt-invariant), "
      "= {:.4f} rad/step at the switch ({:.1f} steps/gyration)",
      omegac, wc_at_switch,
      wc_at_switch > 0.0 ? 2.0 * M_PI / wc_at_switch : 0.0);

  // Boris resolution guard.  The switch rate is dt-invariant by design,
  // which means a coarse enough dt would silently hand Boris gyrations it
  // cannot integrate.  ~0.5 rad/step is ~12 steps/gyration; past that the
  // Boris side is garbage and the whole hybrid is meaningless.
  if (wc_at_switch > 0.5) {
    Logger::print_err(
        "gca_switch_omegac * dt = {:.3f} rad/step (only {:.1f} "
        "steps/gyration at the switch): Boris cannot resolve the gyrations "
        "it is being handed.  Reduce dt, or lower gca_switch_omegac (which "
        "moves the physical switching surface -- do not do this to silence "
        "the check).",
        wc_at_switch, 2.0 * M_PI / wc_at_switch);
    std::abort();
  } else if (wc_at_switch > 0.25) {
    Logger::print_info(
        "  WARNING: {:.3f} rad/step at the switch is only {:.1f} "
        "steps/gyration -- Boris is marginal here; prefer <= 0.25",
        wc_at_switch, 2.0 * M_PI / wc_at_switch);
  }
}

// ===========================================================================
// Synchrotron cooling setup (Landau-Lifshitz drag on the Boris branch).
//
// The knob is deliberately a RATE coefficient, never a per-step quantity:
// runbook pitfall #14 (and the dt-coupled GCA switch that motivated this
// work) both came from knobs denominated in steps or cells, which silently
// change meaning at a new resolution.  c_r here has the same meaning at
// every dt and level.
//
// Primary knob is sync_gamma_rad, the radiation-reaction-limited Lorentz
// factor: the gamma at which the drag balances acceleration by a
// reconnection field E ~ B_LC.  Ultrarelativistic, 90 deg pitch,
//   dgamma/dt = -c_r gamma^2 B^2   and   accel = |q/m| E,
// so  c_r gamma_rad^2 B_LC^2 = |q/m| B_LC  =>
//   c_r = |q/m| / (gamma_rad^2 B_LC).
// sync_cooling_coef, if present, sets c_r directly and overrides that.
// ===========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::init_sync_cooling() {
  bool use_cooling = false;
  sim_env().params().get_value("use_sync_cooling", use_cooling);
  if (!use_cooling) {
    m_sync_cool_coef = Scalar(0);
    Logger::print_info("Synchrotron cooling: OFF");
    return;
  }

  const double q_over_m = std::abs(double(m_charge_e) / double(m_mass_e));
  double b_lc = 0.0, gamma_rad = 0.0, coef = 0.0;
  sim_env().params().get_value("sync_cool_b_lc", b_lc);
  sim_env().params().get_value("sync_gamma_rad", gamma_rad);

  if (sim_env().params().has("sync_cooling_coef")) {
    sim_env().params().get_value("sync_cooling_coef", coef);
    if (coef <= 0.0) {
      Logger::print_err(
          "use_sync_cooling is on but sync_cooling_coef = {} is not "
          "positive.  Give a positive coefficient, or drop the key and set "
          "sync_gamma_rad + sync_cool_b_lc instead.",
          coef);
      std::abort();
    }
    // Back out the equivalent gamma_rad purely for the log below.
    if (b_lc > 0.0) gamma_rad = std::sqrt(q_over_m / (coef * b_lc));
  } else {
    if (gamma_rad <= 0.0 || b_lc <= 0.0) {
      Logger::print_err(
          "use_sync_cooling is on but the cooling strength is unset.  Set "
          "sync_gamma_rad (radiation-reaction-limited Lorentz factor) and "
          "sync_cool_b_lc (the field it is anchored to, e.g. B_LC = Bp / "
          "R_LC^3), or set sync_cooling_coef directly.  Got "
          "sync_gamma_rad = {}, sync_cool_b_lc = {}.",
          gamma_rad, b_lc);
      std::abort();
    }
    coef = q_over_m / (gamma_rad * gamma_rad * b_lc);
  }
  m_sync_cool_coef = Scalar(coef);

  // Report the implied cooling time so the regime is visible in the log:
  // t_cool(gamma, B) = 1 / (c_r gamma B^2), evaluated at the anchor field
  // and gamma_rad.  The "locked limit" this scheme targets wants that to
  // be short compared with the dynamical time.
  double dt = 0.0;
  sim_env().params().get_value("dt", dt);
  Logger::print_info(
      "Synchrotron cooling: ON (Landau-Lifshitz drag, Boris branch only), "
      "coef = {:.6e}",
      coef);
  if (b_lc > 0.0 && gamma_rad > 0.0) {
    const double t_cool = 1.0 / (coef * gamma_rad * b_lc * b_lc);
    Logger::print_info(
        "  gamma_rad = {:.4g} anchored at B = {:.4g}; "
        "t_cool(gamma_rad, B_LC) = {:.4e} = {:.2f} steps",
        gamma_rad, b_lc, t_cool, dt > 0.0 ? t_cool / dt : 0.0);
  }
}

// ===========================================================================
// Field sync point: E/B pic halos + the recovery Bv (owned-slot fit +
// 3-component vertex halo exchange).  Collective; idempotent per step.
// ===========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::sync_fields(uint32_t step) {
  if (step == m_synced_step) return;
  m_synced_step = step;
  const auto t0 = std::chrono::steady_clock::now();

  if (m_distributed) {
    m_ex.exchange_edge(m_E->data(), m_E->split());
    m_ex.exchange_face(m_B->data(), m_B->split());
  }

  if (m_use_recovery_gather) {
    auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});
    const int n_shell_own = lmp.shell_own_hi - lmp.shell_own_lo;
    const int n_own_slots = n_shell_own * lmp.n_vert_s_own;
    ExecPolicy::launch(
        [lmp, n_own_slots] LAMBDA(auto B_f, auto Bv) {
          ExecPolicy::loop(0, n_own_slots, [&] LAMBDA(int i) {
            const int k = lmp.shell_own_lo + i / lmp.n_vert_s_own;
            const int s = i % lmp.n_vert_s_own;
            compute_vertex_B_local(lmp, B_f, Bv, k, s);
          });
        },
        m_B->data(), m_Bv);
    ExecPolicy::sync();
    if (m_distributed) {
      const int stride = m_lmesh.host_ptrs().N_verts;
      for (int c = 0; c < 3; ++c) {
        m_ex.exchange_vertex(m_Bv, c * stride);
      }
    }
  }
  m_t_sync += std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::update(double dt, uint32_t step) {
  sync_fields(step);
  const auto t_push0 = std::chrono::steady_clock::now();

  auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});
  const int N_tri = lmp.N_tri;
  size_t num = m_ptc->number();
  Scalar charge_e = m_charge_e;
  Scalar mass_e = m_mass_e;

  // Clear rho, J, and the diagnostic deposits before deposit (full local
  // arrays — ghost slots absorb off-rank stencil ends until reduce()).
  ExecPolicy::launch(
      [N_edges = int(m_J->data().size()), N_verts = int(m_rho->data().size())]
      LAMBDA(auto J_e, auto rho, auto rho_abs, auto gamma_wsum) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          J_e[e] = Scalar(0);
        });
        ExecPolicy::loop(0, N_verts, [&] LAMBDA(int v) {
          rho[v] = Scalar(0);
          rho_abs[v] = Scalar(0);
          gamma_wsum[v] = Scalar(0);
        });
      },
      m_J->data(), m_rho->data(), m_rho_abs->data(), m_gamma_wsum->data());

  const Scalar* Bv_rec = nullptr;
  if (m_use_recovery_gather) {
    Bv_rec = m_Bv.dev_ptr() != nullptr ? m_Bv.dev_ptr() : m_Bv.host_ptr();
  }

  // Particle update loop
  bool use_gca = m_use_gca;
  bool include_curvature = m_include_curvature;
  Scalar absorb_r = m_absorb_radius;
  bool dep_diag = m_deposit_diagnostics;
  Scalar gca_omegac = m_gca_switch_omegac;
  bool gca_zero_mu = m_gca_zero_mu;
  Scalar sync_cool_coef = m_sync_cool_coef;
  ExecPolicy::launch(
      [num, N_tri, charge_e, mass_e, dt, lmp, use_gca, include_curvature,
       Bv_rec, absorb_r, dep_diag, gca_omegac, gca_zero_mu, sync_cool_coef]
      LAMBDA(auto ptc, auto E_e, auto B_f, auto J_e, auto rho,
             auto rho_abs, auto gamma_wsum) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int sp = get_ptc_type(ptc.flag[n]);
          Scalar q = (sp == (int)PtcType::positron) ? -charge_e : charge_e;
          update_single_particle(lmp, N_tri, ptc, n, E_e, B_f, J_e, rho,
                                 q, mass_e, Scalar(dt),
                                 use_gca, include_curvature, Bv_rec,
                                 absorb_r,
                                 dep_diag ? (Scalar*)rho_abs : nullptr,
                                 dep_diag ? (Scalar*)gamma_wsum : nullptr,
                                 gca_omegac, gca_zero_mu, sync_cool_coef);
        });
      },
      *m_ptc, m_E->data(), m_B->data(), m_J->data(), m_rho->data(),
      m_rho_abs->data(), m_gamma_wsum->data());

  ExecPolicy::sync();
  const auto t_push1 = std::chrono::steady_clock::now();
  m_t_push += std::chrono::duration<double>(t_push1 - t_push0).count();

  if (m_distributed) {
    // Fold ghost deposits into their owners (radial round first — the
    // corner relay), then refresh the ghost slots of the deposits the
    // next step's injector criteria read through their stencils.
    m_ex.reduce_edge(m_J->data(), m_J->split());
    m_ex.exchange_edge(m_J->data(), m_J->split());
    m_ex.reduce_vertex(m_rho->data());
    if (m_deposit_diagnostics) {
      m_ex.reduce_vertex(m_rho_abs->data());
      m_ex.exchange_vertex(m_rho_abs->data());
      m_ex.reduce_vertex(m_gamma_wsum->data());
    }
    const auto t_red1 = std::chrono::steady_clock::now();
    m_t_reduce += std::chrono::duration<double>(t_red1 - t_push1).count();

    migrate();
    m_t_migrate += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t_red1)
                       .count();
  }

  // Periodically sort particles by cell for GPU cache efficiency
  if (m_sort_interval > 0 && step % m_sort_interval == 0) {
    const auto t_s0 = std::chrono::steady_clock::now();
    size_t max_cell = m_lmesh.max_cell();
    ptc_sort_by_cell(typename ExecPolicy::exec_tag{}, *m_ptc, max_cell);
    m_t_sort += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_s0)
                    .count();
  }

  if (m_timer_interval > 0 && step > 0 && step % m_timer_interval == 0) {
    report_timers(step);
  }
}

// ===========================================================================
// 7E scaling harness: min/mean/max of the accumulated per-phase wall
// times across ranks, reported on logical rank 0 and reset.  Collective
// when distributed (the step condition is uniform across ranks).
// ===========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::report_timers(uint32_t step) {
  double loc[5] = {m_t_sync, m_t_push, m_t_reduce, m_t_migrate, m_t_sort};
  double mn[5], mx[5], sm[5];
  int ws = 1;
  if (m_distributed) {
    const MPI_Comm wcomm = m_comm->world();
    MPI_Reduce(loc, mn, 5, MPI_DOUBLE, MPI_MIN, 0, wcomm);
    MPI_Reduce(loc, mx, 5, MPI_DOUBLE, MPI_MAX, 0, wcomm);
    MPI_Reduce(loc, sm, 5, MPI_DOUBLE, MPI_SUM, 0, wcomm);
    ws = m_world_size;
    if (m_world_rank != 0) {
      m_t_sync = m_t_push = m_t_reduce = m_t_migrate = m_t_sort = 0;
      return;
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      mn[i] = mx[i] = sm[i] = loc[i];
    }
  }
  const char* names[5] = {"sync", "push", "reduce", "migrate", "sort"};
  const double per = double(m_timer_interval);
  for (int i = 0; i < 5; ++i) {
    Logger::print_info(
        "step timing [{}..{}] {:8s}: min {:.3f} / mean {:.3f} / max {:.3f} "
        "ms/step",
        step - m_timer_interval + 1, step, names[i], 1e3 * mn[i] / per,
        1e3 * sm[i] / (ws * per), 1e3 * mx[i] / per);
  }
  m_t_sync = m_t_push = m_t_reduce = m_t_migrate = m_t_sort = 0;
}

// =========================================================================
// Migration (plan F7): particles whose LOCAL cell left the owned region
// move to the owning world rank = rad·A + ang (angular rank from the
// per-tri table, radial from the global slab map).  The wire carries
// GLOBAL cell ids; pack translates local→global on the device, unpack
// global→local on the host via tri_g2l.  Device-side pack (two kernels:
// count, then place at per-destination cursors), host-staged
// MPI_Alltoallv per component, arrivals appended at the end of the
// particle array.  Sent particles become empty slots, compacted by the
// periodic sort.
// =========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::migrate() {
  const size_t num = m_ptc->number();
  const int ws = m_world_size;
  auto lmp = m_lmesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Pass 1: count leavers per destination.
  //
  // The `dest < ws` guard is not paranoia: migrate_dest indexes
  // tri_ang_rank[tri] and derives the radial slab from k0 + lay, so a
  // particle carrying a CORRUPT cell yields an out-of-range rank and
  // atomic_add(&count[dest]) then writes past m_mig_count — silent heap
  // corruption whose symptom appears arbitrarily far away.  Vacate such
  // particles here instead, where the cause is still local; the count is
  // reported by the caller.
  m_mig_count.assign(0);
  const int ws_guard = m_world_size;
  ExecPolicy::launch(
      [num, lmp, ws_guard] LAMBDA(auto ptc, auto count, auto bad) {
        ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
          if (ptc.cell[n] == empty_cell) return;
          int dest = lmp.migrate_dest(ptc.cell[n]);
          if (dest < 0) return;
          if (dest >= ws_guard) {
            atomic_add(&bad[0], 1);
            ptc.cell[n] = empty_cell;  // vacate; do not route on garbage
            return;
          }
          atomic_add(&count[dest], 1);
        });
      },
      *m_ptc, m_mig_count, m_mig_bad);
  ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_mig_bad.copy_to_host();
#endif
  if (m_mig_bad[0] > 0) {
    m_n_bad_dest += m_mig_bad[0];
    Logger::print_err_all(
        "migrate: {} particle(s) had a corrupt cell (destination rank >= "
        "world_size {}) and were vacated; {} cumulative.  This indicates a "
        "particle was created or moved into an invalid cell — check any "
        "source that writes ptc.cell (injector, pair producer, restart).",
        m_mig_bad[0], m_world_size, m_n_bad_dest);
    m_mig_bad.assign(0);
  }
  ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_mig_count.copy_to_host();
#endif

  // Host: exclusive scan -> send offsets.  (The count exchange lives in
  // exchange_wire, shared with the restart path.)
  std::vector<int> snd_cnt(ws), snd_off(ws);
  int n_send = 0;
  for (int r = 0; r < ws; ++r) {
    snd_cnt[r] = m_mig_count[r];
    snd_off[r] = n_send;
    n_send += snd_cnt[r];
  }

  // Pass 2: pack leavers at per-destination cursors and vacate them.
  // Cells are translated to the GLOBAL wire encoding here.
  if (int(m_snd_cell.size()) < n_send) {
    const size_t cap = size_t(n_send) * 2;
    for (auto& b : m_snd_s) b.resize(cap);
    m_snd_cell.resize(cap);
    m_snd_flag.resize(cap);
    m_snd_id.resize(cap);
  }
  if (n_send > 0) {
    for (int r = 0; r < ws; ++r) m_mig_cursor[r] = snd_off[r];
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    m_mig_cursor.copy_to_device();
#endif
    ExecPolicy::launch(
        [num, lmp] LAMBDA(auto ptc, auto cursor, auto sx1, auto sx2,
                          auto sx3, auto sp1, auto sp2, auto sp3,
                          auto sE, auto sw, auto scell, auto sflag,
                          auto sid) {
          ExecPolicy::loop(0, (int)num, [&] LAMBDA(int n) {
            if (ptc.cell[n] == empty_cell) return;
            int dest = lmp.migrate_dest(ptc.cell[n]);
            if (dest < 0) return;
            int slot = atomic_add(&cursor[dest], 1);
            sx1[slot] = ptc.x1[n];
            sx2[slot] = ptc.x2[n];
            sx3[slot] = ptc.x3[n];
            sp1[slot] = ptc.p1[n];
            sp2[slot] = ptc.p2[n];
            sp3[slot] = ptc.p3[n];
            sE[slot] = ptc.E[n];
            sw[slot] = ptc.weight[n];
            scell[slot] = lmp.wire_cell(ptc.cell[n]);
            sflag[slot] = ptc.flag[n];
            sid[slot] = ptc.id[n];
            ptc.cell[n] = empty_cell;
          });
        },
        *m_ptc, m_mig_cursor, m_snd_s[0], m_snd_s[1], m_snd_s[2],
        m_snd_s[3], m_snd_s[4], m_snd_s[5], m_snd_s[6], m_snd_s[7],
        m_snd_cell, m_snd_flag, m_snd_id);
    ExecPolicy::sync();
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    for (auto& b : m_snd_s) b.copy_to_host(0, n_send);
    m_snd_cell.copy_to_host(0, n_send);
    m_snd_flag.copy_to_host(0, n_send);
    m_snd_id.copy_to_host(0, n_send);
#endif
  }

  // Exchange, then append the arrivals (shared with the restart path).
  const Scalar* comps[8] = {
      m_snd_s[0].host_ptr(), m_snd_s[1].host_ptr(), m_snd_s[2].host_ptr(),
      m_snd_s[3].host_ptr(), m_snd_s[4].host_ptr(), m_snd_s[5].host_ptr(),
      m_snd_s[6].host_ptr(), m_snd_s[7].host_ptr()};
  const int n_arrived = exchange_wire(comps, m_snd_cell.host_ptr(),
                                      m_snd_flag.host_ptr(),
                                      m_snd_id.host_ptr(), snd_cnt, snd_off);
  append_wire_arrivals(n_arrived);
}

// Component-wise Alltoallv of packed leavers into the m_rcv_* staging.
// Counts are identical across components; receive buffers are sized at
// least 1 so .data() is a real pointer under zero counts.
template <typename ExecPolicy>
int prismatic_ptc_updater<ExecPolicy>::exchange_wire(
    const Scalar* const comps[8], const uint64_t* cells,
    const uint32_t* flags, const uint64_t* ids,
    const std::vector<int>& snd_cnt, const std::vector<int>& snd_off) {
  const int ws = m_world_size;
  const MPI_Comm wcomm = m_comm->world();

  std::vector<int> rcv_cnt(ws), rcv_off(ws);
  MPI_Alltoall(const_cast<int*>(snd_cnt.data()), 1, MPI_INT, rcv_cnt.data(),
               1, MPI_INT, wcomm);
  int n_recv = 0;
  for (int r = 0; r < ws; ++r) {
    rcv_off[r] = n_recv;
    n_recv += rcv_cnt[r];
  }

  const int rcv_cap = n_recv > 0 ? n_recv : 1;
  m_rcv_cell.resize(rcv_cap);
  m_rcv_flag.resize(rcv_cap);
  for (auto& v : m_rcv_s) v.resize(rcv_cap);
  m_rcv_id.resize(rcv_cap);
  const MPI_Datatype st = mpi_scalar_type();
  for (int c = 0; c < 8; ++c) {
    MPI_Alltoallv(comps[c], snd_cnt.data(), snd_off.data(), st,
                  m_rcv_s[c].data(), rcv_cnt.data(), rcv_off.data(), st,
                  wcomm);
  }
  MPI_Alltoallv(cells, snd_cnt.data(), snd_off.data(), MPI_UINT64_T,
                m_rcv_cell.data(), rcv_cnt.data(), rcv_off.data(),
                MPI_UINT64_T, wcomm);
  MPI_Alltoallv(flags, snd_cnt.data(), snd_off.data(), MPI_UINT32_T,
                m_rcv_flag.data(), rcv_cnt.data(), rcv_off.data(),
                MPI_UINT32_T, wcomm);
  MPI_Alltoallv(ids, snd_cnt.data(), snd_off.data(), MPI_UINT64_T,
                m_rcv_id.data(), rcv_cnt.data(), rcv_off.data(),
                MPI_UINT64_T, wcomm);
  return n_recv;
}

// Append arrivals at the end of the particle array, translating the
// GLOBAL wire cells in m_rcv_cell to this rank's local encoding
// (host-side, per arrival — cheap at migration counts).
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::append_wire_arrivals(int n_recv) {
  if (n_recv == 0) return;
  const size_t num = m_ptc->number();

  if (num + n_recv > m_ptc->size()) {
    // print_err_all, not print_err: this fires on ONE rank and the
    // rank-0-only logger would swallow it, leaving a bare SIGABRT with
    // no message (which is exactly how the 2026-08-01 misroute had to be
    // diagnosed from a core dump).
    Logger::print_err_all(
        "prismatic_ptc_updater::append_wire_arrivals: particle buffer "
        "overflow ({} + {} arrivals > {}) — raise max_ptc_num or pull "
        "ptc_absorb_radius inward",
        num, n_recv, m_ptc->size());
    std::abort();
  }
  const auto& g2l = m_lmesh.tri_g2l();
  const uint64_t N_tri_glob = uint64_t(m_lmesh.n_tri_global());
  const int n_tri_loc = m_lmesh.n_tri_local();
  const int k0 = m_lmesh.k0();
  auto hp = m_ptc->get_host_ptrs();
  int n_bad = 0;
  for (int i = 0; i < n_recv; ++i) {
    const uint64_t wc = m_rcv_cell[i];
    const int glay = int(wc / N_tri_glob);
    const int gtri = int(wc % N_tri_glob);
    const int ltri = (gtri >= 0 && uint64_t(gtri) < N_tri_glob)
                         ? g2l[gtri] : -1;
    const int llay = glay - k0;
    if (ltri < 0 || llay < 0) {
      // NON-FATAL by design.  A misrouted arrival means some particle
      // carried a cell this rank cannot own — a correctness bug, but one
      // stray particle must not take down a 320-rank job mid-campaign.
      // Land it in an inert slot (the injector's convention; the periodic
      // sort compacts it) and account for it.  A SYSTEMATIC failure still
      // aborts via the tolerance below, so this cannot hide a real break.
      if (m_n_misrouted < 20) {
        Logger::print_err_all(
            "append_wire_arrivals: MISROUTED arrival — wire cell {} "
            "decodes to (glay {}, gtri {}); this rank has k0 {}, "
            "n_tri_local {}, N_tri_global {} -> (llay {}, ltri {}). "
            "Dropping the particle.",
            wc, glay, gtri, k0, n_tri_loc, N_tri_glob, llay, ltri);
      }
      hp.cell[num + i] = empty_cell;
      ++n_bad;
      continue;
    }
    // Local cells fit uint32 by the lmesh build guard.
    hp.cell[num + i] = uint32_t(llay * n_tri_loc + ltri);
  }
  if (n_bad > 0) {
    m_n_misrouted += n_bad;
    const uint64_t tol = uint64_t(m_misroute_tolerance);
    if (m_misroute_tolerance >= 0 && m_n_misrouted > tol) {
      Logger::print_err_all(
          "append_wire_arrivals: {} misrouted arrivals cumulative (> "
          "ptc_misroute_tolerance = {}).  This is systematic, not a "
          "stray — aborting.  Set ptc_misroute_tolerance = -1 to keep "
          "running while diagnosing.",
          m_n_misrouted, m_misroute_tolerance);
      std::abort();
    }
  }

  const Scalar* rs[8] = {m_rcv_s[0].data(), m_rcv_s[1].data(),
                         m_rcv_s[2].data(), m_rcv_s[3].data(),
                         m_rcv_s[4].data(), m_rcv_s[5].data(),
                         m_rcv_s[6].data(), m_rcv_s[7].data()};
  Scalar* ds[8] = {hp.x1, hp.x2, hp.x3, hp.p1, hp.p2, hp.p3, hp.E,
                   hp.weight};
  for (int c = 0; c < 8; ++c) {
    std::copy(rs[c], rs[c] + n_recv, ds[c] + num);
  }
  std::copy(m_rcv_flag.begin(), m_rcv_flag.begin() + n_recv, hp.flag + num);
  std::copy(m_rcv_id.begin(), m_rcv_id.begin() + n_recv, hp.id + num);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_ptc->x1.copy_to_device(num, n_recv);
  m_ptc->x2.copy_to_device(num, n_recv);
  m_ptc->x3.copy_to_device(num, n_recv);
  m_ptc->p1.copy_to_device(num, n_recv);
  m_ptc->p2.copy_to_device(num, n_recv);
  m_ptc->p3.copy_to_device(num, n_recv);
  m_ptc->E.copy_to_device(num, n_recv);
  m_ptc->weight.copy_to_device(num, n_recv);
  m_ptc->cell.copy_to_device(num, n_recv);
  m_ptc->flag.copy_to_device(num, n_recv);
  m_ptc->id.copy_to_device(num, n_recv);
#endif
  m_ptc->add_num(n_recv);
}

// =========================================================================
// Restart support (checkpoint plan D3): arbitrary-distribution particle
// load.  Every rank holds SOME chunk of the checkpoint's concatenated
// particle datasets (global wire cells); destinations are computed by
// pure arithmetic on the global cell — the canonical path-ordered
// angular rank of the tri's unit × the uniform radial slab map — which
// is exactly migrate_dest's math applied to global ids, valid for ANY
// current decomposition.  The exchange and append reuse the migration
// machinery verbatim.
// =========================================================================
template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::inject_wire_particles(
    const std::vector<Scalar> comps[8], const std::vector<uint64_t>& gcells,
    const std::vector<uint32_t>& flags, const std::vector<uint64_t>& ids) {
  const size_t n = gcells.size();

  if (!m_distributed) {
    // Single-rank: the local mesh is the identity bundle — wire cells
    // ARE local cells; stage into m_rcv_* and append.
    const int nn = int(n);
    const int cap = nn > 0 ? nn : 1;
    for (auto& v : m_rcv_s) v.resize(cap);
    m_rcv_cell.resize(cap);
    m_rcv_flag.resize(cap);
    m_rcv_id.resize(cap);
    for (int c = 0; c < 8; ++c) {
      std::copy(comps[c].begin(), comps[c].end(), m_rcv_s[c].begin());
    }
    std::copy(gcells.begin(), gcells.end(), m_rcv_cell.begin());
    std::copy(flags.begin(), flags.end(), m_rcv_flag.begin());
    std::copy(ids.begin(), ids.end(), m_rcv_id.begin());
    append_wire_arrivals(nn);
    return;
  }

  // Destination world rank from the GLOBAL cell.
  const auto& part = m_mp->partition();
  const int N_tri_glob = m_lmesh.n_tri_global();
  const int A = m_lmesh.n_angular_ranks();
  const int slab_base = m_lmesh.slab_base();
  const int slab_rem = m_lmesh.slab_rem();
  const int split = slab_rem * (slab_base + 1);
  auto dest_of = [&](uint64_t gcell) -> int {
    const int glay = int(gcell / uint64_t(N_tri_glob));
    const int gtri = int(gcell % uint64_t(N_tri_glob));
    const int ang =
        part.angular_rank_of_path_unit(part.path_of_unit(part.unit_of_tri(gtri)));
    const int rad = glay < split ? glay / (slab_base + 1)
                                 : slab_rem + (glay - split) / slab_base;
    return rad * A + ang;
  };

  const int ws = m_world_size;
  std::vector<int> snd_cnt(ws, 0), snd_off(ws, 0);
  for (size_t i = 0; i < n; ++i) snd_cnt[dest_of(gcells[i])]++;
  int n_send = 0;
  for (int r = 0; r < ws; ++r) {
    snd_off[r] = n_send;
    n_send += snd_cnt[r];
  }

  // Pack per destination (host — the data just came off the disk).
  const int cap = n_send > 0 ? n_send : 1;
  std::vector<Scalar> snd_s[8];
  for (auto& v : snd_s) v.resize(cap);
  std::vector<uint64_t> snd_cell(cap), snd_id(cap);
  std::vector<uint32_t> snd_flag(cap);
  std::vector<int> cursor(snd_off);
  for (size_t i = 0; i < n; ++i) {
    const int slot = cursor[dest_of(gcells[i])]++;
    for (int c = 0; c < 8; ++c) snd_s[c][slot] = comps[c][i];
    snd_cell[slot] = gcells[i];
    snd_flag[slot] = flags[i];
    snd_id[slot] = ids[i];
  }

  const Scalar* sc[8] = {snd_s[0].data(), snd_s[1].data(), snd_s[2].data(),
                         snd_s[3].data(), snd_s[4].data(), snd_s[5].data(),
                         snd_s[6].data(), snd_s[7].data()};
  const int n_arrived = exchange_wire(sc, snd_cell.data(), snd_flag.data(),
                                      snd_id.data(), snd_cnt, snd_off);
  append_wire_arrivals(n_arrived);
}

template <typename ExecPolicy>
void prismatic_ptc_updater<ExecPolicy>::refresh_deposit_ghosts() {
  if (!m_distributed) return;
  m_ex.exchange_edge(m_J->data(), m_J->split());
  m_ex.exchange_vertex(m_rho->data());
  m_ex.exchange_vertex(m_rho_abs->data());
  m_ex.exchange_vertex(m_gamma_wsum->data());
}

template <typename ExecPolicy>
int prismatic_ptc_updater<ExecPolicy>::add_particle(
    Scalar x, Scalar y, Scalar z, Scalar px, Scalar py, Scalar pz,
    Scalar weight, uint32_t flag) {
  auto lmp = m_lmesh.host_ptrs();
  int tri_idx, layer_idx;
  Scalar l1, l2, zeta;
  if (!cartesian_to_local_impl(lmp, x, y, z, tri_idx, layer_idx, l1, l2,
                               zeta))
    return -1;
  // Only the owner of the target cell creates the particle (callers add
  // globally; each rank keeps its own).  Single-rank owns everything.
  if (!lmp.owns_cell(tri_idx, layer_idx)) return -1;

  size_t idx = m_ptc->number();
  if (idx >= m_ptc->size()) return -1;

  auto ptrs = m_ptc->get_host_ptrs();
  ptrs.x1[idx] = l1; ptrs.x2[idx] = l2; ptrs.x3[idx] = zeta;
  ptrs.p1[idx] = px; ptrs.p2[idx] = py; ptrs.p3[idx] = pz;
  ptrs.E[idx] = std::sqrt(Scalar(1) + px*px + py*py + pz*pz);
  ptrs.weight[idx] = weight;
  ptrs.cell[idx] = uint32_t(layer_idx * lmp.N_tri + tri_idx);
  ptrs.flag[idx] = flag;
  ptrs.id[idx] = idx;
  m_ptc->set_num(idx + 1);
  return static_cast<int>(idx);
}

}  // namespace Aperture
