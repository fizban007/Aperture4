#pragma once

#include "systems/prismatic/cavity_modes.hpp"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "framework/environment.h"
#include "utils/gauss_quadrature.h"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace Aperture {

// Spherical-geometry quadrature helpers and analytic dipole / Deutsch
// evaluators live in dec_solver_geometry.hpp (shared with the
// distributed solver core).
// =========================================================================
// Constructor and init
// =========================================================================

template <typename ExecPolicy>
dec_field_solver<ExecPolicy>::dec_field_solver(
    prismatic_mesh& mesh, const prismatic_mpi_comm* comm,
    const prismatic_mesh_partition* mp_ext)
    : m_mesh(mesh) {
  if (comm != nullptr && !comm->is_single_rank()) {
    // Distributed mode must be established HERE: the framework calls
    // register_data_components right after construction, and the field
    // buffers are sized from this partition.
    m_mpi = comm;
    m_distributed = true;
    if (mp_ext != nullptr) {
      // Phase 7C: share the main-built (pic-depth) bundle with the
      // particle systems — one partition, one set of layouts.  The
      // bundle's topology lives in main and must outlive us.
      m_mesh_part = *mp_ext;
      m_part = m_mesh_part.partition();
    } else {
      m_topo = icosphere_topology::build_from_mesh(m_mesh);
      // Canonical A·K comms (create(world, A, K)) get the generalized
      // path-ordered partition; the legacy 20·K identity comm keeps
      // the per-ico-face partition until its callers retire.
      m_part = comm->canonical_rank_order()
                   ? prismatic_partition::combined(
                         m_mesh.m_L, m_mesh.m_N_r, comm->n_angular_ranks(),
                         comm->n_radial_ranks(), comm->world_rank())
                   : prismatic_partition::combined_ico_face(
                         m_mesh.m_L, m_mesh.m_N_r, comm->n_radial_ranks(),
                         comm->radial_rank(), comm->angular_rank());
      m_part.set_topology(&m_topo);
      m_mesh_part = prismatic_mesh_partition::build(m_part, m_topo);
    }
  }
  int ne = mesh.m_N_edges, nf = mesh.m_N_faces;
  if (m_distributed) {
    ne = m_mesh_part.layout(cochain_type::h_edge).local_size() +
         m_mesh_part.layout(cochain_type::v_edge).local_size();
    nf = m_mesh_part.layout(cochain_type::tri_face).local_size() +
         m_mesh_part.layout(cochain_type::rect_face).local_size();
  }
  for (auto b : {&m_tmp_E, &m_dE_dt, &m_dE_dt_new}) {
    b->set_memtype(ExecPolicy::data_mem_type());
    b->resize(ne);
  }
  for (auto b : {&m_tmp_B, &m_dB_dt, &m_dB_dt_new}) {
    b->set_memtype(ExecPolicy::data_mem_type());
    b->resize(nf);
  }
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  if (m_distributed) {
    // LOCAL-sized fields (owned + ghost per cochain, 4.1b.6 part 2).
    // Combined-range consumers must not be registered in this mode.
    m_Etotal = sim_env().template register_data<prismatic_edge_field>(
        "E", m_mesh_part, mem);
    m_Btotal = sim_env().template register_data<prismatic_face_field>(
        "B", m_mesh_part, mem);
    m_E = sim_env().template register_data<prismatic_edge_field>(
        "Edelta", m_mesh_part, mem);
    m_B = sim_env().template register_data<prismatic_face_field>(
        "Bdelta", m_mesh_part, mem);
    m_E0 = sim_env().template register_data<prismatic_edge_field>(
        "E0", m_mesh_part, mem);
    m_B0 = sim_env().template register_data<prismatic_face_field>(
        "B0", m_mesh_part, mem);
    m_J = sim_env().template register_data<prismatic_edge_field>(
        "J", m_mesh_part, mem);
    m_J->set_edge_kind(EdgeCochainKind::dual_2);
    return;
  }
  // Totals (background + delta): what particles, sph output, and the
  // exporter consume.
  m_Etotal = sim_env().template register_data<prismatic_edge_field>(
      "E", m_mesh, mem);
  m_Btotal = sim_env().template register_data<prismatic_face_field>(
      "B", m_mesh, mem);
  // Evolved delta fields and the static background.
  m_E = sim_env().template register_data<prismatic_edge_field>(
      "Edelta", m_mesh, mem);
  m_B = sim_env().template register_data<prismatic_face_field>(
      "Bdelta", m_mesh, mem);
  m_E0 = sim_env().template register_data<prismatic_edge_field>(
      "E0", m_mesh, mem);
  m_B0 = sim_env().template register_data<prismatic_face_field>(
      "B0", m_mesh, mem);
  m_J = sim_env().template register_data<prismatic_edge_field>(
      "J", m_mesh, mem);
  // Ampere applies h1inv to (d1t h2 B - J): J[e] is the DUAL 2-cochain
  // (current through the dual face of edge e).  Tag it so consumers
  // (sph output, injector) apply the h1inv conversion before Whitney
  // interpolation.  The GR solver does the same.
  m_J->set_edge_kind(EdgeCochainKind::dual_2);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::init() {
  // 4.1b: build the distributed core over this rank's partition — the
  // real one from set_mpi under MPI, otherwise single_rank (identity
  // layouts onto the global ordering, registered field buffers pass
  // straight into the core's kernels).
  if (!m_distributed) {
    m_topo = icosphere_topology::build_from_mesh(m_mesh);
    m_part = prismatic_partition::single_rank(m_mesh.m_L, m_mesh.m_N_r);
    m_part.set_topology(&m_topo);
    m_mesh_part = prismatic_mesh_partition::build(m_part, m_topo);
  }
  m_dist.build(m_mesh, m_mesh_part);
  if (m_distributed) {
    // Packed device-direct exchange by default; "halo_device_direct =
    // false" falls back to the legacy full-buffer host-staged path
    // (wire-compatible, for debugging).
    bool halo_device_direct = true;
    sim_env().params().get_value("halo_device_direct", halo_device_direct);
    m_ex.init(m_mesh_part, *m_mpi, halo_device_direct);
    Logger::print_info("Halo exchange path: {}",
                       halo_device_direct
                           ? (mpi_gpu_direct_available()
                                  ? "packed, GPU-direct MPI"
                                  : "packed, host-staged messages")
                           : "full-buffer host staging");
    // Phase 6: particles are supported when the distributed particle
    // stack is registered (replicator + partition-aware updater; the
    // updater verifies its own requirements and aborts otherwise).
    Logger::print_info(
        "Distributed DEC solver: rank ({}, {}) of {}x{}, local edges {} / {} "
        "global, local faces {} / {}",
        m_mpi->angular_rank(), m_mpi->radial_rank(),
        m_mpi->n_angular_ranks(), m_mpi->n_radial_ranks(),
        m_dist.n_edges_local(), m_mesh.m_N_edges, m_dist.n_faces_local(),
        m_mesh.m_N_faces);
  }
  sim_env().params().get_value("rank_dump_interval", m_rank_dump_interval);
  sim_env().params().get_value("output_dir", m_output_dir);

  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Omega", m_Omega);
  sim_env().params().get_value("obliquity", m_obliquity);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
  sim_env().params().get_value("damping_exponent", m_damping_exponent);
  sim_env().params().get_value("update_e", m_update_e);
  sim_env().params().get_value("update_b", m_update_b);
  sim_env().params().get_value("use_implicit", m_use_implicit);
  sim_env().params().get_value("implicit_beta", m_beta);
  sim_env().params().get_value("implicit_iters", m_implicit_iters);
  sim_env().params().get_value("use_deutsch_bc", m_use_deutsch_bc);
  sim_env().params().get_value("use_pec_bc", m_use_pec_bc);
  sim_env().params().get_value("inner_bc_overwrite_b", m_inner_bc_overwrite_b);
  sim_env().params().get_value("use_reconstruction_hodge", m_use_recon_hodge);
  if (m_use_recon_hodge && m_distributed) {
    Logger::print_err(
        "use_reconstruction_hodge is single-rank only (global path); "
        "disabling");
    m_use_recon_hodge = false;
  }
  if (m_use_recon_hodge) {
    if (m_use_implicit) {
      Logger::print_err(
          "use_reconstruction_hodge requires explicit stepping; disabling");
      m_use_recon_hodge = false;
    } else {
      m_recon_hodge.build(m_mesh);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
      m_recon_hodge.copy_to_device();
#endif
    }
  }
  sim_env().params().get_value("resonator_amp", m_resonator_amp);

  // ---- Frame dragging ("fake GR") ----
  sim_env().params().get_value("use_frame_dragging", m_use_frame_drag);
  if (m_use_frame_drag) {
    if (m_use_deutsch_bc || m_use_pec_bc || m_use_recon_hodge) {
      Logger::print_err(
          "use_frame_dragging is incompatible with use_deutsch_bc (flat "
          "vacuum analytic BC), use_pec_bc, and use_reconstruction_hodge.");
      std::abort();
    }
    double compactness = 0.0, lt_frac = -1.0, r_star = 0.0;
    sim_env().params().get_value("gr_compactness", compactness);
    sim_env().params().get_value("gr_omega_lt_frac", lt_frac);
    sim_env().params().get_value("r_min", r_star);
    sim_env().params().get_value("gr_lt_exponent", m_lt_exponent);
    if (lt_frac < 0.0) {
      // Uniform-density stellar moment of inertia: omega_LT(R*) =
      // (2/5) (r_s/R*) Omega.
      lt_frac = 0.4 * compactness;
    }
    if (lt_frac <= 0.0 || lt_frac >= 1.0) {
      Logger::print_err(
          "use_frame_dragging is on but the drag strength is unset or "
          "unphysical: gr_omega_lt_frac = {} (derived from gr_compactness "
          "= {}).  Set gr_compactness (r_s/R*, e.g. 0.5) or "
          "gr_omega_lt_frac (omega_LT(R*)/Omega, e.g. 0.2) explicitly.",
          lt_frac, compactness);
      std::abort();
    }
    if (r_star <= 0.0) r_star = 1.0;
    m_gr_compactness = Scalar(compactness);
    m_omega_lt0 = Scalar(lt_frac) * m_Omega;
    m_lt_r_star = Scalar(r_star);
    // Lapse defaults ON with the shift (PCTS15 keep both).  Setting it
    // false gives the shift-only slow-rotation scheme, which is
    // self-consistent but drops a real O(30%) factor at the surface.
    m_use_lapse = true;
    sim_env().params().get_value("use_gr_lapse", m_use_lapse);
    if (m_use_lapse && compactness <= 0.0) {
      Logger::print_err(
          "use_gr_lapse is on but gr_compactness = {} (<= 0), so the lapse "
          "would be identically 1.  Set gr_compactness (r_s/R*), or set "
          "gr_omega_lt_frac directly and use_gr_lapse = false for a "
          "shift-only run.",
          compactness);
    }
  }

  sim_env().params().get_value("use_static_background", m_use_static_background);
  if (m_use_static_background) {
    if (m_use_pec_bc) {
      Logger::print_err(
          "use_static_background with use_pec_bc is unsupported (the PEC "
          "boundary acts on the delta fields only); disabling background");
      m_use_static_background = false;
    } else {
      // Aligned (static) dipole component only — see header note.
      fill_dipole_B(m_B0->data(), Scalar(0), Scalar(0),
                    m_Bp * std::cos(m_obliquity));
      // Owned slots only were filled; refresh B0 ghosts once so
      // refresh_total_fields is consistent on the full local range.
      m_ex.exchange_face(m_B0->data(), m_dist.b_split());
      Logger::print_info(
          "Static background enabled: aligned dipole mz = {}",
          m_Bp * std::cos(m_obliquity));
    }
  }
  refresh_total_fields();

  if (m_use_frame_drag) {
    m_dist.build_frame_drag(m_omega_lt0, m_lt_r_star, m_lt_exponent);
    if (m_use_lapse && m_gr_compactness > Scalar(0)) {
      m_dist.build_lapse(m_gr_compactness, m_lt_r_star);
    }
    m_Eeff.set_memtype(ExecPolicy::data_mem_type());
    m_Eeff.resize(m_dist.n_edges_local());
    m_Eeff.assign(Scalar(0));
    Logger::print_info(
        "Frame dragging ON: omega_LT(R*)/Omega = {:.4g} (compactness {}), "
        "profile (R*/r)^{}, R* = {}; surface rho_GJ reduced by {:.3g}",
        double(m_omega_lt0) / double(m_Omega), m_gr_compactness,
        m_lt_exponent, m_lt_r_star,
        double(m_omega_lt0) / double(m_Omega));
    Logger::print_info(
        "  shift convention beta = -v_LT (PCTS15 3+1): Faraday operand is "
        "alpha E - v_LT x B, particles get dx/dt = alpha v + v_LT");
    if (m_dist.lapse_built()) {
      Logger::print_info(
          "  lapse ON: alpha(r) = sqrt(1 - {}*R*/r), alpha(R*) = {:.4g}, "
          "alpha(R_LC) = {:.4g}",
          m_gr_compactness,
          double(gr_lapse(m_lt_r_star, m_gr_compactness, m_lt_r_star)),
          double(gr_lapse(Scalar(1.0 / m_Omega), m_gr_compactness,
                          m_lt_r_star)));
    } else {
      Logger::print_info(
          "  lapse OFF (use_gr_lapse = false): shift-only slow-rotation "
          "scheme, alpha == 1.  This is a documented approximation -- at "
          "compactness {} the true alpha(R*) is {:.4g}.",
          m_gr_compactness,
          double(std::sqrt(std::max(1e-4, 1.0 - m_gr_compactness))));
    }
  }

  m_time = 0.0;
  if (m_use_implicit) {
    Logger::print_info("DEC field solver initialized (semi-implicit, beta={}, "
                       "iters={}): Bp={}, Omega={}, obliquity={}",
                       m_beta, m_implicit_iters, m_Bp, m_Omega, m_obliquity);
  } else {
    Logger::print_info("DEC field solver initialized (explicit): Bp={}, "
                       "Omega={}, obliquity={}",
                       m_Bp, m_Omega, m_obliquity);
  }
}

// =========================================================================
// Main update dispatch
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update(double dt, uint32_t step) {
  if (m_use_implicit) {
    update_semi_implicit(dt);
  } else {
    update_explicit(dt);
  }
  refresh_total_fields();
  m_time += dt;
  // Same phase as prismatic_data_exporter: the step-s file holds the
  // post-update state E((s+1)dt), B((s+1/2)dt).
  if (m_distributed && m_rank_dump_interval > 0 &&
      step % m_rank_dump_interval == 0) {
    dump_rank_fields(step);
  }
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::refresh_total_fields() {
  ExecPolicy::launch(
      [Ne = m_dist.n_edges_local(), Nf = m_dist.n_faces_local()]
      LAMBDA(auto E, auto E0, auto Et, auto B, auto B0, auto Bt) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { Et[e] = E0[e] + E[e]; });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { Bt[f] = B0[f] + B[f]; });
      },
      m_E->data(), m_E0->data(), m_Etotal->data(),
      m_B->data(), m_B0->data(), m_Btotal->data());
  ExecPolicy::sync();
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::refresh_delta_ghosts() {
  // m_ex is inactive (all no-ops) when single-rank.
  m_ex.exchange_edge(m_E->data(), m_dist.e_split());
  m_ex.exchange_face(m_B->data(), m_dist.b_split());
}

// =========================================================================
// Compute RHS: dB/dt = -d1*E, dE/dt = h1inv*(d1t*h2*B - J)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::compute_rhs(
    buffer<Scalar>& E_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dE_out, buffer<Scalar>& dB_out) {
  // [halo sync point] E_in (h+v) and B_in (tri+rect) ghosts must be
  // fresh before the RHS.  Because update_semi_implicit calls this on
  // the Picard iterate each iteration, this IS the per-iteration ghost
  // refresh.  No-ops when single-rank.
  m_ex.exchange_edge(E_in, m_dist.e_split());
  m_ex.exchange_face(B_in, m_dist.b_split());
  if (m_use_frame_drag) {
    // Frame dragging: the Faraday rows of the RHS take the effective
    // circulation E + W(B).  Built from the CURRENT iterate (inside the
    // Picard loop), so the shift term is fully implicit-consistent.
    // The Ampere rows read B and J only, so passing Eeff through is
    // exact for them.
    m_dist.frame_drag_eff_E(E_in, B_in, m_B0->data(), m_Eeff);
    m_ex.exchange_edge(m_Eeff, m_dist.e_split());
    m_dist.compute_rhs(m_Eeff, B_in, m_J->data(), dE_out, dB_out);
  } else {
    m_dist.compute_rhs(E_in, B_in, m_J->data(), dE_out, dB_out);
  }
}

// =========================================================================
// Explicit update (original leapfrog)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update_explicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Faraday: B -= dt * d1 * E   (frame dragging: E -> E + W(B), the
  // effective circulation including the Lense-Thirring EMF; W needs
  // fresh B ghosts, and Eeff needs its own exchange because d1 reads
  // ghost edge columns.)
  // [halo sync point] exchange E (h+v) — no-op single-rank.
  if (m_update_b && m_use_frame_drag) {
    m_ex.exchange_face(m_B->data(), m_dist.b_split());
    m_dist.frame_drag_eff_E(m_E->data(), m_B->data(), m_B0->data(), m_Eeff);
    m_ex.exchange_edge(m_Eeff, m_dist.e_split());
    m_dist.faraday(m_Eeff, m_B->data(), dt);
  } else if (m_update_b) {
    m_ex.exchange_edge(m_E->data(), m_dist.e_split());
    m_dist.faraday(m_E->data(), m_B->data(), dt);
  }

  // Ampere: E += dt * h1inv * (d1t * h2 * B - J)
  // [halo sync point] exchange B (tri+rect).
  if (m_update_e && !m_use_recon_hodge) {
    m_ex.exchange_face(m_B->data(), m_dist.b_split());
    m_dist.ampere(m_E->data(), m_B->data(), m_J->data(), dt);
  } else if (m_update_e) {
    // Reconstruction-corrected Ampere (see prismatic_recon_hodge.h):
    //   circ[f] = W2-row(f) . B          (dual-segment circulations)
    //   S[e]    = sum_f d1t . circ - J   (exact dual loop sums)
    //   dE[e]   = W1-row(e) . S          (corrected pairing)
    // The d1t stage between the two reconstructions keeps Gauss-law /
    // charge conservation with deposited J topologically exact.
    auto rp = m_recon_hodge.get_ptrs(typename ExecPolicy::exec_tag{});
    ExecPolicy::launch(
        [N_faces = mp.N_faces, mp, rp] LAMBDA(auto B_f, auto circ) {
          ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
            circ[f] = rp.circ_face(mp, f, B_f);
          });
        },
        m_B->data(), m_tmp_B);
    ExecPolicy::launch(
        [N_edges = mp.N_edges, mp] LAMBDA(auto circ, auto J_e, auto S) {
          ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
            Scalar phi = Scalar(0);
            for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
              phi += mp.d1t_val[j] * circ[mp.d1t_col_idx[j]];
            }
            S[e] = phi - J_e[e];
          });
        },
        m_tmp_B, m_J->data(), m_tmp_E);
    ExecPolicy::launch(
        [N_edges = mp.N_edges, dt, mp, rp] LAMBDA(auto S, auto E_e) {
          ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
            E_e[e] += dt * rp.pair_edge(mp, e, S);
          });
        },
        m_tmp_E, m_E->data());
  }

  apply_damping(m_E->data(), m_B->data(), dt);
  if (m_use_pec_bc) {
    apply_pec_bc(m_E->data(), m_B->data());
  } else {
    // Leapfrog staggering: after this step E sits at t+dt but B (updated
    // by Faraday BEFORE Ampere from the same E) sits at t+dt/2.
    apply_inner_bc(m_E->data(), m_B->data(), m_time + dt, m_time + 0.5 * dt);
  }
  ExecPolicy::sync();
}

// =========================================================================
// Semi-implicit predictor-corrector update
//
//   F^{n+1} = F^n + dt * [alpha * RHS(F^n) + beta * RHS(F^{n+1})]
//
// where alpha = 1 - beta.  Solved by fixed-point iteration:
//   1. Compute RHS^n = RHS(F^n)
//   2. Euler predict: F* = F^n + dt * RHS^n
//   3. For i = 1..N_iter:
//        RHS* = RHS(F*)
//        F* = F^n + dt * (alpha * RHS^n + beta * RHS*)
//        Apply BC to F*
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update_semi_implicit(double dt) {
  Scalar alpha = Scalar(1) - m_beta;
  Scalar beta = m_beta;

  // Step 1: Compute RHS at current state
  compute_rhs(m_E->data(), m_B->data(), m_dE_dt, m_dB_dt);

  // Step 2: Euler predict — F* = F^n + dt * RHS^n
  m_dist.euler_predict(m_E->data(), m_dE_dt, m_tmp_E,
                       m_B->data(), m_dB_dt, m_tmp_B, dt);

  // Damp + apply BC to the Euler predict so the first RHS evaluation is
  // consistent.  Every candidate state F* is damped exactly once (here and
  // in each corrector iteration); the final copy-back is already damped.
  apply_damping(m_tmp_E, m_tmp_B, dt);
  if (m_use_pec_bc) {
    apply_pec_bc(m_tmp_E, m_tmp_B);
  } else {
    apply_inner_bc(m_tmp_E, m_tmp_B, m_time + dt, m_time + dt);
  }
  ExecPolicy::sync();

  // Step 3: Iterate corrector.  compute_rhs is a halo sync point — the
  // ghost refresh must happen inside EVERY iteration (see the plan).
  for (int iter = 0; iter < m_implicit_iters; iter++) {
    compute_rhs(m_tmp_E, m_tmp_B, m_dE_dt_new, m_dB_dt_new);

    // F* = F^n + dt * (alpha * RHS^n + beta * RHS*)
    m_dist.picard_combine(m_E->data(), m_dE_dt, m_dE_dt_new, m_tmp_E,
                          m_B->data(), m_dB_dt, m_dB_dt_new, m_tmp_B,
                          dt, alpha, beta);

    apply_damping(m_tmp_E, m_tmp_B, dt);
    if (m_use_pec_bc) {
      apply_pec_bc(m_tmp_E, m_tmp_B);
    } else {
      apply_inner_bc(m_tmp_E, m_tmp_B, m_time + dt, m_time + dt);
    }
    ExecPolicy::sync();
  }

  // Step 4: Copy result back — F^{n+1} = F* (all local slots; ghosts
  // are refreshed at the next sync point before any read)
  ExecPolicy::launch(
      [Ne = m_dist.n_edges_local(), Nf = m_dist.n_faces_local()]
      LAMBDA(auto E, auto tmpE, auto B, auto tmpB) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { E[e] = tmpE[e]; });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { B[f] = tmpB[f]; });
      },
      m_E->data(), m_tmp_E, m_B->data(), m_tmp_B);

  // Step 5: BC (the copied-back state was already damped as the final
  // iterate; damping it again here would double-apply exp(-sigma*dt))
  if (m_use_pec_bc) {
    apply_pec_bc(m_E->data(), m_B->data());
  } else {
    // Semi-implicit fields are co-located in time.
    apply_inner_bc(m_E->data(), m_B->data(), m_time + dt, m_time + dt);
  }
  ExecPolicy::sync();
}

// =========================================================================
// Damping — runs on GPU
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_damping(
    buffer<Scalar>& E, buffer<Scalar>& B, double dt) {
  // sigma(k) = coef * ramp^p with ramp in (0, 1].  p >= 3 keeps the layer
  // entrance adiabatic: the entrance reflection interferes with the hard
  // inner-BC driver at FIRST order in the reflected amplitude and shifts
  // the steady-state luminosity (A2 absorber study).
  m_dist.apply_damping(E, B, dt, m_damping_length, m_damping_coef,
                       m_damping_exponent);
}

// =========================================================================
// Inner boundary condition
//
// Computes ∫_face B · dA and ∫_edge E · dl on inner boundary elements
// using Gauss quadrature (gauss_quad on host, gauss_quad_dev on device).
//
// Two modes controlled by m_use_deutsch_bc:
//   false — instantaneous rotating dipole + corotation E
//   true  — full retarded Deutsch solution
// =========================================================================

// Portable gauss_quad dispatch: calls gauss_quad_dev on GPU, gauss_quad on CPU

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_inner_bc(
    buffer<Scalar>& E, buffer<Scalar>& B, double time_E, double time_B) {
  dec_inner_bc_params par;
  par.Bp = m_Bp;
  par.Omega = m_Omega;
  par.obliquity = m_obliquity;
  par.use_deutsch = m_use_deutsch_bc;
  par.overwrite_b = m_inner_bc_overwrite_b;
  if (m_use_frame_drag) {
    par.omega_lt0 = m_omega_lt0;
    par.lt_r_star = m_lt_r_star;
    par.lt_p = m_lt_exponent;
  }
  m_dist.apply_inner_bc(E, B, m_B0->data(), par, time_E, time_B);
}

// =========================================================================
// Initial dipole (host-only, called once during init)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::fill_dipole_B(
    buffer<Scalar>& B, Scalar mx_v, Scalar my_v, Scalar mz_v) {
  m_dist.fill_dipole_B(B, mx_v, my_v, mz_v);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_dipole() {
  fill_dipole_B(m_B->data(), m_Bp * std::sin(m_obliquity), Scalar(0),
                m_Bp * std::cos(m_obliquity));

  // Delta formulation: subtract the static background cochains (zero when
  // use_static_background is off).  E starts at zero.
  m_E->data().assign(Scalar(0));
  m_dist.subtract_face(m_B->data(), m_B0->data());

  ExecPolicy::sync();
  refresh_total_fields();
}

// =========================================================================
// Full Deutsch retarded IC (host-only, called from main)
//
// Initializes both B_f and E_e from the exact retarded solution of a
// rotating magnetic dipole that has been spinning since t = -∞.
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_deutsch() {
  Scalar Bp_v = m_Bp;
  Scalar Omega_v = m_Omega;
  Scalar obl_v = m_obliquity;
  // Base time of the analytic snapshot (config "deutsch_ic_time",
  // default 0).  Nonzero values turn this IC into a GPU-fast generator
  // of analytic reference cochains at arbitrary t for convergence
  // analysis (run with max_steps = 0/1 and dump).
  double t_base = 0.0;
  sim_env().params().get_value("deutsch_ic_time", t_base);
  Scalar t_init = Scalar(t_base);  // E initial time

  // Leapfrog staggering: the first Faraday half-step advances B from
  // -dt/2 to +dt/2, so a consistent start evaluates the analytic B at
  // t = -dt/2 (E stays at 0).  The co-located semi-implicit scheme
  // initializes both at 0.
  double dt_param = 0.0;
  sim_env().params().get_value("dt", dt_param);
  Scalar t_init_B = Scalar(
      m_use_implicit ? t_base : t_base - 0.5 * dt_param);

  m_dist.set_initial_deutsch(m_E->data(), m_B->data(), Bp_v, Omega_v, obl_v,
                             t_init, t_init_B);

  // Delta formulation: subtract the static background B cochains (zero
  // when use_static_background is off; E0 is identically zero).
  m_dist.subtract_face(m_B->data(), m_B0->data());

  ExecPolicy::sync();
  refresh_total_fields();
  Logger::print_info("Deutsch retarded IC set: Bp={}, Omega={}, obliquity={}",
                     m_Bp, m_Omega, m_obliquity);
}

// =========================================================================
// Spherical-cavity TE/TM eigenmode IC (host-only, called from main)
//
// Initializes E_e and B_f from the analytical fields of a single TE or TM
// mode of a spherical resonator with PEC walls at r_min and r_max.
//
// The eigenvalue ω = c k is found at runtime by bisecting the appropriate
// determinant equation between the spherical Bessel functions; α is then
// fixed by f(r_min) = 0 (TE) or (r f)'(r_min) = 0 (TM). With c = 1, ω = k.
//
// Phase choice (start_with_e):
//   true  → E(t=0) = E_pat,  B(t=0) = 0      (E max, B zero)
//   false → E(t=0) = 0,      B(t=0) = B_pat  (B max, E zero) — default
// =========================================================================

// NOTE: still GLOBAL-indexed (single-rank only) — a benchmark-mode IC.
// Convert like set_initial_deutsch (owned-local loops via l2g) if PEC
// cavity runs are ever needed under a real partition.
template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_resonator_mode(
    int l, int m, int n_root, char polarization, bool start_with_e) {
  if (l < 1 || std::abs(m) > l) {
    Logger::print_err("set_initial_resonator_mode: invalid (l, m) = ({}, {})",
                       l, m);
    return;
  }
  if (polarization != 'E' && polarization != 'M') {
    Logger::print_err("set_initial_resonator_mode: polarization must be 'E' "
                      "(TE) or 'M' (TM), got '{}'", polarization);
    return;
  }
  bool is_te = (polarization == 'E');

  // Eigenvalue and inner-BC coefficient: computed once on the host.
  double a = m_mesh.m_r_min;
  double b = m_mesh.m_r_max;
  double k = is_te ? cavity_modes::te_eigenvalue(l, a, b, n_root)
                   : cavity_modes::tm_eigenvalue(l, a, b, n_root);
  double alpha = cavity_modes::inner_bc_alpha(l, k, a, is_te);

  cavity_modes::mode_params mp_params{l, m, k, alpha,
                                       static_cast<double>(m_resonator_amp), is_te};

  Logger::print_info(
      "Resonator IC: {}_{}_{}_{} mode, k={:.6f} (omega={:.6f}), "
      "alpha={:.6e}, start_with_e={}",
      is_te ? "TE" : "TM", l, m, n_root, k, k, alpha, start_with_e);

  // For analytic comparison later, the spatial patterns satisfy
  //   start_with_e = true:  E(t) = +E_pat cos(ωt), B(t) = +B_pat sin(ωt)
  //   start_with_e = false: E(t) = -E_pat sin(ωt), B(t) = +B_pat cos(ωt)
  // So at t = 0:
  //   start_with_e = true:  E = E_pat,  B = 0
  //   start_with_e = false: E = 0,      B = B_pat

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);

  // ---- Initialize B (face fluxes) on device ----
  ExecPolicy::launch(
      [N_faces = mp.N_faces, n_tri_faces, mp_params, mp, start_with_e]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          if (start_with_e) {
            B_f[f] = Scalar(0);
            return;
          }
          if (f < n_tri_faces) {
            // Spherical triangular face on shell radius r_face.
            int vi0 = mp.tri_face_v0[f];
            int vi1 = mp.tri_face_v1[f];
            int vi2 = mp.tri_face_v2[f];
            Scalar r_face, a0x, a0y, a0z, r1_, a1x, a1y, a1z, r2_, a2x, a2y, a2z;
            vertex_unit(mp, vi0, r_face, a0x, a0y, a0z);
            vertex_unit(mp, vi1, r1_, a1x, a1y, a1z);
            vertex_unit(mp, vi2, r2_, a2x, a2y, a2z);
            (void)r1_; (void)r2_;

            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double t) -> double {
                Scalar x, y, z, nx, ny, nz;
                tri_sphere_sample(r_face, a0x, a0y, a0z, a1x, a1y, a1z,
                                  a2x, a2y, a2z,
                                  static_cast<Scalar>(u), static_cast<Scalar>(t),
                                  x, y, z, nx, ny, nz);
                double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
                cavity_modes::evaluate_mode_patterns(
                    mp_params, x, y, z, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
                return Bx_p*nx + By_p*ny + Bz_p*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          } else {
            // Ruled rectangular face P(u,v) = r(v) · slerp(û_a, û_b, u).
            int fi = f - n_tri_faces;
            int vi0 = mp.rect_face_v0[fi];
            int vi1 = mp.rect_face_v1[fi];
            int vi3 = mp.rect_face_v3[fi];
            Scalar r_lo, uax, uay, uaz, r_tmp, ubx, uby, ubz, r_hi, ux3, uy3, uz3;
            vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
            vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
            vertex_unit(mp, vi3, r_hi, ux3, uy3, uz3);
            (void)r_tmp; (void)ux3; (void)uy3; (void)uz3;

            double flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double v) -> double {
                Scalar x, y, z, nx, ny, nz;
                rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                                   static_cast<Scalar>(u), static_cast<Scalar>(v),
                                   x, y, z, nx, ny, nz);
                double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
                cavity_modes::evaluate_mode_patterns(
                    mp_params, x, y, z, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
                return Bx_p*nx + By_p*ny + Bz_p*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          }
        });
      },
      m_B->data());

  // ---- Initialize E (edge circulations) on device ----
  ExecPolicy::launch(
      [N_edges = mp.N_edges, mp_params, mp, start_with_e]
      LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          if (!start_with_e) {
            E_e[e] = Scalar(0);
            return;
          }
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
          vertex_unit(mp, v0, r0, a0x, a0y, a0z);
          vertex_unit(mp, v1, r1, a1x, a1y, a1z);
          bool is_radial = (r0 != r1);

          double circ = gauss_quad([&](double t) -> double {
            Scalar x, y, z, dlx, dly, dlz;
            if (is_radial) {
              Scalar rt = (Scalar(1) - static_cast<Scalar>(t)) * r0 +
                          static_cast<Scalar>(t) * r1;
              Scalar dr = r1 - r0;
              x = rt * a0x; y = rt * a0y; z = rt * a0z;
              dlx = dr * a0x; dly = dr * a0y; dlz = dr * a0z;
            } else {
              h_edge_sphere_sample(r0, a0x, a0y, a0z, a1x, a1y, a1z,
                                   static_cast<Scalar>(t),
                                   x, y, z, dlx, dly, dlz);
            }
            double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
            cavity_modes::evaluate_mode_patterns(
                mp_params, x, y, z, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
            return Ex_p*dlx + Ey_p*dly + Ez_p*dlz;
          }, 0.0, 1.0);
          E_e[e] = static_cast<Scalar>(circ);
        });
      },
      m_E->data());

  // Enforce PEC on the boundary so the IC is exactly compatible.
  apply_pec_bc(m_E->data(), m_B->data());
  ExecPolicy::sync();
  refresh_total_fields();
}

// =========================================================================
// PEC boundary condition: zero tangential E and normal B on r = r_min, r_max
//
// Tangential E lives on horizontal edges of shells k = 0 and k = N_r.
// Normal B lives on triangular faces of shells k = 0 and k = N_r.
// (Vertical edges and rectangular faces of the boundary layers are NOT
//  on the conducting surface itself, only adjacent — leave them alone.)
// =========================================================================

// =========================================================================
// Per-rank field dump (4.1b.7 option (i)).  Owned slots of the TOTAL
// E/B cochains, plus the owned l2g maps for stitching.  Blocks are
// written separately (h/v edges, tri/rect faces) since the global
// combined offsets are reconstructible from the mesh dimensions.
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::dump_rank_fields(uint32_t step) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  m_Etotal->data().copy_to_host();
  m_Btotal->data().copy_to_host();
#endif
  auto const& L_he = m_mesh_part.layout(cochain_type::h_edge);
  auto const& L_ve = m_mesh_part.layout(cochain_type::v_edge);
  auto const& L_tri = m_mesh_part.layout(cochain_type::tri_face);
  auto const& L_rect = m_mesh_part.layout(cochain_type::rect_face);
  int world_rank = m_mpi == nullptr ? 0 : m_mpi->world_rank();
  std::filesystem::create_directories(m_output_dir);
  char fname[512];
  std::snprintf(fname, sizeof(fname), "%s/rank%04d_step_%06u.h5",
                m_output_dir.c_str(), world_rank, step);
  auto file = hdf_create(fname);

  auto dump_block = [&](const distributed_cochain_layout& L,
                        const Scalar* vals, const char* vname,
                        const char* gname) {
    const int n = L.owned_size();
    std::vector<Scalar> v(n);
    std::vector<gidx_t> g(n);
    for (int l = 0; l < n; ++l) {
      v[l] = vals[l];
      g[l] = L.to_global(l);
    }
    file.write(v.data(), n, vname);
    file.write(g.data(), n, gname);
  };
  dump_block(L_he, m_Etotal->host_ptr_a(), "E_h", "E_h_g");
  dump_block(L_ve, m_Etotal->host_ptr_b(), "E_v", "E_v_g");
  dump_block(L_tri, m_Btotal->host_ptr_a(), "B_tri", "B_tri_g");
  dump_block(L_rect, m_Btotal->host_ptr_b(), "B_rect", "B_rect_g");
  file.write(double(m_time), "time");
  file.close();
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_pec_bc(
    buffer<Scalar>& E, buffer<Scalar>& B) {
  m_dist.apply_pec_bc(E, B);
}

}  // namespace Aperture
