#pragma once

#include "systems/prismatic/cavity_modes.hpp"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "framework/environment.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

// Spherical-geometry quadrature helpers and analytic dipole / Deutsch
// evaluators live in dec_solver_geometry.hpp (shared with the
// distributed solver core).
// =========================================================================
// Constructor and init
// =========================================================================

template <typename ExecPolicy>
dec_field_solver<ExecPolicy>::dec_field_solver(prismatic_mesh& mesh)
    : m_mesh(mesh),
      m_tmp_E(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_tmp_B(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_dE_dt(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_dB_dt(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_dE_dt_new(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_dB_dt_new(mesh.m_N_faces, ExecPolicy::data_mem_type()) {}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
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
  // 4.1b: build the distributed core.  Single-rank partition for now —
  // the local cochain layouts are then identity maps onto the global
  // ordering, so the registered field buffers pass straight into the
  // core's kernels.  The MPI driver (B2) replaces this with the rank's
  // real partition.
  m_topo = icosphere_topology::build_from_mesh(m_mesh);
  m_part = prismatic_partition::single_rank(m_mesh.m_L, m_mesh.m_N_r);
  m_part.set_topology(&m_topo);
  m_mesh_part = prismatic_mesh_partition::build(m_part, m_topo);
  m_dist.build(m_mesh, m_mesh_part);

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
      Logger::print_info(
          "Static background enabled: aligned dipole mz = {}",
          m_Bp * std::cos(m_obliquity));
    }
  }
  refresh_total_fields();

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
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::refresh_total_fields() {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces]
      LAMBDA(auto E, auto E0, auto Et, auto B, auto B0, auto Bt) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { Et[e] = E0[e] + E[e]; });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { Bt[f] = B0[f] + B[f]; });
      },
      m_E->data(), m_E0->data(), m_Etotal->data(),
      m_B->data(), m_B0->data(), m_Btotal->data());
  ExecPolicy::sync();
}

// =========================================================================
// Compute RHS: dB/dt = -d1*E, dE/dt = h1inv*(d1t*h2*B - J)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::compute_rhs(
    buffer<Scalar>& E_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dE_out, buffer<Scalar>& dB_out) {
  // [halo sync point] E_in (h+v) and B_in (tri+rect) ghosts must be
  // fresh before the RHS — no-op under single_rank; B2 exchanges here.
  m_dist.compute_rhs(E_in, B_in, m_J->data(), dE_out, dB_out);
}

// =========================================================================
// Explicit update (original leapfrog)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update_explicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Faraday: B -= dt * d1 * E
  // [halo sync point] exchange E (h+v) before this — no-op single-rank.
  if (m_update_b) {
    m_dist.faraday(m_E->data(), m_B->data(), dt);
  }

  // Ampere: E += dt * h1inv * (d1t * h2 * B - J)
  // [halo sync point] exchange B (tri+rect) before this.
  if (m_update_e && !m_use_recon_hodge) {
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
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
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

  // Step 4: Copy result back — F^{n+1} = F*
  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces]
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
  m_dist.apply_inner_bc(E, B, m_B0->data(), par, time_E, time_B);
}

// =========================================================================
// Initial dipole (host-only, called once during init)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::fill_dipole_B(
    buffer<Scalar>& B, Scalar mx_v, Scalar my_v, Scalar mz_v) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);

  // ---- B (face fluxes) via Gauss quadrature on device ----
  ExecPolicy::launch(
      [N_faces = mp.N_faces, n_tri_faces, mx_v, my_v, mz_v, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
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
                Scalar bx, by, bz;
                dipole_B_impl(x, y, z, mx_v, my_v, mz_v, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
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
                Scalar bx, by, bz;
                dipole_B_impl(x, y, z, mx_v, my_v, mz_v, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          }
        });
      },
      B);
  ExecPolicy::sync();
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_dipole() {
  fill_dipole_B(m_B->data(), m_Bp * std::sin(m_obliquity), Scalar(0),
                m_Bp * std::cos(m_obliquity));

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  // Delta formulation: subtract the static background cochains (zero when
  // use_static_background is off).  E starts at zero.
  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces]
      LAMBDA(auto E_e, auto B_f, auto B0_f) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { E_e[e] = Scalar(0); });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { B_f[f] -= B0_f[f]; });
      },
      m_E->data(), m_B->data(), m_B0->data());

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

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);

  // ---- B (face fluxes) on device ----
  ExecPolicy::launch(
      [N_faces = mp.N_faces, n_tri_faces, Bp_v, Omega_v, obl_v,
       t_init = t_init_B, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          if (f < n_tri_faces) {
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
                Scalar bx, by, bz;
                deutsch_B_impl(x, y, z, t_init, Bp_v, Omega_v, obl_v, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          } else {
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
                Scalar bx, by, bz;
                deutsch_B_impl(x, y, z, t_init, Bp_v, Omega_v, obl_v, bx, by, bz);
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          }
        });
      },
      m_B->data());

  // ---- E (edge circulations) on device ----
  ExecPolicy::launch(
      [N_edges = mp.N_edges, Bp_v, Omega_v, obl_v, t_init, mp]
      LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
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
            Scalar ex, ey, ez;
            deutsch_E_impl(x, y, z, t_init, Bp_v, Omega_v, obl_v, ex, ey, ez);
            return ex*dlx + ey*dly + ez*dlz;
          }, 0.0, 1.0);
          E_e[e] = static_cast<Scalar>(circ);
        });
      },
      m_E->data());

  // Delta formulation: subtract the static background B cochains (zero
  // when use_static_background is off; E0 is identically zero).
  ExecPolicy::launch(
      [Nf = mp.N_faces] LAMBDA(auto B_f, auto B0_f) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { B_f[f] -= B0_f[f]; });
      },
      m_B->data(), m_B0->data());

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

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_pec_bc(
    buffer<Scalar>& E, buffer<Scalar>& B) {
  m_dist.apply_pec_bc(E, B);
}

}  // namespace Aperture
