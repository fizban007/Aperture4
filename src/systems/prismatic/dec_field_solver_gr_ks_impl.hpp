#pragma once

#include "systems/prismatic/dec_field_solver_gr_ks.h"
#include "systems/prismatic/prismatic_mesh_metric_ptrs.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Helper: scalar triple product  det(a, b, c) = a · (b × c)
// =========================================================================
HD_INLINE Scalar triple(Scalar ax, Scalar ay, Scalar az,
                        Scalar bx, Scalar by, Scalar bz,
                        Scalar cx, Scalar cy, Scalar cz) {
  return ax * (by*cz - bz*cy) +
         ay * (bz*cx - bx*cz) +
         az * (bx*cy - by*cx);
}

// =========================================================================
// Helper: edge tangent vector (v1 - v0).
// =========================================================================
HD_INLINE void edge_tangent(const prismatic_mesh_metric_ptrs& mp, int e,
                            Scalar& tx, Scalar& ty, Scalar& tz) {
  int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
  tx = mp.vert_x[v1] - mp.vert_x[v0];
  ty = mp.vert_y[v1] - mp.vert_y[v0];
  tz = mp.vert_z[v1] - mp.vert_z[v0];
}

// =========================================================================
// Helper: edge midpoint (Cartesian).
// =========================================================================
HD_INLINE void edge_midpoint(const prismatic_mesh_metric_ptrs& mp, int e,
                             Scalar& mx, Scalar& my, Scalar& mz) {
  int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
  mx = Scalar(0.5) * (mp.vert_x[v0] + mp.vert_x[v1]);
  my = Scalar(0.5) * (mp.vert_y[v0] + mp.vert_y[v1]);
  mz = Scalar(0.5) * (mp.vert_z[v0] + mp.vert_z[v1]);
}

// =========================================================================
// Helper: face normal area vector (unnormalized).
//   Triangles: 0.5 * (edge1 × edge2)
//   Quads:     edge1 × edge2
// =========================================================================
HD_INLINE void face_normal_area(const prismatic_mesh_metric_ptrs& mp, int f,
                                Scalar& nx, Scalar& ny, Scalar& nz) {
  if (mp.is_tri_face(f)) {
    int va = mp.tri_face_v0[f], vb = mp.tri_face_v1[f], vc = mp.tri_face_v2[f];
    Scalar ax = mp.vert_x[vb] - mp.vert_x[va];
    Scalar ay = mp.vert_y[vb] - mp.vert_y[va];
    Scalar az = mp.vert_z[vb] - mp.vert_z[va];
    Scalar bx = mp.vert_x[vc] - mp.vert_x[va];
    Scalar by = mp.vert_y[vc] - mp.vert_y[va];
    Scalar bz = mp.vert_z[vc] - mp.vert_z[va];
    nx = Scalar(0.5) * (ay*bz - az*by);
    ny = Scalar(0.5) * (az*bx - ax*bz);
    nz = Scalar(0.5) * (ax*by - ay*bx);
  } else {
    int local = f - mp.N_tri_faces;
    int va = mp.rect_face_v0[local], vb = mp.rect_face_v1[local];
    int vd = mp.rect_face_v3[local];
    Scalar ax = mp.vert_x[vb] - mp.vert_x[va];
    Scalar ay = mp.vert_y[vb] - mp.vert_y[va];
    Scalar az = mp.vert_z[vb] - mp.vert_z[va];
    Scalar bx = mp.vert_x[vd] - mp.vert_x[va];
    Scalar by = mp.vert_y[vd] - mp.vert_y[va];
    Scalar bz = mp.vert_z[vd] - mp.vert_z[va];
    nx = ay*bz - az*by;
    ny = az*bx - ax*bz;
    nz = ax*by - ay*bx;
  }
}

// =========================================================================
// Helper: face centroid (Cartesian).
// =========================================================================
HD_INLINE void face_centroid(const prismatic_mesh_metric_ptrs& mp, int f,
                             Scalar& cx, Scalar& cy, Scalar& cz) {
  if (mp.is_tri_face(f)) {
    int va = mp.tri_face_v0[f], vb = mp.tri_face_v1[f], vc = mp.tri_face_v2[f];
    cx = (mp.vert_x[va] + mp.vert_x[vb] + mp.vert_x[vc]) / Scalar(3);
    cy = (mp.vert_y[va] + mp.vert_y[vb] + mp.vert_y[vc]) / Scalar(3);
    cz = (mp.vert_z[va] + mp.vert_z[vb] + mp.vert_z[vc]) / Scalar(3);
  } else {
    int local = f - mp.N_tri_faces;
    int va = mp.rect_face_v0[local], vb = mp.rect_face_v1[local];
    int vc = mp.rect_face_v2[local], vd = mp.rect_face_v3[local];
    cx = Scalar(0.25) * (mp.vert_x[va] + mp.vert_x[vb] +
                         mp.vert_x[vc] + mp.vert_x[vd]);
    cy = Scalar(0.25) * (mp.vert_y[va] + mp.vert_y[vb] +
                         mp.vert_y[vc] + mp.vert_y[vd]);
    cz = Scalar(0.25) * (mp.vert_z[va] + mp.vert_z[vb] +
                         mp.vert_z[vc] + mp.vert_z[vd]);
  }
}

// =========================================================================
// Constructor
// =========================================================================
template <typename ExecPolicy>
dec_field_solver_gr_ks<ExecPolicy>::dec_field_solver_gr_ks(
    prismatic_mesh_metric& mesh)
    : m_mesh(mesh),
      m_E_aux(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_H_aux(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_tmp_D(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_tmp_B(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_dD_dt(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_dB_dt(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_dD_dt_new(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_dB_dt_new(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_D_bg(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_B_bg(mesh.m_N_faces, ExecPolicy::data_mem_type()) {}

template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  // The evolved edge field is physically D (the electric displacement
  // 1-form in the 3+1 decomposition) but we register it under "E" to
  // share a common primal-edge data slot with the flat-space solver and
  // the existing data exporter.  The auxiliary  E_i = α D_i + (β×B)_i
  // is constructed inside compute_rhs and is not separately registered.
  m_D = sim_env().template register_data<prismatic_edge_field>(
      "E", m_mesh, mem);
  m_B = sim_env().template register_data<prismatic_face_field>(
      "B", m_mesh, mem);
  m_J = sim_env().template register_data<prismatic_edge_field>(
      "J", m_mesh, mem);
}

template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::init() {
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
  sim_env().params().get_value("update_d", m_update_d);
  sim_env().params().get_value("update_b", m_update_b);
  sim_env().params().get_value("use_implicit", m_use_implicit);
  sim_env().params().get_value("implicit_beta", m_beta);
  sim_env().params().get_value("implicit_iters", m_implicit_iters);

  // Optional safety damping inside the horizon.  Default: disabled; the
  // solver relies on causal disconnection for the inner BC (see
  // apply_horizon_damping docstring).  Enable by setting r_horizon_damp
  // > 0 in the config if long-time integration shows numerical leakage.
  sim_env().params().get_value("r_horizon_damp", m_r_horizon_damp);
  sim_env().params().get_value("r_horizon_inner", m_r_horizon_inner);

  m_time = 0.0;
  if (m_r_horizon_damp > Scalar(0)) {
    Logger::print_info(
        "DEC GR field solver initialized: horizon safety damping ON "
        "[{:.3f}, {:.3f}], implicit={}",
        m_r_horizon_inner, m_r_horizon_damp, m_use_implicit);
  } else {
    Logger::print_info(
        "DEC GR field solver initialized: horizon safety damping OFF "
        "(causal disconnection only), implicit={}",
        m_use_implicit);
  }
}

// =========================================================================
// Main update dispatch
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::update(double dt, uint32_t step) {
  if (m_use_implicit) {
    update_semi_implicit(dt);
  } else {
    update_explicit(dt);
  }
  m_time += dt;
}

// =========================================================================
// Faraday half-step: construct E_aux from (D̃, B) then dB = -d1·E_aux.
//
// Option C (primal/dual DEC): the primary 1-cochain D̃[e] stores the
// dual 2-cochain of the D 2-form (i.e., ∫_{e*} D_{2form}).  To build the
// line integral of E_aux along a primal edge, we first convert to the
// primal 1-cochain via the Hodge star:
//    D[e] = hodge1_inv[e] · D̃[e]   (covariant 1-form line integral)
// and then
//    E_aux_integral[e] = α_edge · D[e] + (shift cross term)
//
// dB[f] = -Σ_e d1[f,e] · E_aux_integral[e]   (pure topological Stokes)
//
// The Hodge star appears only in the constitutive step (D̃ → D), not in
// the time-stepping operator.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::compute_dB_dt(
    buffer<Scalar>& D_in, buffer<Scalar>& B_in, buffer<Scalar>& dB_out) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // E_aux on vertical edges — pure lapse, no shift cross term.
  ExecPolicy::launch(
      [Nh = mp.N_h_edges, Ne = mp.N_edges, mp]
      LAMBDA(auto D_e, auto E_aux) {
        ExecPolicy::loop(Nh, Ne, [&] LAMBDA(int e) {
          E_aux[e] = mp.edge_alpha[e] * mp.hodge1_inv[e] * D_e[e];
        });
      },
      D_in, m_E_aux);

  // E_aux on horizontal edges — lapse · D[e] + shift cross term.
  //
  // Flux-conservative construction (matches the 2D solver structure):
  //   (β × B)·t̂_e at edge e = [Σ_{f' adj e, rect} √γ_{f'}·β^r_{f'}·
  //                             (B^i_{f'} projected onto (r̂×t̂_e))] / √γ_e / N
  // then multiplied by |t_e| for the line integral.  The flux quantity
  // √γ·β^r·B^i is summed at the source (face) locations; division by
  // √γ_e at the destination (edge) converts back to the field value.
  // Using the same √γ-weighted sum / √γ-at-destination pattern as the
  // 2D Yee code: (w_- gb1_- X_- + w_+ gb1_+ X_+)/(w_- + w_+) with
  // w ≡ √γ/sinθ and gb1 ≡ √γ β^r.
  ExecPolicy::launch(
      [Nh = mp.N_h_edges, mp]
      LAMBDA(auto D_e, auto B_f, auto E_aux) {
        ExecPolicy::loop(0, Nh, [&] LAMBDA(int e) {
          Scalar e_aux = mp.edge_alpha[e] * mp.hodge1_inv[e] * D_e[e];

          Scalar tx, ty, tz;
          edge_tangent(mp, e, tx, ty, tz);
          Scalar t_mag = math::sqrt(tx*tx + ty*ty + tz*tz);
          Scalar tx_h = tx / t_mag, ty_h = ty / t_mag, tz_h = tz / t_mag;

          Scalar mx, my, mz;
          edge_midpoint(mp, e, mx, my, mz);
          Scalar r = mp.edge_r_coord[e];
          Scalar lx = mx / r, ly = my / r, lz = mz / r;

          Scalar inv_sgma_e = Scalar(1) / mp.edge_sqrt_gamma[e];

          // Σ √γ_f β^r_f · (B^i_f projected onto r̂×t̂_e).
          // Triangular faces contribute 0 (n̂_tri ∥ r̂ ⇒ triple = 0).
          Scalar cross = Scalar(0);
          int count = 0;
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            if (mp.is_tri_face(f)) continue;
            Scalar nx, ny, nz;
            face_normal_area(mp, f, nx, ny, nz);
            Scalar n_mag = math::sqrt(nx*nx + ny*ny + nz*nz);
            Scalar nx_h = nx / n_mag, ny_h = ny / n_mag, nz_h = nz / n_mag;
            Scalar tu = triple(lx, ly, lz, nx_h, ny_h, nz_h,
                               tx_h, ty_h, tz_h);
            // B^i normal-to-face reconstruction: B·n̂ = B[f]/|n_f|.
            Scalar B_normal = B_f[f] / n_mag;
            cross += mp.face_sq_gamma_beta_r[f] * B_normal * tu;
            count += 1;
          }
          if (count > 0) {
            e_aux += t_mag * inv_sgma_e * cross / Scalar(count);
          }
          E_aux[e] = e_aux;
        });
      },
      D_in, B_in, m_E_aux);

  // Faraday: dB[f] = -Σ_e d1[f,e] · E_aux[e].  Pure topological curl.
  ExecPolicy::launch(
      [Nf = mp.N_faces, mp] LAMBDA(auto E_aux, auto dB) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          Scalar curl_E = Scalar(0);
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            curl_E += mp.d1_val[j] * E_aux[mp.d1_col_idx[j]];
          }
          dB[f] = -curl_E;
        });
      },
      m_E_aux, dB_out);
}

// =========================================================================
// Ampère half-step: construct H_aux_line (line integral of H_aux along
// dual edges) from (D̃, B) then dD̃/dt = Σ d1t · H_aux_line − J̃.
//
// Option C (primal/dual DEC): the primary variable D̃[e] is the dual
// 2-cochain of the D 2-form.  The Ampère update is a pure topological
// Stokes theorem on the dual mesh — NO Hodge star in the time-stepping
// operator.
//
// H_aux_line[f] = ∫_{f*} H_aux_i dx^i  (line integral along dual edge
// passing through primal face f), constructed from the constitutive
// relation.  The Hodge star hodge2 enters here only to convert the
// primal 2-cochain B[f] to the dual 1-cochain B̃[f] = hodge2 · B[f]:
//
//    H_aux_line[f] = α_face · hodge2[f] · B[f]  +  (shift cross term)
//
// Shift cross term at a rectangular face: same structure as before but
// now interpreted as an approximation of the dual-edge line integral of
// -ε_{ijk} √γ β^j D^k dx^i.  Triangular faces have no shift cross term
// (dual edge ∥ ∂_r ⇒ det(r̂, t, dual_edge) = 0 identically).
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::compute_dD_dt(
    buffer<Scalar>& D_in, buffer<Scalar>& B_in, buffer<Scalar>& dD_out) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // H_aux_line on triangular faces — pure lapse · Hodge2 · B[f].
  ExecPolicy::launch(
      [Ntri = mp.N_tri_faces, mp]
      LAMBDA(auto B_f, auto H_aux) {
        ExecPolicy::loop(0, Ntri, [&] LAMBDA(int f) {
          H_aux[f] = mp.face_alpha[f] * mp.hodge2[f] * B_f[f];
        });
      },
      B_in, m_H_aux);

  // H_aux_line on rectangular faces — lapse · Hodge2 · B[f] + shift cross term.
  //
  // Symmetric to the Faraday construction:
  //   (β × D)·n̂_f at face f = [Σ_{e' adj f, horiz} √γ_{e'}·β^r_{e'}·
  //                             (D^i_{e'} projected onto r̂×n̂_f)] / √γ_f / N
  // multiplied by the dual-edge length |f*| ≈ hodge2[f]·|n_f| for the
  // line integral along the dual edge through f.  Sign is -(β×D) in
  // H = αB - β×D.
  ExecPolicy::launch(
      [Ntri = mp.N_tri_faces, Nf = mp.N_faces, mp]
      LAMBDA(auto D_e, auto B_f, auto H_aux) {
        ExecPolicy::loop(Ntri, Nf, [&] LAMBDA(int f) {
          Scalar h_aux = mp.face_alpha[f] * mp.hodge2[f] * B_f[f];

          Scalar nx, ny, nz;
          face_normal_area(mp, f, nx, ny, nz);
          Scalar n_mag = math::sqrt(nx*nx + ny*ny + nz*nz);
          Scalar nx_h = nx / n_mag, ny_h = ny / n_mag, nz_h = nz / n_mag;

          Scalar cx, cy, cz;
          face_centroid(mp, f, cx, cy, cz);
          Scalar r = mp.face_r_coord[f];
          Scalar lx = cx / r, ly = cy / r, lz = cz / r;

          Scalar inv_sgma_f = Scalar(1) / mp.face_sqrt_gamma[f];
          Scalar dual_len = mp.hodge2[f] * n_mag;  // |f*| ≈ h2 · |f_area|

          // Σ √γ_e β^r_e · (D^i_e projected onto r̂×n̂_f).
          // Vertical edges contribute 0 (t̂_vert ∥ r̂ ⇒ triple = 0).
          Scalar cross = Scalar(0);
          int count = 0;
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            int e = mp.d1_col_idx[j];
            if (mp.is_vertical_edge(e)) continue;
            Scalar tx, ty, tz;
            edge_tangent(mp, e, tx, ty, tz);
            Scalar t_mag = math::sqrt(tx*tx + ty*ty + tz*tz);
            Scalar tx_h = tx / t_mag, ty_h = ty / t_mag, tz_h = tz / t_mag;
            Scalar tu = triple(lx, ly, lz, tx_h, ty_h, tz_h,
                               nx_h, ny_h, nz_h);
            // D^i tangent-to-edge reconstruction: D·t̂ = D_primal/|t_e|,
            // with D_primal = hodge1_inv·D̃[e].
            Scalar D_tangent = mp.hodge1_inv[e] * D_e[e] / t_mag;
            cross += mp.edge_sq_gamma_beta_r[e] * D_tangent * tu;
            count += 1;
          }
          if (count > 0) {
            h_aux -= dual_len * inv_sgma_f * cross / Scalar(count);
          }
          H_aux[f] = h_aux;
        });
      },
      D_in, B_in, m_H_aux);

  // Ampère: dD̃[e] = Σ_f d1t[e,f] · H_aux_line[f] − J̃[e].
  // Pure topological curl on the dual mesh — NO Hodge star here.
  ExecPolicy::launch(
      [Ne = mp.N_edges, mp] LAMBDA(auto H_aux, auto J_e, auto dD) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          Scalar curl_H = Scalar(0);
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            curl_H += mp.d1t_val[j] * H_aux[f];
          }
          dD[e] = curl_H - J_e[e];
        });
      },
      m_H_aux, m_J->data(), dD_out);
}

// =========================================================================
// Full RHS: dB from (D, B), then dD from (D, B).  Used by the
// semi-implicit predictor-corrector (not leapfrog — the corrector iterates
// both halves together).  For the explicit scheme use update_explicit,
// which calls compute_dB_dt and compute_dD_dt separately with the updated
// B to preserve the leapfrog structure.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::compute_rhs(
    buffer<Scalar>& D_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dD_dt, buffer<Scalar>& dB_dt) {
  compute_dB_dt(D_in, B_in, dB_dt);
  compute_dD_dt(D_in, B_in, dD_dt);
}

// =========================================================================
// Explicit leapfrog update.
//
//   (1)  dB = -d1·E_aux(D^n, B^n)             # Faraday at time n
//   (2)  B^{n+1} = B^n + dt·dB
//   (3)  dD = h1inv·(d1t·h2·H_aux(D^n, B^{n+1}) - J)
//                                              # Ampère using the NEW B
//   (4)  D^{n+1} = D^n + dt·dD
//
// The staggered order is essential: applying Ampère with the updated B
// (and leaving D^n in place for the H_aux cross term) gives the
// symplectic-Euler / leapfrog flavor that is stable under CFL for
// wave equations.  Using the same (D, B) for both half-steps (Forward
// Euler) is unconditionally unstable for wave systems.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::update_explicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Step 1-2: Faraday — compute dB from current state, advance B.
  if (m_update_b) {
    compute_dB_dt(m_D->data(), m_B->data(), m_dB_dt);
    ExecPolicy::launch(
        [Nf = mp.N_faces, dt] LAMBDA(auto B, auto dB) {
          ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
            B[f] += dt * dB[f];
          });
        },
        m_B->data(), m_dB_dt);
  }

  // Step 3-4: Ampère — compute dD using the updated B (leapfrog), advance D.
  if (m_update_d) {
    compute_dD_dt(m_D->data(), m_B->data(), m_dD_dt);
    ExecPolicy::launch(
        [Ne = mp.N_edges, dt] LAMBDA(auto D, auto dD) {
          ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
            D[e] += dt * dD[e];
          });
        },
        m_D->data(), m_dD_dt);
  }

  apply_damping(m_D->data(), m_B->data(), dt);
  apply_inner_boundary(m_D->data(), m_B->data());
  ExecPolicy::sync();
}

// =========================================================================
// Semi-implicit predictor-corrector update
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::update_semi_implicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  Scalar alpha = Scalar(1) - m_beta;
  Scalar beta = m_beta;

  compute_rhs(m_D->data(), m_B->data(), m_dD_dt, m_dB_dt);

  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces, dt]
      LAMBDA(auto D, auto tmpD, auto dD, auto B, auto tmpB, auto dB) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          tmpD[e] = D[e] + dt * dD[e];
        });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          tmpB[f] = B[f] + dt * dB[f];
        });
      },
      m_D->data(), m_tmp_D, m_dD_dt, m_B->data(), m_tmp_B, m_dB_dt);

  apply_inner_boundary(m_tmp_D, m_tmp_B);
  ExecPolicy::sync();

  for (int iter = 0; iter < m_implicit_iters; iter++) {
    compute_rhs(m_tmp_D, m_tmp_B, m_dD_dt_new, m_dB_dt_new);

    ExecPolicy::launch(
        [Ne = mp.N_edges, Nf = mp.N_faces, dt, alpha, beta]
        LAMBDA(auto D, auto tmpD, auto dD_n, auto dD_new,
               auto B, auto tmpB, auto dB_n, auto dB_new) {
          ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
            tmpD[e] = D[e] + dt * (alpha * dD_n[e] + beta * dD_new[e]);
          });
          ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
            tmpB[f] = B[f] + dt * (alpha * dB_n[f] + beta * dB_new[f]);
          });
        },
        m_D->data(), m_tmp_D, m_dD_dt, m_dD_dt_new,
        m_B->data(), m_tmp_B, m_dB_dt, m_dB_dt_new);

    apply_inner_boundary(m_tmp_D, m_tmp_B);
    ExecPolicy::sync();
  }

  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces]
      LAMBDA(auto D, auto tmpD, auto B, auto tmpB) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { D[e] = tmpD[e]; });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { B[f] = tmpB[f]; });
      },
      m_D->data(), m_tmp_D, m_B->data(), m_tmp_B);

  apply_damping(m_D->data(), m_B->data(), dt);
  apply_inner_boundary(m_D->data(), m_B->data());
  ExecPolicy::sync();
}

// =========================================================================
// Outer damping layer — relaxes (D - D_bg, B - B_bg) toward zero with an
// exponential rate that ramps up over the last m_damping_length radial
// layers.  If no background has been set (m_has_background = false) the
// target is zero, recovering the flat-space convention.
//
// Damping toward a stored background is essential for tests that start
// with a non-trivial equilibrium (e.g. Wald on a BH): damping toward
// zero would otherwise erase the uniform B₀ near the outer boundary and
// launch spurious inward-going waves from the resulting gradient.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::apply_damping(
    buffer<Scalar>& D, buffer<Scalar>& B, double dt) {
  if (m_damping_length <= 0) return;

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_r = mp.N_r;
  int k_start = N_r - m_damping_length;
  if (k_start < 1) k_start = 1;
  Scalar damp_coef = m_damping_coef;
  int damp_len = m_damping_length;
  bool has_bg = m_has_background;

  ExecPolicy::launch(
      [Ne = mp.N_edges, k_start, damp_coef, damp_len, dt, has_bg, mp]
      LAMBDA(auto D_e, auto D_bg) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          int k = mp.edge_radial_layer[e];
          if (k >= k_start) {
            Scalar sigma = damp_coef *
                           Scalar(k - k_start + 1) / Scalar(damp_len);
            Scalar factor = std::exp(-sigma * Scalar(dt));
            Scalar bg = has_bg ? D_bg[e] : Scalar(0);
            D_e[e] = bg + (D_e[e] - bg) * factor;
          }
        });
      },
      D, m_D_bg);

  ExecPolicy::launch(
      [Nf = mp.N_faces, k_start, damp_coef, damp_len, dt, has_bg, mp]
      LAMBDA(auto B_f, auto B_bg) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          int k = mp.face_radial_layer[f];
          if (k >= k_start) {
            Scalar sigma = damp_coef *
                           Scalar(k - k_start + 1) / Scalar(damp_len);
            Scalar factor = std::exp(-sigma * Scalar(dt));
            Scalar bg = has_bg ? B_bg[f] : Scalar(0);
            B_f[f] = bg + (B_f[f] - bg) * factor;
          }
        });
      },
      B, m_B_bg);
}

// =========================================================================
// Inner boundary condition — mirrors the 2D GR-KS solver's treatment
// (field_solver_gr_ks_mod_impl.hpp:725-742).
//
// Overwrites innermost-shell field values with an extrapolation from
// shell 1 that preserves the background gradient:
//
//   D_shell0 = D_shell1 + D0_shell1 - D0_shell0
//   B_slab0  = B_slab1  + B0_slab1  - B0_slab0
//
// Equivalently: the jump (field - background) across the inner layer is
// zero-gradient.  This is a Neumann-like BC on the perturbation δD, δB,
// preventing spurious radial gradients inside the horizon from driving
// numerical instability.  Applied after each time step (on both the
// evolved field and any intermediate buffers for the semi-implicit
// corrector).
//
// Element-kind mapping (from prismatic_mesh indexing):
//   horizontal edge on shell 0 (idx e ∈ [0, N_edge_s))
//     -> "interior" = horizontal edge on shell 1 at idx e + N_edge_s
//   vertical edge in slab 0 (idx e ∈ [(N_r+1)N_edge_s, +N_vert_s))
//     -> "interior" = vertical edge in slab 1 at idx e + N_vert_s
//   tri face on shell 0 (idx f ∈ [0, N_tri))
//     -> "interior" = tri face on shell 1 at idx f + N_tri
//   rect face in slab 0 (idx f ∈ [(N_r+1)N_tri, +N_edge_s))
//     -> "interior" = rect face in slab 1 at idx f + N_edge_s
//
// Requires m_has_background (populated by set_initial_wald).  If no
// background is set, damps δ to zero instead (i.e. extrapolates toward
// the implicit background of 0).
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::apply_inner_boundary(
    buffer<Scalar>& D, buffer<Scalar>& B) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  bool has_bg = m_has_background;
  int N_edge_s = mp.N_edge_s;
  int N_vert_s = mp.N_vert_s;
  int N_tri    = mp.N_tri;
  int v_edge_off = (mp.N_r + 1) * N_edge_s;      // first vertical edge idx
  int rect_face_off = (mp.N_r + 1) * N_tri;      // first rect face idx

  // Horizontal edges on shell 0: idx in [0, N_edge_s).
  // Interior counterpart on shell 1: idx + N_edge_s.
  ExecPolicy::launch(
      [N_edge_s, has_bg] LAMBDA(auto D_e, auto D_bg) {
        ExecPolicy::loop(0, N_edge_s, [&] LAMBDA(int e0) {
          int e1 = e0 + N_edge_s;
          Scalar bg0 = has_bg ? D_bg[e0] : Scalar(0);
          Scalar bg1 = has_bg ? D_bg[e1] : Scalar(0);
          D_e[e0] = D_e[e1] + bg1 - bg0;
        });
      },
      D, m_D_bg);

  // Vertical edges in slab 0: idx in [v_edge_off, v_edge_off + N_vert_s).
  // Interior counterpart in slab 1: idx + N_vert_s.
  ExecPolicy::launch(
      [N_vert_s, v_edge_off, has_bg] LAMBDA(auto D_e, auto D_bg) {
        ExecPolicy::loop(0, N_vert_s, [&] LAMBDA(int s) {
          int e0 = v_edge_off + s;
          int e1 = e0 + N_vert_s;
          Scalar bg0 = has_bg ? D_bg[e0] : Scalar(0);
          Scalar bg1 = has_bg ? D_bg[e1] : Scalar(0);
          D_e[e0] = D_e[e1] + bg1 - bg0;
        });
      },
      D, m_D_bg);

  // Tri faces on shell 0: idx in [0, N_tri).  Interior on shell 1: +N_tri.
  ExecPolicy::launch(
      [N_tri, has_bg] LAMBDA(auto B_f, auto B_bg) {
        ExecPolicy::loop(0, N_tri, [&] LAMBDA(int f0) {
          int f1 = f0 + N_tri;
          Scalar bg0 = has_bg ? B_bg[f0] : Scalar(0);
          Scalar bg1 = has_bg ? B_bg[f1] : Scalar(0);
          B_f[f0] = B_f[f1] + bg1 - bg0;
        });
      },
      B, m_B_bg);

  // Rect faces in slab 0: idx in [rect_face_off, rect_face_off + N_edge_s).
  // Interior in slab 1: +N_edge_s.
  ExecPolicy::launch(
      [N_edge_s, rect_face_off, has_bg] LAMBDA(auto B_f, auto B_bg) {
        ExecPolicy::loop(0, N_edge_s, [&] LAMBDA(int e) {
          int f0 = rect_face_off + e;
          int f1 = f0 + N_edge_s;
          Scalar bg0 = has_bg ? B_bg[f0] : Scalar(0);
          Scalar bg1 = has_bg ? B_bg[f1] : Scalar(0);
          B_f[f0] = B_f[f1] + bg1 - bg0;
        });
      },
      B, m_B_bg);
}

// =========================================================================
// Optional safety damping inside the horizon.
//
// This is NOT the physical inner boundary condition.  The primary inner
// BC for this solver is causal disconnection: with r_min placed inside
// the outer horizon r_+, the DEC stencil is self-closing at the inner
// boundary (every innermost element's Ampère/Faraday update finds all
// its required neighbors within the mesh), and GR causality guarantees
// that anything that happens at r < r_+ cannot propagate outward to the
// physics domain of interest.
//
// The only reason to enable this damping (r_horizon_damp > 0) is as
// numerical insurance: DEC discretization has O(h²) dispersion that can
// slightly exceed c at high-k modes, so in principle some numerical
// noise from inside r_+ could leak outward over long integration times.
// If that becomes a problem in practice, a light quadratic ramp here
// absorbs it before it can.
//
// Profile: for r_horizon_inner ≤ r < r_horizon_damp, fields are
// multiplied by  t² = ((r - r_horizon_inner) / (r_horizon_damp - r_horizon_inner))²
// which is 1 at the outer edge and 0 at r_horizon_inner.
// Disabled when m_r_horizon_damp <= 0.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::apply_horizon_damping(
    buffer<Scalar>& D, buffer<Scalar>& B) {
  if (m_r_horizon_damp <= Scalar(0)) return;

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  Scalar r_damp = m_r_horizon_damp;
  Scalar r_h = m_r_horizon_inner;

  ExecPolicy::launch(
      [Ne = mp.N_edges, r_damp, r_h, mp] LAMBDA(auto D_e) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          Scalar r = mp.edge_r_coord[e];
          if (r < r_damp) {
            Scalar t = (r - r_h) / (r_damp - r_h);
            if (t < Scalar(0)) t = Scalar(0);
            D_e[e] *= t * t;
          }
        });
      },
      D);

  ExecPolicy::launch(
      [Nf = mp.N_faces, r_damp, r_h, mp] LAMBDA(auto B_f) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          Scalar r = mp.face_r_coord[f];
          if (r < r_damp) {
            Scalar t = (r - r_h) / (r_damp - r_h);
            if (t < Scalar(0)) t = Scalar(0);
            B_f[f] *= t * t;
          }
        });
      },
      B);
}

// =========================================================================
// Wald initial condition: uniform B_z in Cartesian.
// Sets B[f] = B0 * (ẑ · face_area_vector); D[e] = 0.
// Also populates the background buffers (m_B_bg, m_D_bg) with the same
// values so the outer damping layer relaxes toward the Wald background
// rather than toward zero — essential for this test, since otherwise the
// damping layer would eat the uniform B₀ at the outer boundary and
// launch spurious gradient-driven waves inward.
// Runs on the device via ExecPolicy::launch.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::set_initial_wald(Scalar B0) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // B[f] = B0 * (ẑ · face_area_vector);  also copy into the background.
  ExecPolicy::launch(
      [Nf = mp.N_faces, B0, mp] LAMBDA(auto B_f, auto B_bg) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          Scalar fnx, fny, fnz;
          face_normal_area(mp, f, fnx, fny, fnz);
          Scalar v = B0 * fnz;
          B_f[f]  = v;
          B_bg[f] = v;
        });
      },
      m_B->data(), m_B_bg);

  // D[e] = 0;  D_bg = 0.
  ExecPolicy::launch(
      [Ne = mp.N_edges] LAMBDA(auto D_e, auto D_bg) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          D_e[e]  = Scalar(0);
          D_bg[e] = Scalar(0);
        });
      },
      m_D->data(), m_D_bg);

  m_has_background = true;
  ExecPolicy::sync();
}

}  // namespace Aperture
