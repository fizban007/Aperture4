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
      m_dB_dt_new(mesh.m_N_faces, ExecPolicy::data_mem_type()) {}

template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::register_data_components() {
  auto mem = ExecPolicy::data_mem_type();
  m_D = sim_env().template register_data<prismatic_edge_field>(
      "D", m_mesh, mem);
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
// Compute RHS with 3+1 constitutive relations for a radial-shift metric.
//
// The auxiliary fields encode:
//   E_aux_i = α D_i  +  ε_{ijk} (√γ β^j) B^k
//   H_aux_i = α B_i  -  ε_{ijk} (√γ β^j) D^k
//
// For a purely radial shift β^i = β^r ∂_r, the cross term vanishes on
// elements whose tangent / normal aligns with ∂_r:
//
//   - Vertical edges (tangent ∥ ∂_r):    E_aux = α D, no cross term.
//   - Triangular faces (normal ∥ ∂_r):   H_aux = α B, no cross term.
//   - Horizontal edges: cross term uses only adjacent rectangular faces.
//   - Rectangular faces: cross term uses only adjacent horizontal edges.
//
// The cross term is computed via the scalar triple product identity
//   (sgb × B) · t_e  =  |sgb| det(r̂, B, t_e)
// with r̂ = (midpoint)/|midpoint|, the spherical-coordinate radial
// direction at the element's location.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::compute_rhs(
    buffer<Scalar>& D_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dD_dt, buffer<Scalar>& dB_dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // -----------------------------------------------------------------------
  // Step 1a: E_aux for vertical edges — pure lapse scaling.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [Nh = mp.N_h_edges, Ne = mp.N_edges, mp]
      LAMBDA(auto D_e, auto E_aux) {
        ExecPolicy::loop(Nh, Ne, [&] LAMBDA(int e) {
          E_aux[e] = mp.edge_alpha[e] * D_e[e];
        });
      },
      D_in, m_E_aux);

  // -----------------------------------------------------------------------
  // Step 1b: E_aux for horizontal edges — lapse + shift cross term.
  //
  //   E_aux[e] = α[e] · D[e]
  //            + (√γ β^r)[e] · <det(r̂, B_rec, t)>
  //
  // where the average is over adjacent rectangular faces:
  //   det(r̂, B_rec, t) ≈ (1/N_rect) Σ B[f] · det(r̂, n_f, t_e) / |n_f|²
  //
  // Triangular face contributions vanish: n_tri ∥ r̂ → det(r̂, n_tri, ·) = 0.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [Nh = mp.N_h_edges, mp]
      LAMBDA(auto D_e, auto B_f, auto E_aux) {
        ExecPolicy::loop(0, Nh, [&] LAMBDA(int e) {
          Scalar e_aux = mp.edge_alpha[e] * D_e[e];

          // Edge tangent and the radial direction at the midpoint.
          Scalar tx, ty, tz;
          edge_tangent(mp, e, tx, ty, tz);
          Scalar mx, my, mz;
          edge_midpoint(mp, e, mx, my, mz);
          Scalar r = mp.edge_r_coord[e];
          Scalar lx = mx / r, ly = my / r, lz = mz / r;

          // Cross term: accumulate over adjacent rectangular faces only.
          Scalar cross = Scalar(0);
          int n_rect = 0;
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            if (mp.is_tri_face(f)) continue;

            Scalar nx, ny, nz;
            face_normal_area(mp, f, nx, ny, nz);
            Scalar n2 = nx*nx + ny*ny + nz*nz;
            cross += B_f[f] * triple(lx, ly, lz, nx, ny, nz,
                                     tx, ty, tz) / n2;
            n_rect++;
          }
          if (n_rect > 0) {
            e_aux += mp.edge_sq_gamma_beta_r[e] * cross / Scalar(n_rect);
          }
          E_aux[e] = e_aux;
        });
      },
      D_in, B_in, m_E_aux);

  // -----------------------------------------------------------------------
  // Step 2a: H_aux for triangular faces — pure lapse scaling.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [Ntri = mp.N_tri_faces, mp]
      LAMBDA(auto B_f, auto H_aux) {
        ExecPolicy::loop(0, Ntri, [&] LAMBDA(int f) {
          H_aux[f] = mp.face_alpha[f] * B_f[f];
        });
      },
      B_in, m_H_aux);

  // -----------------------------------------------------------------------
  // Step 2b: H_aux for rectangular faces — lapse - shift cross term.
  //
  //   H_aux[f] = α[f] · B[f]
  //            - (√γ β^r)[f] · <det(r̂, D_rec, n)>
  //
  // Vertical edge contributions vanish: t_vert ∥ r̂ → det(r̂, t_vert, ·) = 0.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [Ntri = mp.N_tri_faces, Nh = mp.N_h_edges, Nf = mp.N_faces, mp]
      LAMBDA(auto D_e, auto B_f, auto H_aux) {
        ExecPolicy::loop(Ntri, Nf, [&] LAMBDA(int f) {
          Scalar h_aux = mp.face_alpha[f] * B_f[f];

          Scalar nx, ny, nz;
          face_normal_area(mp, f, nx, ny, nz);
          Scalar cx, cy, cz;
          face_centroid(mp, f, cx, cy, cz);
          Scalar r = mp.face_r_coord[f];
          Scalar lx = cx / r, ly = cy / r, lz = cz / r;

          Scalar cross = Scalar(0);
          int n_horiz = 0;
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            int e = mp.d1_col_idx[j];
            if (mp.is_vertical_edge(e)) continue;

            Scalar tx, ty, tz;
            edge_tangent(mp, e, tx, ty, tz);
            Scalar t2 = tx*tx + ty*ty + tz*tz;
            cross += D_e[e] * triple(lx, ly, lz, tx, ty, tz,
                                     nx, ny, nz) / t2;
            n_horiz++;
          }
          if (n_horiz > 0) {
            h_aux -= mp.face_sq_gamma_beta_r[f] * cross / Scalar(n_horiz);
          }
          H_aux[f] = h_aux;
        });
      },
      D_in, B_in, m_H_aux);

  // -----------------------------------------------------------------------
  // Step 3: Faraday — dB[f]/dt = -Σ_e d1[f,e] · E_aux[e]
  // -----------------------------------------------------------------------
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
      m_E_aux, dB_dt);

  // -----------------------------------------------------------------------
  // Step 4: Ampère — dD[e]/dt = h1inv[e] · (Σ_f d1t[e,f] · h2[f] · H_aux[f] - J[e])
  //
  // The Hodge stars here are already metric-weighted (computed from the
  // user-supplied metric at mesh build time).  The shift cross-coupling
  // that a pure Hodge star cannot represent is folded into H_aux.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [Ne = mp.N_edges, mp] LAMBDA(auto H_aux, auto J_e, auto dD) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          Scalar curl_H = Scalar(0);
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            curl_H += mp.d1t_val[j] * mp.hodge2[f] * H_aux[f];
          }
          dD[e] = mp.hodge1_inv[e] * (curl_H - J_e[e]);
        });
      },
      m_H_aux, m_J->data(), dD_dt);
}

// =========================================================================
// Explicit leapfrog update
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::update_explicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  compute_rhs(m_D->data(), m_B->data(), m_dD_dt, m_dB_dt);

  if (m_update_b) {
    ExecPolicy::launch(
        [Nf = mp.N_faces, dt] LAMBDA(auto B, auto dB) {
          ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
            B[f] += dt * dB[f];
          });
        },
        m_B->data(), m_dB_dt);
  }

  if (m_update_d) {
    ExecPolicy::launch(
        [Ne = mp.N_edges, dt] LAMBDA(auto D, auto dD) {
          ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
            D[e] += dt * dD[e];
          });
        },
        m_D->data(), m_dD_dt);
  }

  apply_damping(m_D->data(), m_B->data(), dt);
  apply_horizon_damping(m_D->data(), m_B->data());
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

  apply_horizon_damping(m_tmp_D, m_tmp_B);
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

    apply_horizon_damping(m_tmp_D, m_tmp_B);
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
  apply_horizon_damping(m_D->data(), m_B->data());
  ExecPolicy::sync();
}

// =========================================================================
// Damping layer (outer boundary absorption)
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

  ExecPolicy::launch(
      [Ne = mp.N_edges, k_start, damp_coef, damp_len, dt, mp]
      LAMBDA(auto D_e) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          int k = mp.edge_radial_layer[e];
          if (k >= k_start) {
            Scalar sigma = damp_coef *
                           Scalar(k - k_start + 1) / Scalar(damp_len);
            D_e[e] *= std::exp(-sigma * Scalar(dt));
          }
        });
      },
      D);

  ExecPolicy::launch(
      [Nf = mp.N_faces, k_start, damp_coef, damp_len, dt, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          int k = mp.face_radial_layer[f];
          if (k >= k_start) {
            Scalar sigma = damp_coef *
                           Scalar(k - k_start + 1) / Scalar(damp_len);
            B_f[f] *= std::exp(-sigma * Scalar(dt));
          }
        });
      },
      B);
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
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::set_initial_wald(Scalar B0) {
  auto mp = m_mesh.host_ptrs_metric();

  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    Scalar fnx, fny, fnz;
    face_normal_area(mp, f, fnx, fny, fnz);
    m_B->data()[f] = B0 * fnz;
  }
  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    m_D->data()[e] = Scalar(0);
  }
}

}  // namespace Aperture
