#pragma once

#include "systems/prismatic/dec_field_solver_gr_ks.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/physics/metric_ks_cartesian.hpp"
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
// Helper: compute the edge tangent vector (v1 - v0, unnormalized).
// =========================================================================
HD_INLINE void edge_tangent(const prismatic_mesh_gr_ks_ptrs& mp, int e,
                            Scalar& tx, Scalar& ty, Scalar& tz) {
  int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
  tx = mp.vert_x[v1] - mp.vert_x[v0];
  ty = mp.vert_y[v1] - mp.vert_y[v0];
  tz = mp.vert_z[v1] - mp.vert_z[v0];
}

// =========================================================================
// Helper: compute the face normal area vector (unnormalized).
//   For triangles: 0.5 * (edge1 × edge2)
//   For quads: edge1 × edge2  (full parallelogram)
// =========================================================================
HD_INLINE void face_normal_area(const prismatic_mesh_gr_ks_ptrs& mp, int f,
                                Scalar& nx, Scalar& ny, Scalar& nz) {
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);
  if (f < n_tri_faces) {
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
    int local = f - n_tri_faces;
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
// Helper: project a Cartesian vector onto a primal edge (line integral).
//   Returns V · (v1 - v0), i.e. V_i * dx^i along the edge.
// =========================================================================
HD_INLINE Scalar project_on_edge(const prismatic_mesh_gr_ks_ptrs& mp, int e,
                                 Scalar Vx, Scalar Vy, Scalar Vz) {
  int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
  return Vx * (mp.vert_x[v1] - mp.vert_x[v0]) +
         Vy * (mp.vert_y[v1] - mp.vert_y[v0]) +
         Vz * (mp.vert_z[v1] - mp.vert_z[v0]);
}

// =========================================================================
// Constructor
// =========================================================================
template <typename ExecPolicy>
dec_field_solver_gr_ks<ExecPolicy>::dec_field_solver_gr_ks(
    prismatic_mesh_gr_ks& mesh)
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
  m_a = m_mesh.m_a;
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
  sim_env().params().get_value("update_d", m_update_d);
  sim_env().params().get_value("update_b", m_update_b);
  sim_env().params().get_value("use_implicit", m_use_implicit);
  sim_env().params().get_value("implicit_beta", m_beta);
  sim_env().params().get_value("implicit_iters", m_implicit_iters);

  // Default: damp inside 1.05 * r_+
  m_r_horizon_damp = Scalar(1.05) * Metric_KS_Cart::rH(m_a);
  sim_env().params().get_value("r_horizon_damp", m_r_horizon_damp);

  m_time = 0.0;
  Logger::print_info(
      "DEC GR KS field solver initialized: a={}, r_horizon_damp={}, "
      "implicit={}",
      m_a, m_r_horizon_damp, m_use_implicit);
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
// Compute RHS with GR 3+1 constitutive relations
//
// The auxiliary fields encode:
//   E_aux_i = alpha * D_i  +  epsilon_{ijk} (sqrt(gamma) beta^j) B^k
//   H_aux_i = alpha * B_i  -  epsilon_{ijk} (sqrt(gamma) beta^j) D^k
//
// Key simplification: the KS shift vector is exactly radial (beta ∝ l).
// Therefore  sgb × V  is always perpendicular to l.  This means:
//
//   - Vertical edges (tangent ∥ l):  (sgb × B) · t = 0  exactly.
//     → E_aux = alpha * D, no cross term.
//
//   - Triangular faces (normal ∥ l):  (sgb × D) · n = 0  exactly.
//     → H_aux = alpha * B, no cross term.
//
//   - Horizontal edges: cross term uses only adjacent rectangular faces.
//     Triangular face contributions vanish because det(l, n_tri, t) = 0
//     when n_tri ∥ l.
//
//   - Rectangular faces: cross term uses only adjacent horizontal edges.
//     Vertical edge contributions vanish because det(l, t_vert, n) = 0
//     when t_vert ∥ l.
//
// The cross terms are computed via the scalar triple product identity:
//   (sgb × B) · t = (f/sqrt(1+f)) * det(l, B, t)
// without ever reconstructing the full B vector.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::compute_rhs(
    buffer<Scalar>& D_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dD_dt, buffer<Scalar>& dB_dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  int N_h_edges = (mp.N_r + 1) * mp.N_edge_s;    // horizontal edge count
  int N_tri_faces = (mp.N_r + 1) * mp.N_tri;      // triangular face count

  // -----------------------------------------------------------------------
  // Step 1a: E_aux for vertical edges — pure lapse scaling, no cross term.
  //
  //   E_aux[e] = alpha[e] * D[e]
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_h_edges, N_edges = mp.N_edges, mp]
      LAMBDA(auto D_e, auto E_aux) {
        ExecPolicy::loop(N_h_edges, N_edges, [&] LAMBDA(int e) {
          E_aux[e] = mp.edge_alpha[e] * D_e[e];
        });
      },
      D_in, m_E_aux);

  // -----------------------------------------------------------------------
  // Step 1b: E_aux for horizontal edges — lapse + shift cross term.
  //
  //   E_aux[e] = alpha[e] * D[e]
  //            + (f/sqrt(1+f)) * <det(l, B_rec, t)>
  //
  // where the average is over adjacent rectangular faces:
  //   det(l, B_rec, t) ≈ (1/N_rect) Σ_{rect f adj e}
  //                         B[f] * det(l_e, n_f, t_e) / |n_f|²
  //
  // (Triangular face contributions vanish: n_tri ∥ l ⟹ det = 0.)
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_h_edges, N_tri_faces, mp]
      LAMBDA(auto D_e, auto B_f, auto E_aux) {
        ExecPolicy::loop(0, N_h_edges, [&] LAMBDA(int e) {
          // Diagonal term
          Scalar e_aux = mp.edge_alpha[e] * D_e[e];

          // Edge tangent vector t = v1 - v0
          Scalar tx, ty, tz;
          edge_tangent(mp, e, tx, ty, tz);

          // Null vector l at edge midpoint
          Scalar lx = mp.edge_lx[e];
          Scalar ly = mp.edge_ly[e];
          Scalar lz = mp.edge_lz[e];

          // Cross term: accumulate from adjacent rectangular faces only
          Scalar cross = Scalar(0);
          int n_rect = 0;
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            if (f < N_tri_faces) continue;  // skip triangular faces

            // Rectangular face normal area vector
            Scalar nx, ny, nz;
            face_normal_area(mp, f, nx, ny, nz);
            Scalar n2 = nx*nx + ny*ny + nz*nz;

            // det(l, n_f, t_e) / |n_f|^2 * B[f]
            // = B[f] * l · (n_f × t_e) / |n_f|^2
            cross += B_f[f] * triple(lx, ly, lz, nx, ny, nz,
                                     tx, ty, tz) / n2;
            n_rect++;
          }
          if (n_rect > 0) {
            // f_e * alpha_e = f / sqrt(1+f) = |sgb|
            e_aux += mp.edge_f[e] * mp.edge_alpha[e] *
                     cross / Scalar(n_rect);
          }

          E_aux[e] = e_aux;
        });
      },
      D_in, B_in, m_E_aux);

  // -----------------------------------------------------------------------
  // Step 2a: H_aux for triangular faces — pure lapse scaling, no cross term.
  //
  //   H_aux[f] = alpha[f] * B[f]
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_tri_faces, mp]
      LAMBDA(auto B_f, auto H_aux) {
        ExecPolicy::loop(0, N_tri_faces, [&] LAMBDA(int f) {
          H_aux[f] = mp.face_alpha[f] * B_f[f];
        });
      },
      B_in, m_H_aux);

  // -----------------------------------------------------------------------
  // Step 2b: H_aux for rectangular faces — lapse - shift cross term.
  //
  //   H_aux[f] = alpha[f] * B[f]
  //            - (f/sqrt(1+f)) * <det(l, D_rec, n)>
  //
  // where the average is over adjacent horizontal edges:
  //   det(l, D_rec, n) ≈ (1/N_horiz) Σ_{horiz e adj f}
  //                         D[e] * det(l_f, t_e, n_f) / |t_e|²
  //
  // (Vertical edge contributions vanish: t_vert ∥ l ⟹ det = 0.)
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_h_edges, N_tri_faces, N_faces = mp.N_faces, mp]
      LAMBDA(auto D_e, auto B_f, auto H_aux) {
        ExecPolicy::loop(N_tri_faces, N_faces, [&] LAMBDA(int f) {
          // Diagonal term
          Scalar h_aux = mp.face_alpha[f] * B_f[f];

          // Face normal area vector
          Scalar nx, ny, nz;
          face_normal_area(mp, f, nx, ny, nz);

          // Null vector l at face centroid
          Scalar lx = mp.face_lx[f];
          Scalar ly = mp.face_ly[f];
          Scalar lz = mp.face_lz[f];

          // Cross term: accumulate from adjacent horizontal edges only
          Scalar cross = Scalar(0);
          int n_horiz = 0;
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            int e = mp.d1_col_idx[j];
            if (e >= N_h_edges) continue;  // skip vertical edges

            Scalar tx, ty, tz;
            edge_tangent(mp, e, tx, ty, tz);
            Scalar t2 = tx*tx + ty*ty + tz*tz;

            // det(l, t_e, n_f) / |t_e|^2 * D[e]
            cross += D_e[e] * triple(lx, ly, lz, tx, ty, tz,
                                     nx, ny, nz) / t2;
            n_horiz++;
          }
          if (n_horiz > 0) {
            h_aux -= mp.face_f[f] * mp.face_alpha[f] *
                     cross / Scalar(n_horiz);
          }

          H_aux[f] = h_aux;
        });
      },
      D_in, B_in, m_H_aux);

  // -----------------------------------------------------------------------
  // Step 3: Faraday — dB[f]/dt = -sum_e d1[f,e] * E_aux[e]
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_faces = mp.N_faces, mp] LAMBDA(auto E_aux, auto dB) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          Scalar curl_E = Scalar(0);
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            curl_E += mp.d1_val[j] * E_aux[mp.d1_col_idx[j]];
          }
          dB[f] = -curl_E;
        });
      },
      m_E_aux, dB_dt);

  // -----------------------------------------------------------------------
  // Step 4: Ampere — dD[e]/dt = hodge1_inv * (d1t * hodge2 * H_aux - J)
  //
  // The Hodge stars here still carry the flat geometric normalization.
  // The metric-dependent part is already folded into H_aux via the
  // constitutive relation.  This is the "auxiliary field" approach:
  // the flat DEC structure is preserved, with all GR physics in the
  // auxiliary fields.
  // -----------------------------------------------------------------------
  ExecPolicy::launch(
      [N_edges = mp.N_edges, mp] LAMBDA(auto H_aux, auto J_e, auto dD) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
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

  // Full RHS computation (auxiliary fields + curls)
  compute_rhs(m_D->data(), m_B->data(), m_dD_dt, m_dB_dt);

  // Advance B
  if (m_update_b) {
    ExecPolicy::launch(
        [Nf = mp.N_faces, dt] LAMBDA(auto B, auto dB) {
          ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
            B[f] += dt * dB[f];
          });
        },
        m_B->data(), m_dB_dt);
  }

  // Advance D
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
  apply_horizon_bc(m_D->data(), m_B->data());
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

  // Step 1: RHS at current state
  compute_rhs(m_D->data(), m_B->data(), m_dD_dt, m_dB_dt);

  // Step 2: Euler predict
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

  apply_horizon_bc(m_tmp_D, m_tmp_B);
  ExecPolicy::sync();

  // Step 3: Corrector iterations
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

    apply_horizon_bc(m_tmp_D, m_tmp_B);
    ExecPolicy::sync();
  }

  // Step 4: Copy result
  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces]
      LAMBDA(auto D, auto tmpD, auto B, auto tmpB) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) { D[e] = tmpD[e]; });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) { B[f] = tmpB[f]; });
      },
      m_D->data(), m_tmp_D, m_B->data(), m_tmp_B);

  apply_damping(m_D->data(), m_B->data(), dt);
  apply_horizon_bc(m_D->data(), m_B->data());
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
      [N_edges = mp.N_edges, k_start, damp_coef, damp_len, dt, mp]
      LAMBDA(auto D_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
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
      [N_faces = mp.N_faces, k_start, damp_coef, damp_len, dt, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
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
// Horizon boundary condition: damp fields inside r_horizon_damp
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::apply_horizon_bc(
    buffer<Scalar>& D, buffer<Scalar>& B) {
  if (m_r_horizon_damp <= Scalar(0)) return;

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  Scalar r_damp = m_r_horizon_damp;
  Scalar r_h = Metric_KS_Cart::rH(m_a);

  ExecPolicy::launch(
      [N_edges = mp.N_edges, r_damp, r_h, mp] LAMBDA(auto D_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          Scalar r = mp.edge_r[e];
          if (r < r_damp) {
            // Smooth ramp from 1 at r_damp to 0 at r_h
            Scalar t = (r - r_h) / (r_damp - r_h);
            if (t < Scalar(0)) t = Scalar(0);
            D_e[e] *= t * t;
          }
        });
      },
      D);

  ExecPolicy::launch(
      [N_faces = mp.N_faces, r_damp, r_h, mp] LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          Scalar r = mp.face_r[f];
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
// Wald initial condition: uniform B_z on a Kerr background.
//
// In flat space this is B = B0 * z_hat.  The DEC cochain B[f] is the
// flux of B through each primal face:  B[f] = B0 * (z_hat · dA_f).
//
// The Wald solution also induces an E field from frame-dragging:
//   E = -v × B  where v comes from the shift vector.
// For the initial condition, D[e] = 0 is a valid choice — the solver
// will self-consistently develop the correct E from the evolution.
// =========================================================================
template <typename ExecPolicy>
void dec_field_solver_gr_ks<ExecPolicy>::set_initial_wald(Scalar B0) {
  auto mp = m_mesh.host_ptrs_gr();

  // Set B[f] = B0 * (z_hat · face_normal_area)
  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    Scalar fnx, fny, fnz;
    face_normal_area(mp, f, fnx, fny, fnz);
    m_B->data()[f] = B0 * fnz;
  }

  // D = 0 initially
  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    m_D->data()[e] = Scalar(0);
  }
}

}  // namespace Aperture
