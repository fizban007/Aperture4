#pragma once

#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "framework/environment.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Device-callable helper functions
// =========================================================================

HD_INLINE void dipole_B_impl(Scalar x, Scalar y, Scalar z,
                              Scalar mx, Scalar my, Scalar mz,
                              Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r5 = r2*r2*r;
  Scalar mdotr = mx*x + my*y + mz*z;
  Scalar factor = Scalar(3.0) * mdotr / r5;
  Scalar r3 = r2*r;
  Bx = factor*x - mx/r3;
  By = factor*y - my/r3;
  Bz = factor*z - mz/r3;
}

HD_INLINE Scalar project_B_on_face_impl(const prismatic_mesh_ptrs& mp, int f,
                                         Scalar Bx, Scalar By, Scalar Bz) {
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);
  if (f < n_tri_faces) {
    int va = mp.tri_face_v0[f], vb = mp.tri_face_v1[f], vc = mp.tri_face_v2[f];
    Scalar ax = mp.vert_x[vb]-mp.vert_x[va], ay = mp.vert_y[vb]-mp.vert_y[va], az = mp.vert_z[vb]-mp.vert_z[va];
    Scalar bx = mp.vert_x[vc]-mp.vert_x[va], by = mp.vert_y[vc]-mp.vert_y[va], bz = mp.vert_z[vc]-mp.vert_z[va];
    return Scalar(0.5) * (Bx*(ay*bz-az*by) + By*(az*bx-ax*bz) + Bz*(ax*by-ay*bx));
  }
  int local = f - n_tri_faces;
  int va = mp.rect_face_v0[local], vb = mp.rect_face_v1[local], vd = mp.rect_face_v3[local];
  Scalar ax = mp.vert_x[vb]-mp.vert_x[va], ay = mp.vert_y[vb]-mp.vert_y[va], az = mp.vert_z[vb]-mp.vert_z[va];
  Scalar bx = mp.vert_x[vd]-mp.vert_x[va], by = mp.vert_y[vd]-mp.vert_y[va], bz = mp.vert_z[vd]-mp.vert_z[va];
  return Bx*(ay*bz-az*by) + By*(az*bx-ax*bz) + Bz*(ax*by-ay*bx);
}

HD_INLINE Scalar project_E_on_edge_impl(const prismatic_mesh_ptrs& mp, int e,
                                         Scalar Ex, Scalar Ey, Scalar Ez) {
  int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
  return Ex*(mp.vert_x[v1]-mp.vert_x[v0]) +
         Ey*(mp.vert_y[v1]-mp.vert_y[v0]) +
         Ez*(mp.vert_z[v1]-mp.vert_z[v0]);
}

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
  m_E = sim_env().template register_data<prismatic_edge_field>(
      "E", m_mesh, mem);
  m_B = sim_env().template register_data<prismatic_face_field>(
      "B", m_mesh, mem);
  m_J = sim_env().template register_data<prismatic_edge_field>(
      "J", m_mesh, mem);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::init() {
  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Omega", m_Omega);
  sim_env().params().get_value("obliquity", m_obliquity);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
  sim_env().params().get_value("update_e", m_update_e);
  sim_env().params().get_value("update_b", m_update_b);
  sim_env().params().get_value("use_implicit", m_use_implicit);
  sim_env().params().get_value("implicit_beta", m_beta);
  sim_env().params().get_value("implicit_iters", m_implicit_iters);

  set_initial_dipole();

  m_E->data().copy_to_device();
  m_B->data().copy_to_device();

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
  m_time += dt;
}

// =========================================================================
// Compute RHS: dB/dt = -d1*E, dE/dt = h1inv*(d1t*h2*B - J)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::compute_rhs(
    buffer<Scalar>& E_in, buffer<Scalar>& B_in,
    buffer<Scalar>& dE_out, buffer<Scalar>& dB_out) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // dB/dt = -d1 * E
  ExecPolicy::launch(
      [N_faces = mp.N_faces, mp] LAMBDA(auto E_e, auto dB) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          Scalar curl_E = Scalar(0);
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            curl_E += mp.d1_val[j] * E_e[mp.d1_col_idx[j]];
          }
          dB[f] = -curl_E;
        });
      },
      E_in, dB_out);

  // dE/dt = h1inv * (d1t * h2 * B - J)
  ExecPolicy::launch(
      [N_edges = mp.N_edges, mp] LAMBDA(auto B_f, auto J_e, auto dE) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          Scalar curl_H = Scalar(0);
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            curl_H += mp.d1t_val[j] * mp.hodge2[f] * B_f[f];
          }
          dE[e] = mp.hodge1_inv[e] * (curl_H - J_e[e]);
        });
      },
      B_in, m_J->data(), dE_out);
}

// =========================================================================
// Explicit update (original leapfrog)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update_explicit(double dt) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});

  // Faraday: B -= dt * d1 * E
  if (m_update_b) {
    ExecPolicy::launch(
        [N_faces = mp.N_faces, dt, mp] LAMBDA(auto E_e, auto B_f) {
          ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
            Scalar curl_E = Scalar(0);
            for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
              curl_E += mp.d1_val[j] * E_e[mp.d1_col_idx[j]];
            }
            B_f[f] -= dt * curl_E;
          });
        },
        m_E->data(), m_B->data());
  }

  // Ampere: E += dt * h1inv * (d1t * h2 * B - J)
  if (m_update_e) {
    ExecPolicy::launch(
        [N_edges = mp.N_edges, dt, mp] LAMBDA(auto E_e, auto B_f, auto J_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          Scalar curl_H = Scalar(0);
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            curl_H += mp.d1t_val[j] * mp.hodge2[f] * B_f[f];
          }
          E_e[e] += dt * mp.hodge1_inv[e] * (curl_H - J_e[e]);
        });
      },
      m_E->data(), m_B->data(), m_J->data());
  }

  apply_damping(m_E->data(), m_B->data(), dt);
  apply_inner_bc(m_E->data(), m_B->data(), m_time + dt);
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
  ExecPolicy::launch(
      [Ne = mp.N_edges, Nf = mp.N_faces, dt]
      LAMBDA(auto E, auto tmpE, auto dE, auto B, auto tmpB, auto dB) {
        ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
          tmpE[e] = E[e] + dt * dE[e];
        });
        ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
          tmpB[f] = B[f] + dt * dB[f];
        });
      },
      m_E->data(), m_tmp_E, m_dE_dt, m_B->data(), m_tmp_B, m_dB_dt);

  // Apply BC to the Euler predict so the first RHS evaluation is consistent
  apply_inner_bc(m_tmp_E, m_tmp_B, m_time + dt);
  ExecPolicy::sync();

  // Step 3: Iterate corrector
  for (int iter = 0; iter < m_implicit_iters; iter++) {
    compute_rhs(m_tmp_E, m_tmp_B, m_dE_dt_new, m_dB_dt_new);

    // F* = F^n + dt * (alpha * RHS^n + beta * RHS*)
    ExecPolicy::launch(
        [Ne = mp.N_edges, Nf = mp.N_faces, dt, alpha, beta]
        LAMBDA(auto E, auto tmpE, auto dE_n, auto dE_new,
               auto B, auto tmpB, auto dB_n, auto dB_new) {
          ExecPolicy::loop(0, Ne, [&] LAMBDA(int e) {
            tmpE[e] = E[e] + dt * (alpha * dE_n[e] + beta * dE_new[e]);
          });
          ExecPolicy::loop(0, Nf, [&] LAMBDA(int f) {
            tmpB[f] = B[f] + dt * (alpha * dB_n[f] + beta * dB_new[f]);
          });
        },
        m_E->data(), m_tmp_E, m_dE_dt, m_dE_dt_new,
        m_B->data(), m_tmp_B, m_dB_dt, m_dB_dt_new);

    apply_damping(m_E->data(), m_B->data(), dt);
    apply_inner_bc(m_tmp_E, m_tmp_B, m_time + dt);
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

  // Step 5: Damping + BC + clear J
  apply_damping(m_E->data(), m_B->data(), dt);
  apply_inner_bc(m_E->data(), m_B->data(), m_time + dt);
  ExecPolicy::sync();
}

// =========================================================================
// Damping — runs on GPU
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_damping(
    buffer<Scalar>& E, buffer<Scalar>& B, double dt) {
  if (m_damping_length <= 0) return;

  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_r = mp.N_r;
  int k_start = N_r - m_damping_length;
  if (k_start < 1) k_start = 1;
  Scalar damp_coef = m_damping_coef;
  int damp_len = m_damping_length;

  // Damp E on edges
  ExecPolicy::launch(
      [N_edges = mp.N_edges, k_start, damp_coef, damp_len, dt, mp]
      LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          int k = mp.edge_radial_layer[e];
          if (k >= k_start) {
            Scalar sigma = damp_coef *
                           Scalar(k - k_start + 1) / Scalar(damp_len);
            E_e[e] *= std::exp(-sigma * Scalar(dt));
          }
        });
      },
      E);

  // Damp B on faces
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
// Inner boundary condition — runs on GPU
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_inner_bc(
    buffer<Scalar>& E, buffer<Scalar>& B, double time) {
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  Scalar mx = m_Bp * std::sin(m_obliquity) * std::cos(m_Omega * time);
  Scalar my = m_Bp * std::sin(m_obliquity) * std::sin(m_Omega * time);
  Scalar mz = m_Bp * std::cos(m_obliquity);
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);
  Scalar Omega = m_Omega;

  // Overwrite B_f on inner boundary faces
  ExecPolicy::launch(
      [N_faces = mp.N_faces, n_tri_faces, mx, my, mz, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          if (mp.face_boundary[f] != 1) return;
          Scalar fx, fy, fz;
          if (f < n_tri_faces) {
            int v0 = mp.tri_face_v0[f], v1 = mp.tri_face_v1[f],
                v2 = mp.tri_face_v2[f];
            fx = (mp.vert_x[v0]+mp.vert_x[v1]+mp.vert_x[v2]) / Scalar(3.0);
            fy = (mp.vert_y[v0]+mp.vert_y[v1]+mp.vert_y[v2]) / Scalar(3.0);
            fz = (mp.vert_z[v0]+mp.vert_z[v1]+mp.vert_z[v2]) / Scalar(3.0);
          } else {
            int local = f - n_tri_faces;
            int v0 = mp.rect_face_v0[local], v1 = mp.rect_face_v1[local],
                v2 = mp.rect_face_v2[local], v3 = mp.rect_face_v3[local];
            fx = (mp.vert_x[v0]+mp.vert_x[v1]+mp.vert_x[v2]+mp.vert_x[v3]) / Scalar(4.0);
            fy = (mp.vert_y[v0]+mp.vert_y[v1]+mp.vert_y[v2]+mp.vert_y[v3]) / Scalar(4.0);
            fz = (mp.vert_z[v0]+mp.vert_z[v1]+mp.vert_z[v2]+mp.vert_z[v3]) / Scalar(4.0);
          }
          Scalar Bx, By, Bz;
          dipole_B_impl(fx, fy, fz, mx, my, mz, Bx, By, Bz);
          B_f[f] = project_B_on_face_impl(mp, f, Bx, By, Bz);
        });
      },
      B);

  // Overwrite E_e on inner boundary edges
  ExecPolicy::launch(
      [N_edges = mp.N_edges, mx, my, mz, Omega, mp]
      LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          if (mp.edge_boundary[e] != 1) return;
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          Scalar ex = (mp.vert_x[v0]+mp.vert_x[v1]) / Scalar(2.0);
          Scalar ey = (mp.vert_y[v0]+mp.vert_y[v1]) / Scalar(2.0);
          Scalar ez = (mp.vert_z[v0]+mp.vert_z[v1]) / Scalar(2.0);
          Scalar Bx, By, Bz;
          dipole_B_impl(ex, ey, ez, mx, my, mz, Bx, By, Bz);
          Scalar vx = -Omega * ey;
          Scalar vy = Omega * ex;
          Scalar Ex = -(vy * Bz);
          Scalar Ey = -(- vx * Bz);
          Scalar Ez = -(vx * By - vy * Bx);
          E_e[e] = project_E_on_edge_impl(mp, e, Ex, Ey, Ez);
        });
      },
      E);
}

// =========================================================================
// Initial dipole (host-only, called once during init)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_dipole() {
  double mx = m_Bp * std::sin(m_obliquity);
  double my = 0.0;
  double mz = m_Bp * std::cos(m_obliquity);

  // Helper: dipole B field at (x,y,z) in double precision
  auto dip_B = [mx, my, mz](double x, double y, double z,
                             double& Bx, double& By, double& Bz) {
    double r2 = x*x + y*y + z*z;
    double r = std::sqrt(r2);
    double r5 = r2*r2*r;
    double r3 = r2*r;
    double mdotr = mx*x + my*y + mz*z;
    double fac = 3.0*mdotr/r5;
    Bx = fac*x - mx/r3;
    By = fac*y - my/r3;
    Bz = fac*z - mz/r3;
  };

  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);

  // Triangular faces: ∫_triangle B · dA via Gauss quadrature
  // Parameterize: x(u,v) = p0 + u*(p1-p0) + v*(p2-p0), 0≤u, 0≤v, u+v≤1
  // dA = (p1-p0) × (p2-p0) du dv (constant for flat triangle)
  // ∫∫ B·dA = (∫₀¹ ∫₀^(1-u) B(x(u,v)) du dv) · n
  // where n = (p1-p0) × (p2-p0)
  for (int f = 0; f < n_tri_faces; f++) {
    int vi0 = m_mesh.tri_face_v0[f];
    int vi1 = m_mesh.tri_face_v1[f];
    int vi2 = m_mesh.tri_face_v2[f];
    double p0x = m_mesh.vert_x[vi0], p0y = m_mesh.vert_y[vi0], p0z = m_mesh.vert_z[vi0];
    double p1x = m_mesh.vert_x[vi1], p1y = m_mesh.vert_y[vi1], p1z = m_mesh.vert_z[vi1];
    double p2x = m_mesh.vert_x[vi2], p2y = m_mesh.vert_y[vi2], p2z = m_mesh.vert_z[vi2];

    // Face normal (unnormalized, includes area factor)
    double e1x = p1x-p0x, e1y = p1y-p0y, e1z = p1z-p0z;
    double e2x = p2x-p0x, e2y = p2y-p0y, e2z = p2z-p0z;
    double nx = e1y*e2z - e1z*e2y;
    double ny = e1z*e2x - e1x*e2z;
    double nz = e1x*e2y - e1y*e2x;

    // Integrate B·n̂ over the triangle using nested Gauss quadrature
    // Map triangle to unit square: u ∈ [0,1], v ∈ [0, 1-u]
    // ∫₀¹ du ∫₀^{1-u} dv f(u,v) = ∫₀¹ du (1-u) ∫₀¹ dt f(u, (1-u)t)
    double flux = gauss_quad([&](double u) -> double {
      return (1.0 - u) * gauss_quad([&](double t) -> double {
        double v = (1.0 - u) * t;
        double x = p0x + u*e1x + v*e2x;
        double y = p0y + u*e1y + v*e2y;
        double z = p0z + u*e1z + v*e2z;
        double Bx, By, Bz;
        dip_B(x, y, z, Bx, By, Bz);
        return Bx*nx + By*ny + Bz*nz;
      }, 0.0, 1.0);
    }, 0.0, 1.0);

    m_B->data()[f] = static_cast<Scalar>(flux);
  }

  // Rectangular faces: ∫_quad B · dA via tensor product Gauss quadrature
  // Parameterize: x(u,v) = (1-u)(1-v)p0 + u(1-v)p1 + uv p2 + (1-u)v p3
  // with u,v ∈ [0,1]. The Jacobian cross product gives the area element.
  for (int fi = 0; fi < m_mesh.m_N_edge_s * m_mesh.m_N_r; fi++) {
    int f = n_tri_faces + fi;
    int vi0 = m_mesh.rect_face_v0[fi], vi1 = m_mesh.rect_face_v1[fi];
    int vi2 = m_mesh.rect_face_v2[fi], vi3 = m_mesh.rect_face_v3[fi];
    double p0x = m_mesh.vert_x[vi0], p0y = m_mesh.vert_y[vi0], p0z = m_mesh.vert_z[vi0];
    double p1x = m_mesh.vert_x[vi1], p1y = m_mesh.vert_y[vi1], p1z = m_mesh.vert_z[vi1];
    double p2x = m_mesh.vert_x[vi2], p2y = m_mesh.vert_y[vi2], p2z = m_mesh.vert_z[vi2];
    double p3x = m_mesh.vert_x[vi3], p3y = m_mesh.vert_y[vi3], p3z = m_mesh.vert_z[vi3];

    double flux = gauss_quad([&](double u) -> double {
      return gauss_quad([&](double v) -> double {
        // Bilinear interpolation
        double x = (1-u)*(1-v)*p0x + u*(1-v)*p1x + u*v*p2x + (1-u)*v*p3x;
        double y = (1-u)*(1-v)*p0y + u*(1-v)*p1y + u*v*p2y + (1-u)*v*p3y;
        double z = (1-u)*(1-v)*p0z + u*(1-v)*p1z + u*v*p2z + (1-u)*v*p3z;

        // Tangent vectors dx/du and dx/dv
        double dxdu = -(1-v)*p0x + (1-v)*p1x + v*p2x - v*p3x;
        double dydu = -(1-v)*p0y + (1-v)*p1y + v*p2y - v*p3y;
        double dzdu = -(1-v)*p0z + (1-v)*p1z + v*p2z - v*p3z;
        double dxdv = -(1-u)*p0x - u*p1x + u*p2x + (1-u)*p3x;
        double dydv = -(1-u)*p0y - u*p1y + u*p2y + (1-u)*p3y;
        double dzdv = -(1-u)*p0z - u*p1z + u*p2z + (1-u)*p3z;

        // Normal = du × dv
        double nx = dydu*dzdv - dzdu*dydv;
        double ny = dzdu*dxdv - dxdu*dzdv;
        double nz = dxdu*dydv - dydu*dxdv;

        double Bx, By, Bz;
        dip_B(x, y, z, Bx, By, Bz);
        return Bx*nx + By*ny + Bz*nz;
      }, 0.0, 1.0);
    }, 0.0, 1.0);

    m_B->data()[f] = static_cast<Scalar>(flux);
  }
}

// =========================================================================
// Legacy wrappers (for compatibility — delegate to _impl functions)
// =========================================================================

template <typename ExecPolicy>
Scalar dec_field_solver<ExecPolicy>::project_B_on_face(
    int f, Scalar Bx, Scalar By, Scalar Bz) const {
  return project_B_on_face_impl(m_mesh.host_ptrs(), f, Bx, By, Bz);
}

template <typename ExecPolicy>
Scalar dec_field_solver<ExecPolicy>::project_E_on_edge(
    int e, Scalar Ex, Scalar Ey, Scalar Ez) const {
  return project_E_on_edge_impl(m_mesh.host_ptrs(), e, Ex, Ey, Ez);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::dipole_B(
    Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my, Scalar mz,
    Scalar& Bx, Scalar& By, Scalar& Bz) {
  dipole_B_impl(x, y, z, mx, my, mz, Bx, By, Bz);
}

}  // namespace Aperture
