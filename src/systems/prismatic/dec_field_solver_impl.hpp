#pragma once

#include "systems/prismatic/cavity_modes.hpp"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "framework/environment.h"
#include "utils/gauss_quadrature.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

// =========================================================================
// Spherical-geometry helpers for face/edge quadratures.
//
// Mesh vertices are stored in (r, θ, φ) and faces live on the sphere
// surface, not on chord triangles.  All position-evaluations in the
// quadratures below go through these helpers so that the Gauss samples
// actually fall on the curved primal element.
// =========================================================================

// Fetch (r, unit-direction) for global vertex index vi.  The angular part
// is read from the Cartesian sphere_v{x,y,z} buffer (which we retain for
// particle operations) to avoid recomputing cos/sin per Gauss point.
HD_INLINE void vertex_unit(const prismatic_mesh_ptrs& mp, int vi,
                           Scalar& r, Scalar& ux, Scalar& uy, Scalar& uz) {
  int k = vi / mp.N_vert_s;
  int s = vi % mp.N_vert_s;
  r = mp.radii[k];
  ux = mp.sphere_vx[s];
  uy = mp.sphere_vy[s];
  uz = mp.sphere_vz[s];
}

// Slerp two unit vectors, plus its u-derivative.  At u=0 returns û_a, at
// u=1 returns û_b.  For very small α falls back to the linear tangent —
// the quadrature inner integrand handles the α→0 limit gracefully.
HD_INLINE void slerp_uv(Scalar ax, Scalar ay, Scalar az,
                        Scalar bx, Scalar by, Scalar bz, Scalar u,
                        Scalar& ux, Scalar& uy, Scalar& uz,
                        Scalar& dux, Scalar& duy, Scalar& duz) {
  Scalar dot = ax*bx + ay*by + az*bz;
  if (dot >  Scalar(1)) dot =  Scalar(1);
  if (dot < -Scalar(1)) dot = -Scalar(1);
  Scalar alpha = std::acos(dot);
  Scalar sa = std::sin(alpha);
  if (sa < Scalar(1e-12)) {
    ux = ax; uy = ay; uz = az;
    dux = bx - ax; duy = by - ay; duz = bz - az;
    return;
  }
  Scalar w0 = std::sin((Scalar(1) - u) * alpha) / sa;
  Scalar w1 = std::sin(u * alpha) / sa;
  ux = w0*ax + w1*bx;
  uy = w0*ay + w1*by;
  uz = w0*az + w1*bz;
  Scalar dw0 = -alpha * std::cos((Scalar(1) - u) * alpha) / sa;
  Scalar dw1 =  alpha * std::cos(u * alpha) / sa;
  dux = dw0*ax + dw1*bx;
  duy = dw0*ay + dw1*by;
  duz = dw0*az + dw1*bz;
}

// Spherical-triangle parametrization using radially-projected barycentric
// interpolation.  Domain: (u, t) ∈ [0, 1]²  with v = (1-u)·t  so that
//     λ_a = (1-u)(1-t),  λ_b = u,  λ_c = (1-u)·t
// Writes (x, y, z) on the sphere of radius r and (nx, ny, nz) = ∂P/∂u × ∂P/∂t,
// i.e. the vector surface element per du·dt (so the integrand in this
// parametrization is B · (nx,ny,nz) — the (1-u) Jacobian is already folded
// in through ∂λ/∂t factors).
HD_INLINE void tri_sphere_sample(Scalar r,
                                 Scalar ax, Scalar ay, Scalar az,
                                 Scalar bx, Scalar by, Scalar bz,
                                 Scalar cx, Scalar cy, Scalar cz,
                                 Scalar u, Scalar t,
                                 Scalar& x, Scalar& y, Scalar& z,
                                 Scalar& nx, Scalar& ny, Scalar& nz) {
  Scalar la = (Scalar(1) - u) * (Scalar(1) - t);
  Scalar lb = u;
  Scalar lc = (Scalar(1) - u) * t;
  Scalar qx = la*ax + lb*bx + lc*cx;
  Scalar qy = la*ay + lb*by + lc*cy;
  Scalar qz = la*az + lb*bz + lc*cz;
  Scalar qn = std::sqrt(qx*qx + qy*qy + qz*qz);
  Scalar ihx = qx / qn, ihy = qy / qn, ihz = qz / qn;
  x = r * ihx;  y = r * ihy;  z = r * ihz;

  // ∂λ/∂u = (-(1-t), 1, -t),  ∂λ/∂t = (-(1-u), 0, (1-u)).
  Scalar dqdu_x = -(Scalar(1) - t) * ax + bx - t * cx;
  Scalar dqdu_y = -(Scalar(1) - t) * ay + by - t * cy;
  Scalar dqdu_z = -(Scalar(1) - t) * az + bz - t * cz;
  Scalar dqdt_x = (Scalar(1) - u) * (cx - ax);
  Scalar dqdt_y = (Scalar(1) - u) * (cy - ay);
  Scalar dqdt_z = (Scalar(1) - u) * (cz - az);

  // ∂û/∂ξ = (I − û⊗û) · ∂Q/∂ξ / |Q|
  Scalar qinv = Scalar(1) / qn;
  Scalar pdu = ihx*dqdu_x + ihy*dqdu_y + ihz*dqdu_z;  // û·∂Q/∂u
  Scalar pdt = ihx*dqdt_x + ihy*dqdt_y + ihz*dqdt_z;
  Scalar duhx = qinv * (dqdu_x - pdu*ihx);
  Scalar duhy = qinv * (dqdu_y - pdu*ihy);
  Scalar duhz = qinv * (dqdu_z - pdu*ihz);
  Scalar dthx = qinv * (dqdt_x - pdt*ihx);
  Scalar dthy = qinv * (dqdt_y - pdt*ihy);
  Scalar dthz = qinv * (dqdt_z - pdt*ihz);

  // n = r² · (∂û/∂u × ∂û/∂t)
  Scalar r2 = r * r;
  nx = r2 * (duhy*dthz - duhz*dthy);
  ny = r2 * (duhz*dthx - duhx*dthz);
  nz = r2 * (duhx*dthy - duhy*dthx);
}

// Rectangular face (ruled surface between shells r0 and r1 along the
// great-circle arc û_a→û_b).  Parametrization:
//     r(v) = (1-v)·r0 + v·r1,   û(u) = slerp(û_a, û_b, u),
//     P    = r(v) · û(u).
// Writes (x, y, z) and (nx, ny, nz) = ∂P/∂u × ∂P/∂v.
HD_INLINE void rect_sphere_sample(Scalar r0, Scalar r1,
                                  Scalar ax, Scalar ay, Scalar az,
                                  Scalar bx, Scalar by, Scalar bz,
                                  Scalar u, Scalar v,
                                  Scalar& x, Scalar& y, Scalar& z,
                                  Scalar& nx, Scalar& ny, Scalar& nz) {
  Scalar ux, uy, uz, dux, duy, duz;
  slerp_uv(ax, ay, az, bx, by, bz, u, ux, uy, uz, dux, duy, duz);
  Scalar rv = (Scalar(1) - v) * r0 + v * r1;
  Scalar drdv = r1 - r0;
  x = rv * ux;  y = rv * uy;  z = rv * uz;
  // ∂P/∂u = rv · dû/du,  ∂P/∂v = drdv · û
  Scalar pux = rv * dux, puy = rv * duy, puz = rv * duz;
  Scalar pvx = drdv * ux, pvy = drdv * uy, pvz = drdv * uz;
  nx = puy*pvz - puz*pvy;
  ny = puz*pvx - pux*pvz;
  nz = pux*pvy - puy*pvx;
}

// Horizontal (arc) edge sample at parameter t ∈ [0,1] on the sphere of
// radius r.  Writes position (x, y, z) and line element dl = ∂P/∂t.
HD_INLINE void h_edge_sphere_sample(Scalar r,
                                    Scalar ax, Scalar ay, Scalar az,
                                    Scalar bx, Scalar by, Scalar bz,
                                    Scalar t,
                                    Scalar& x, Scalar& y, Scalar& z,
                                    Scalar& dlx, Scalar& dly, Scalar& dlz) {
  Scalar ux, uy, uz, dux, duy, duz;
  slerp_uv(ax, ay, az, bx, by, bz, t, ux, uy, uz, dux, duy, duz);
  x = r * ux;  y = r * uy;  z = r * uz;
  dlx = r * dux;  dly = r * duy;  dlz = r * duz;
}

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

// Full retarded Deutsch solution for a rotating magnetic dipole (c = 1).
// Derived from the Hertz potential Π = m(t_r)/r, with A = ∇×Π:
//
//   A = (m_r × n̂)/r² + (ṁ_r × n̂)/r
//   B = ∇×A = [3n̂(n̂·m_r) - m_r]/r³ + [3n̂(n̂·ṁ_r) - ṁ_r]/r² + [n̂(n̂·m̈_r) - m̈_r]/r
//   E = -∂A/∂t = (n̂ × ṁ_r)/r² + (n̂ × m̈_r)/r
//
// where n̂ = r̂, m_r = m(t - r), ṁ_r = ṁ(t - r), m̈_r = m̈(t - r).
HD_INLINE void deutsch_B_impl(Scalar x, Scalar y, Scalar z, Scalar time,
                               Scalar Bp, Scalar Omega, Scalar obliquity,
                               Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r3 = r2*r;
  Scalar t_ret = time - r;

  Scalar m_perp = Bp * std::sin(obliquity);
  Scalar m_par = Bp * std::cos(obliquity);

  Scalar cos_phase = std::cos(Omega * t_ret);
  Scalar sin_phase = std::sin(Omega * t_ret);

  // Retarded dipole moment, its first and second time-derivatives
  Scalar mx = m_perp * cos_phase;
  Scalar my = m_perp * sin_phase;
  Scalar mz = m_par;

  Scalar dmx = -m_perp * Omega * sin_phase;
  Scalar dmy =  m_perp * Omega * cos_phase;

  Scalar ddmx = -m_perp * Omega * Omega * cos_phase;
  Scalar ddmy = -m_perp * Omega * Omega * sin_phase;

  Scalar nx = x / r, ny = y / r, nz = z / r;

  // Near field: [3n(n·m) - m] / r³
  Scalar ndotm = nx*mx + ny*my + nz*mz;
  Scalar Bnx = (Scalar(3.0)*ndotm*nx - mx) / r3;
  Scalar Bny = (Scalar(3.0)*ndotm*ny - my) / r3;
  Scalar Bnz = (Scalar(3.0)*ndotm*nz - mz) / r3;

  // Intermediate field: [3n(n·dm) - dm] / r²
  Scalar ndotdm = nx*dmx + ny*dmy;
  Scalar Bix = (Scalar(3.0)*ndotdm*nx - dmx) / r2;
  Scalar Biy = (Scalar(3.0)*ndotdm*ny - dmy) / r2;
  Scalar Biz = (Scalar(3.0)*ndotdm*nz) / r2;

  // Radiation field: [n(n·ddm) - ddm] / r
  Scalar ndotddm = nx*ddmx + ny*ddmy;
  Scalar Brx = (ndotddm*nx - ddmx) / r;
  Scalar Bry = (ndotddm*ny - ddmy) / r;
  Scalar Brz = (ndotddm*nz) / r;

  Bx = Bnx + Bix + Brx;
  By = Bny + Biy + Bry;
  Bz = Bnz + Biz + Brz;
}

HD_INLINE void deutsch_E_impl(Scalar x, Scalar y, Scalar z, Scalar time,
                               Scalar Bp, Scalar Omega, Scalar obliquity,
                               Scalar& Ex, Scalar& Ey, Scalar& Ez) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar t_ret = time - r;

  Scalar m_perp = Bp * std::sin(obliquity);

  Scalar cos_phase = std::cos(Omega * t_ret);
  Scalar sin_phase = std::sin(Omega * t_ret);

  Scalar dmx = -m_perp * Omega * sin_phase;
  Scalar dmy =  m_perp * Omega * cos_phase;

  Scalar ddmx = -m_perp * Omega * Omega * cos_phase;
  Scalar ddmy = -m_perp * Omega * Omega * sin_phase;

  Scalar nx = x / r, ny = y / r, nz = z / r;

  // E = +(n × dm)/r² + (n × ddm)/r
  // n × dm = (-nz*dmy, nz*dmx, nx*dmy - ny*dmx)
  Scalar cx1 = -nz * dmy;
  Scalar cy1 =  nz * dmx;
  Scalar cz1 =  nx * dmy - ny * dmx;

  Scalar cx2 = -nz * ddmy;
  Scalar cy2 =  nz * ddmx;
  Scalar cz2 =  nx * ddmy - ny * ddmx;

  Ex = cx1 / r2 + cx2 / r;
  Ey = cy1 / r2 + cy2 / r;
  Ez = cz1 / r2 + cz2 / r;
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
  sim_env().params().get_value("use_deutsch_bc", m_use_deutsch_bc);
  sim_env().params().get_value("use_pec_bc", m_use_pec_bc);
  sim_env().params().get_value("resonator_amp", m_resonator_amp);

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
  if (m_use_pec_bc) {
    apply_pec_bc(m_tmp_E, m_tmp_B);
  } else {
    apply_inner_bc(m_tmp_E, m_tmp_B, m_time + dt, m_time + dt);
  }
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

  // Step 5: Damping + BC + clear J
  apply_damping(m_E->data(), m_B->data(), dt);
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
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  Scalar Bp_val = m_Bp;
  Scalar Omega_val = m_Omega;
  Scalar obliq = m_obliquity;
  bool deutsch = m_use_deutsch_bc;
  int n_tri_faces = mp.N_tri * (mp.N_r + 1);

  // Instantaneous dipole moment (for standard BC mode), at each field's
  // own time level (B is half-step staggered under leapfrog).
  Scalar mx_B = Bp_val * std::sin(obliq) * std::cos(Omega_val * time_B);
  Scalar my_B = Bp_val * std::sin(obliq) * std::sin(Omega_val * time_B);
  Scalar mx_E = Bp_val * std::sin(obliq) * std::cos(Omega_val * time_E);
  Scalar my_E = Bp_val * std::sin(obliq) * std::sin(Omega_val * time_E);
  Scalar mz_i = Bp_val * std::cos(obliq);
  Scalar t_bc_B = static_cast<Scalar>(time_B);
  Scalar t_bc_E = static_cast<Scalar>(time_E);

  // --- Overwrite B_f on inner boundary faces (at time_B) ---
  ExecPolicy::launch(
      [N_faces = mp.N_faces, n_tri_faces, mx_i = mx_B, my_i = my_B, mz_i,
       Bp_val, Omega_val, obliq, deutsch, t_bc = t_bc_B, mp]
      LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          if (mp.face_boundary[f] != 1) return;

          if (f < n_tri_faces) {
            // --- Spherical triangular face: Gauss quadrature in (u, t) ∈ [0,1]².
            // Barycentric λ_a = (1-u)(1-t), λ_b = u, λ_c = (1-u)t;
            // Position on the sphere of radius r via radial projection of
            // λ_a·û_a + λ_b·û_b + λ_c·û_c; area element ∂P/∂u × ∂P/∂t absorbs
            // the (1-u) Jacobian.
            int vi0 = mp.tri_face_v0[f];
            int vi1 = mp.tri_face_v1[f];
            int vi2 = mp.tri_face_v2[f];
            Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z, r2, a2x, a2y, a2z;
            vertex_unit(mp, vi0, r0, a0x, a0y, a0z);
            vertex_unit(mp, vi1, r1, a1x, a1y, a1z);
            vertex_unit(mp, vi2, r2, a2x, a2y, a2z);
            Scalar r_face = r0;  // all three vertices share the shell radius

            Scalar flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double t) -> double {
                Scalar x, y, z, nx, ny, nz;
                tri_sphere_sample(r_face, a0x, a0y, a0z, a1x, a1y, a1z,
                                  a2x, a2y, a2z, u, t, x, y, z, nx, ny, nz);
                Scalar bx, by, bz;
                if (deutsch) {
                  deutsch_B_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq, bx, by, bz);
                } else {
                  dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bx, by, bz);
                }
                return bx*nx + by*ny + bz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          } else {
            // --- Ruled rectangular face: P(u,v) = r(v) · slerp(û_a, û_b, u).
            // Layout: v0=(r_lo,û_a), v1=(r_lo,û_b), v2=(r_hi,û_b), v3=(r_hi,û_a).
            int fi = f - n_tri_faces;
            int vi0 = mp.rect_face_v0[fi];
            int vi1 = mp.rect_face_v1[fi];
            int vi3 = mp.rect_face_v3[fi];
            Scalar r_lo, uax, uay, uaz;
            Scalar r_tmp, ubx, uby, ubz;
            Scalar r_hi, uax3, uay3, uaz3;
            vertex_unit(mp, vi0, r_lo, uax, uay, uaz);
            vertex_unit(mp, vi1, r_tmp, ubx, uby, ubz);
            vertex_unit(mp, vi3, r_hi, uax3, uay3, uaz3);
            (void)r_tmp; (void)uax3; (void)uay3; (void)uaz3;

            Scalar flux = gauss_quad([&](double u) -> double {
              return gauss_quad([&](double v) -> double {
                Scalar x, y, z, nx, ny, nz;
                rect_sphere_sample(r_lo, r_hi, uax, uay, uaz, ubx, uby, ubz,
                                   u, v, x, y, z, nx, ny, nz);
                Scalar bbx, bby, bbz;
                if (deutsch) {
                  deutsch_B_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq, bbx, bby, bbz);
                } else {
                  dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bbx, bby, bbz);
                }
                return bbx*nx + bby*ny + bbz*nz;
              }, 0.0, 1.0);
            }, 0.0, 1.0);
            B_f[f] = static_cast<Scalar>(flux);
          }
        });
      },
      B);

  // --- Overwrite E_e on inner boundary edges (at time_E) ---
  ExecPolicy::launch(
      [N_edges = mp.N_edges, mx_i = mx_E, my_i = my_E, mz_i,
       Bp_val, Omega_val, obliq, deutsch, t_bc = t_bc_E, mp]
      LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          if (mp.edge_boundary[e] != 1) return;
          int v0 = mp.edge_v0[e], v1 = mp.edge_v1[e];
          Scalar r0, a0x, a0y, a0z, r1, a1x, a1y, a1z;
          vertex_unit(mp, v0, r0, a0x, a0y, a0z);
          vertex_unit(mp, v1, r1, a1x, a1y, a1z);

          // Horizontal (arc) edge: same angular endpoints at different r is
          // not possible, so r0 == r1 and the edge is an arc on the sphere.
          // Vertical (radial) edge: same sphere vertex at different r.
          bool is_radial = (r0 != r1);

          Scalar circ = gauss_quad([&](double t) -> double {
            Scalar x, y, z, dlx, dly, dlz;
            if (is_radial) {
              // P(t) = ((1-t) r0 + t r1) · û;  dl = (r1-r0) · û dt.
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
            if (deutsch) {
              deutsch_E_impl(x, y, z, t_bc, Bp_val, Omega_val, obliq, ex, ey, ez);
            } else {
              // E = -(v × B) where v = Ω × r.
              Scalar bx, by, bz;
              dipole_B_impl(x, y, z, mx_i, my_i, mz_i, bx, by, bz);
              Scalar vx = -Omega_val * y, vy = Omega_val * x;
              ex = -(vy * bz);
              ey = -(-vx * bz);
              ez = -(vx * by - vy * bx);
            }
            return ex*dlx + ey*dly + ez*dlz;
          }, 0.0, 1.0);
          E_e[e] = static_cast<Scalar>(circ);
        });
      },
      E);
}

// =========================================================================
// Initial dipole (host-only, called once during init)
// =========================================================================

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_dipole() {
  Scalar mx_v = m_Bp * std::sin(m_obliquity);
  Scalar my_v = Scalar(0);
  Scalar mz_v = m_Bp * std::cos(m_obliquity);

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
      m_B->data());

  // ---- E starts at zero ----
  ExecPolicy::launch(
      [N_edges = mp.N_edges] LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          E_e[e] = Scalar(0);
        });
      },
      m_E->data());

  ExecPolicy::sync();
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
  Scalar t_init = Scalar(0);  // E initial time = 0

  // Leapfrog staggering: the first Faraday half-step advances B from
  // -dt/2 to +dt/2, so a consistent start evaluates the analytic B at
  // t = -dt/2 (E stays at 0).  The co-located semi-implicit scheme
  // initializes both at 0.
  double dt_param = 0.0;
  sim_env().params().get_value("dt", dt_param);
  Scalar t_init_B =
      m_use_implicit ? Scalar(0) : Scalar(-0.5 * dt_param);

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

  ExecPolicy::sync();
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
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_edge_s = mp.N_edge_s;
  int N_tri = mp.N_tri;
  int N_r = mp.N_r;

  // Zero tangential E on inner shell (k=0) and outer shell (k=N_r)
  ExecPolicy::launch(
      [N_edge_s, N_r] LAMBDA(auto E_e) {
        ExecPolicy::loop(0, N_edge_s, [&] LAMBDA(int e) {
          E_e[e] = Scalar(0);                          // shell k = 0
          E_e[N_r * N_edge_s + e] = Scalar(0);          // shell k = N_r
        });
      },
      E);

  // Zero normal B on inner shell (k=0) and outer shell (k=N_r)
  ExecPolicy::launch(
      [N_tri, N_r] LAMBDA(auto B_f) {
        ExecPolicy::loop(0, N_tri, [&] LAMBDA(int t) {
          B_f[t] = Scalar(0);                          // shell k = 0
          B_f[N_r * N_tri + t] = Scalar(0);            // shell k = N_r
        });
      },
      B);
}

}  // namespace Aperture
