#pragma once

#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

template <typename ExecPolicy>
dec_field_solver<ExecPolicy>::dec_field_solver(prismatic_mesh& mesh)
    : m_mesh(mesh),
      m_E_e(mesh.m_N_edges, ExecPolicy::data_mem_type()),
      m_B_f(mesh.m_N_faces, ExecPolicy::data_mem_type()),
      m_J_e(mesh.m_N_edges, ExecPolicy::data_mem_type()) {}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::init() {
  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Omega", m_Omega);
  sim_env().params().get_value("obliquity", m_obliquity);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);

  m_E_e.assign(0, m_mesh.m_N_edges, 0.0);
  m_B_f.assign(0, m_mesh.m_N_faces, 0.0);
  m_J_e.assign(0, m_mesh.m_N_edges, 0.0);

  set_initial_dipole();

  m_time = 0.0;
  Logger::print_info("DEC field solver initialized: Bp={}, Omega={}, obliquity={}",
                     m_Bp, m_Omega, m_obliquity);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::update(double dt, uint32_t step) {
  // Get mesh pointers matching the execution target (host or device)
  auto mp = m_mesh.get_ptrs(typename ExecPolicy::exec_tag{});
  int N_edges = mp.N_edges;
  int N_faces = mp.N_faces;

  // Step 1: Faraday — B -= dt * d₁ * E
  ExecPolicy::launch(
      [N_faces, dt, mp] LAMBDA(auto E_e, auto B_f) {
        ExecPolicy::loop(0, N_faces, [&] LAMBDA(int f) {
          Scalar curl_E = 0.0;
          for (int j = mp.d1_row_ptr[f]; j < mp.d1_row_ptr[f + 1]; j++) {
            curl_E += mp.d1_val[j] * E_e[mp.d1_col_idx[j]];
          }
          B_f[f] -= dt * curl_E;
        });
      },
      m_E_e, m_B_f);

  // Step 2: Ampere — E += dt * ★₁⁻¹ * (d₁ᵀ * ★₂ * B - J)
  ExecPolicy::launch(
      [N_edges, dt, mp] LAMBDA(auto E_e, auto B_f, auto J_e) {
        ExecPolicy::loop(0, N_edges, [&] LAMBDA(int e) {
          Scalar curl_H = 0.0;
          for (int j = mp.d1t_row_ptr[e]; j < mp.d1t_row_ptr[e + 1]; j++) {
            int f = mp.d1t_col_idx[j];
            curl_H += mp.d1t_val[j] * mp.hodge2[f] * B_f[f];
          }
          E_e[e] += dt * mp.hodge1_inv[e] * (curl_H - J_e[e]);
        });
      },
      m_E_e, m_B_f, m_J_e);

  ExecPolicy::sync();

  // Sync fields to host for damping/BC (no-op for host-only buffers)
  m_E_e.copy_to_host();
  m_B_f.copy_to_host();

  // Clear J on host and sync to device
  m_J_e.assign(0, N_edges, 0.0);
  m_J_e.copy_to_device();

  // Step 3: Damping at outer boundary (host-side)
  apply_damping(dt);

  // Step 4: Inner boundary condition (host-side)
  m_time += dt;
  apply_inner_bc(m_time);

  // Sync modified fields back to device (no-op for host-only buffers)
  m_E_e.copy_to_device();
  m_B_f.copy_to_device();
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_damping(double dt) {
  if (m_damping_length <= 0) return;

  int N_r = m_mesh.m_N_r;
  int k_start = N_r - m_damping_length;
  if (k_start < 1) k_start = 1;

  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    int k = m_mesh.edge_radial_layer[e];
    if (k >= k_start) {
      Scalar sigma = m_damping_coef *
                     static_cast<Scalar>(k - k_start + 1) / m_damping_length;
      m_E_e[e] *= std::exp(-sigma * dt);
    }
  }

  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    int k = m_mesh.face_radial_layer[f];
    if (k >= k_start) {
      Scalar sigma = m_damping_coef *
                     static_cast<Scalar>(k - k_start + 1) / m_damping_length;
      m_B_f[f] *= std::exp(-sigma * dt);
    }
  }
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::apply_inner_bc(double time) {
  Scalar mx = m_Bp * std::sin(m_obliquity) * std::cos(m_Omega * time);
  Scalar my = m_Bp * std::sin(m_obliquity) * std::sin(m_Omega * time);
  Scalar mz = m_Bp * std::cos(m_obliquity);

  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);

  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    if (m_mesh.face_boundary[f] != 1) continue;
    Scalar fx, fy, fz;
    if (f < n_tri_faces) {
      int v0 = m_mesh.tri_face_v0[f], v1 = m_mesh.tri_face_v1[f],
          v2 = m_mesh.tri_face_v2[f];
      fx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] + m_mesh.vert_x[v2]) / 3.0;
      fy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] + m_mesh.vert_y[v2]) / 3.0;
      fz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] + m_mesh.vert_z[v2]) / 3.0;
    } else {
      int local = f - n_tri_faces;
      int v0 = m_mesh.rect_face_v0[local], v1 = m_mesh.rect_face_v1[local],
          v2 = m_mesh.rect_face_v2[local], v3 = m_mesh.rect_face_v3[local];
      fx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] + m_mesh.vert_x[v2] +
            m_mesh.vert_x[v3]) / 4.0;
      fy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] + m_mesh.vert_y[v2] +
            m_mesh.vert_y[v3]) / 4.0;
      fz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] + m_mesh.vert_z[v2] +
            m_mesh.vert_z[v3]) / 4.0;
    }
    Scalar Bx, By, Bz;
    dipole_B(fx, fy, fz, mx, my, mz, Bx, By, Bz);
    m_B_f[f] = project_B_on_face(f, Bx, By, Bz);
  }

  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    if (m_mesh.edge_boundary[e] != 1) continue;
    int v0 = m_mesh.edge_v0[e], v1 = m_mesh.edge_v1[e];
    Scalar ex = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1]) / 2.0;
    Scalar ey = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1]) / 2.0;
    Scalar ez = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1]) / 2.0;

    Scalar Bx, By, Bz;
    dipole_B(ex, ey, ez, mx, my, mz, Bx, By, Bz);
    Scalar vx = -m_Omega * ey;
    Scalar vy = m_Omega * ex;
    Scalar Ex = -(vy * Bz);
    Scalar Ey = -(- vx * Bz);
    Scalar Ez = -(vx * By - vy * Bx);
    m_E_e[e] = project_E_on_edge(e, Ex, Ey, Ez);
  }
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::set_initial_dipole() {
  Scalar mx = m_Bp * std::sin(m_obliquity);
  Scalar my = 0.0;
  Scalar mz = m_Bp * std::cos(m_obliquity);

  int N_verts = m_mesh.m_N_verts;
  std::vector<Scalar> phi(N_verts);
  for (int v = 0; v < N_verts; v++) {
    Scalar x = m_mesh.vert_x[v], y = m_mesh.vert_y[v], z = m_mesh.vert_z[v];
    Scalar r2 = x*x + y*y + z*z;
    Scalar r = std::sqrt(r2);
    phi[v] = (mx*x + my*y + mz*z) / (r2*r);
  }

  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);
  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    if (f < n_tri_faces) {
      int v0 = m_mesh.tri_face_v0[f], v1 = m_mesh.tri_face_v1[f],
          v2 = m_mesh.tri_face_v2[f];
      Scalar phi_avg = (phi[v0] + phi[v1] + phi[v2]) / 3.0;
      Scalar cx = (m_mesh.vert_x[v0]+m_mesh.vert_x[v1]+m_mesh.vert_x[v2]) / 3.0;
      Scalar cy = (m_mesh.vert_y[v0]+m_mesh.vert_y[v1]+m_mesh.vert_y[v2]) / 3.0;
      Scalar cz = (m_mesh.vert_z[v0]+m_mesh.vert_z[v1]+m_mesh.vert_z[v2]) / 3.0;
      Scalar r = std::sqrt(cx*cx + cy*cy + cz*cz);
      Scalar ax = m_mesh.vert_x[v1]-m_mesh.vert_x[v0];
      Scalar ay = m_mesh.vert_y[v1]-m_mesh.vert_y[v0];
      Scalar az = m_mesh.vert_z[v1]-m_mesh.vert_z[v0];
      Scalar bx = m_mesh.vert_x[v2]-m_mesh.vert_x[v0];
      Scalar by = m_mesh.vert_y[v2]-m_mesh.vert_y[v0];
      Scalar bz = m_mesh.vert_z[v2]-m_mesh.vert_z[v0];
      Scalar nx = 0.5*(ay*bz - az*by);
      Scalar ny = 0.5*(az*bx - ax*bz);
      Scalar nz = 0.5*(ax*by - ay*bx);
      Scalar n_dot_rhat = (nx*cx + ny*cy + nz*cz) / r;
      m_B_f[f] = (2.0 * phi_avg / r) * n_dot_rhat;
    } else {
      int local = f - n_tri_faces;
      int v0 = m_mesh.rect_face_v0[local], v1 = m_mesh.rect_face_v1[local],
          v2 = m_mesh.rect_face_v2[local], v3 = m_mesh.rect_face_v3[local];
      Scalar fx = (m_mesh.vert_x[v0]+m_mesh.vert_x[v1]+m_mesh.vert_x[v2]+m_mesh.vert_x[v3]) / 4.0;
      Scalar fy = (m_mesh.vert_y[v0]+m_mesh.vert_y[v1]+m_mesh.vert_y[v2]+m_mesh.vert_y[v3]) / 4.0;
      Scalar fz = (m_mesh.vert_z[v0]+m_mesh.vert_z[v1]+m_mesh.vert_z[v2]+m_mesh.vert_z[v3]) / 4.0;
      Scalar Bx, By, Bz;
      dipole_B(fx, fy, fz, mx, my, mz, Bx, By, Bz);
      m_B_f[f] = project_B_on_face(f, Bx, By, Bz);
    }
  }
}

template <typename ExecPolicy>
Scalar dec_field_solver<ExecPolicy>::project_B_on_face(
    int f, Scalar Bx, Scalar By, Scalar Bz) const {
  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);
  if (f < n_tri_faces) {
    int va = m_mesh.tri_face_v0[f], vb = m_mesh.tri_face_v1[f],
        vc = m_mesh.tri_face_v2[f];
    Scalar ax = m_mesh.vert_x[vb]-m_mesh.vert_x[va], ay = m_mesh.vert_y[vb]-m_mesh.vert_y[va], az = m_mesh.vert_z[vb]-m_mesh.vert_z[va];
    Scalar bx = m_mesh.vert_x[vc]-m_mesh.vert_x[va], by = m_mesh.vert_y[vc]-m_mesh.vert_y[va], bz = m_mesh.vert_z[vc]-m_mesh.vert_z[va];
    return 0.5 * (Bx*(ay*bz-az*by) + By*(az*bx-ax*bz) + Bz*(ax*by-ay*bx));
  }
  int local = f - n_tri_faces;
  int va = m_mesh.rect_face_v0[local], vb = m_mesh.rect_face_v1[local],
      vd = m_mesh.rect_face_v3[local];
  Scalar ax = m_mesh.vert_x[vb]-m_mesh.vert_x[va], ay = m_mesh.vert_y[vb]-m_mesh.vert_y[va], az = m_mesh.vert_z[vb]-m_mesh.vert_z[va];
  Scalar bx = m_mesh.vert_x[vd]-m_mesh.vert_x[va], by = m_mesh.vert_y[vd]-m_mesh.vert_y[va], bz = m_mesh.vert_z[vd]-m_mesh.vert_z[va];
  return Bx*(ay*bz-az*by) + By*(az*bx-ax*bz) + Bz*(ax*by-ay*bx);
}

template <typename ExecPolicy>
Scalar dec_field_solver<ExecPolicy>::project_E_on_edge(
    int e, Scalar Ex, Scalar Ey, Scalar Ez) const {
  int v0 = m_mesh.edge_v0[e], v1 = m_mesh.edge_v1[e];
  return Ex*(m_mesh.vert_x[v1]-m_mesh.vert_x[v0]) +
         Ey*(m_mesh.vert_y[v1]-m_mesh.vert_y[v0]) +
         Ez*(m_mesh.vert_z[v1]-m_mesh.vert_z[v0]);
}

template <typename ExecPolicy>
void dec_field_solver<ExecPolicy>::dipole_B(
    Scalar x, Scalar y, Scalar z, Scalar mx, Scalar my, Scalar mz,
    Scalar& Bx, Scalar& By, Scalar& Bz) {
  Scalar r2 = x*x + y*y + z*z;
  Scalar r = std::sqrt(r2);
  Scalar r5 = r2*r2*r;
  Scalar mdotr = mx*x + my*y + mz*z;
  Scalar factor = 3.0 * mdotr / r5;
  Scalar r3 = r2*r;
  Bx = factor*x - mx/r3;
  By = factor*y - my/r3;
  Bz = factor*z - mz/r3;
}

}  // namespace Aperture
