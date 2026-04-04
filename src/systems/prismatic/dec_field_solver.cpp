#include "systems/prismatic/dec_field_solver.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

dec_field_solver::dec_field_solver(prismatic_mesh& mesh)
    : m_mesh(mesh),
      m_D_e(mesh.m_N_edges, MemType::host_only),
      m_B_f(mesh.m_N_faces, MemType::host_only),
      m_E_tilde(mesh.m_N_edges, MemType::host_only),
      m_H_tilde(mesh.m_N_faces, MemType::host_only) {}

void dec_field_solver::init() {
  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Omega", m_Omega);
  sim_env().params().get_value("obliquity", m_obliquity);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);

  // Zero all fields
  m_D_e.assign(0, m_mesh.m_N_edges, 0.0);
  m_B_f.assign(0, m_mesh.m_N_faces, 0.0);
  m_E_tilde.assign(0, m_mesh.m_N_edges, 0.0);
  m_H_tilde.assign(0, m_mesh.m_N_faces, 0.0);

  // Set initial magnetic field (static dipole along z)
  set_initial_dipole();

  m_time = 0.0;
  Logger::print_info("DEC field solver initialized: Bp={}, Omega={}, obliquity={}",
                     m_Bp, m_Omega, m_obliquity);
}

void dec_field_solver::update(double dt, uint32_t step) {
  int N_edges = m_mesh.m_N_edges;
  int N_faces = m_mesh.m_N_faces;

  // Step 1: Constitutive relation E_tilde = hodge1_inv * D
  for (int e = 0; e < N_edges; e++) {
    m_E_tilde[e] = m_mesh.hodge1_inv[e] * m_D_e[e];
  }

  // Step 2: Faraday: B_f -= dt * d1 * E_tilde
  for (int f = 0; f < N_faces; f++) {
    Scalar curl_E = 0.0;
    for (int j = m_mesh.d1_row_ptr[f]; j < m_mesh.d1_row_ptr[f + 1]; j++) {
      curl_E += m_mesh.d1_val[j] * m_E_tilde[m_mesh.d1_col_idx[j]];
    }
    m_B_f[f] -= dt * curl_E;
  }

  // Step 3: Constitutive relation H_tilde = ★₂ * B (direct Hodge, NOT inverse)
  for (int f = 0; f < N_faces; f++) {
    m_H_tilde[f] = m_mesh.hodge2[f] * m_B_f[f];
  }

  // Step 4: Ampere: D_e += dt * d1^T * H_tilde  (no current for vacuum)
  for (int e = 0; e < N_edges; e++) {
    Scalar curl_H = 0.0;
    for (int j = m_mesh.d1t_row_ptr[e]; j < m_mesh.d1t_row_ptr[e + 1]; j++) {
      curl_H += m_mesh.d1t_val[j] * m_H_tilde[m_mesh.d1t_col_idx[j]];
    }
    m_D_e[e] += dt * curl_H;
  }

  // Step 5: Damping at outer boundary
  apply_damping(dt);

  // Step 6: Inner boundary condition (rotating dipole)
  m_time += dt;
  apply_inner_bc(m_time);
}

void dec_field_solver::apply_damping(double dt) {
  int N_r = m_mesh.m_N_r;
  int k_start = N_r - m_damping_length;
  if (k_start < 1) k_start = 1;

  // Damp edges
  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    int k = m_mesh.edge_radial_layer[e];
    if (k >= k_start) {
      Scalar sigma = m_damping_coef *
                     static_cast<Scalar>(k - k_start + 1) / m_damping_length;
      m_D_e[e] *= std::exp(-sigma * dt);
    }
  }

  // Damp faces
  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    int k = m_mesh.face_radial_layer[f];
    if (k >= k_start) {
      Scalar sigma = m_damping_coef *
                     static_cast<Scalar>(k - k_start + 1) / m_damping_length;
      m_B_f[f] *= std::exp(-sigma * dt);
    }
  }
}

void dec_field_solver::apply_inner_bc(double time) {
  // Rotating dipole: magnetic moment rotates in the x-z plane
  // m(t) = m_0 * (sin(alpha)*cos(Omega*t), sin(alpha)*sin(Omega*t), cos(alpha))
  Scalar mx = m_Bp * std::sin(m_obliquity) * std::cos(m_Omega * time);
  Scalar my = m_Bp * std::sin(m_obliquity) * std::sin(m_Omega * time);
  Scalar mz = m_Bp * std::cos(m_obliquity);

  // Overwrite B_f on inner boundary faces (shell k=0)
  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    if (m_mesh.face_boundary[f] != 1) continue;

    // Face midpoint (average of face vertices)
    Scalar fx, fy, fz;
    int fi_local = f;  // for triangular faces on shell 0, f = tri_face_idx(0, t) = t
    if (f < m_mesh.m_N_tri) {
      // Triangular face on shell 0
      int v0 = m_mesh.tri_face_v0[f];
      int v1 = m_mesh.tri_face_v1[f];
      int v2 = m_mesh.tri_face_v2[f];
      fx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] + m_mesh.vert_x[v2]) / 3.0;
      fy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] + m_mesh.vert_y[v2]) / 3.0;
      fz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] + m_mesh.vert_z[v2]) / 3.0;
    } else {
      // Rectangular face on layer 0
      int local = f - (m_mesh.m_N_r + 1) * m_mesh.m_N_tri;
      int v0 = m_mesh.rect_face_v0[local];
      int v1 = m_mesh.rect_face_v1[local];
      int v2 = m_mesh.rect_face_v2[local];
      int v3 = m_mesh.rect_face_v3[local];
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

  // Overwrite D_e on inner boundary edges (shell k=0)
  // E = -(Omega x r) x B for a rotating conductor
  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    if (m_mesh.edge_boundary[e] != 1) continue;

    // Edge midpoint
    int v0 = m_mesh.edge_v0[e];
    int v1 = m_mesh.edge_v1[e];
    Scalar ex = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1]) / 2.0;
    Scalar ey = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1]) / 2.0;
    Scalar ez = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1]) / 2.0;

    // B at midpoint
    Scalar Bx, By, Bz;
    dipole_B(ex, ey, ez, mx, my, mz, Bx, By, Bz);

    // v = Omega x r (rotation about z-axis)
    Scalar vx = -m_Omega * ey;
    Scalar vy = m_Omega * ex;
    Scalar vz = 0.0;

    // E = -v x B
    Scalar Ex = -(vy * Bz - vz * By);
    Scalar Ey = -(vz * Bx - vx * Bz);
    Scalar Ez = -(vx * By - vy * Bx);

    // D_e = ★₁ × E_line_integral = E_line_integral / hodge1_inv
    Scalar E_line = project_E_on_edge(e, Ex, Ey, Ez);
    m_D_e[e] = (m_mesh.hodge1_inv[e] > 0)
                   ? E_line / m_mesh.hodge1_inv[e]
                   : 0.0;
  }
}

void dec_field_solver::set_initial_dipole() {
  // Static dipole along z at t=0
  // m = (sin(alpha), 0, cos(alpha)) * Bp
  Scalar mx = m_Bp * std::sin(m_obliquity);
  Scalar my = 0.0;
  Scalar mz = m_Bp * std::cos(m_obliquity);

  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    // Face midpoint
    Scalar fx, fy, fz;
    int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);
    if (f < n_tri_faces) {
      int local = f;  // tri_face index
      int v0 = m_mesh.tri_face_v0[local];
      int v1 = m_mesh.tri_face_v1[local];
      int v2 = m_mesh.tri_face_v2[local];
      fx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] + m_mesh.vert_x[v2]) / 3.0;
      fy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] + m_mesh.vert_y[v2]) / 3.0;
      fz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] + m_mesh.vert_z[v2]) / 3.0;
    } else {
      int local = f - n_tri_faces;
      int v0 = m_mesh.rect_face_v0[local];
      int v1 = m_mesh.rect_face_v1[local];
      int v2 = m_mesh.rect_face_v2[local];
      int v3 = m_mesh.rect_face_v3[local];
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
}

Scalar dec_field_solver::project_B_on_face(int f, Scalar Bx, Scalar By,
                                            Scalar Bz) const {
  // Project B onto the face normal and multiply by face area
  // B_f = integral of B . dS ≈ B . n * area
  // For a triangular face, n = (v1-v0) x (v2-v0) / |...|
  // For a rectangular face, n = (v1-v0) x (v3-v0) / |...|
  Scalar nx, ny, nz;
  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);

  if (f < n_tri_faces) {
    int local = f;
    int va = m_mesh.tri_face_v0[local];
    int vb = m_mesh.tri_face_v1[local];
    int vc = m_mesh.tri_face_v2[local];
    Scalar ax = m_mesh.vert_x[vb] - m_mesh.vert_x[va];
    Scalar ay = m_mesh.vert_y[vb] - m_mesh.vert_y[va];
    Scalar az = m_mesh.vert_z[vb] - m_mesh.vert_z[va];
    Scalar bx = m_mesh.vert_x[vc] - m_mesh.vert_x[va];
    Scalar by = m_mesh.vert_y[vc] - m_mesh.vert_y[va];
    Scalar bz = m_mesh.vert_z[vc] - m_mesh.vert_z[va];
    // Cross product gives area-weighted normal (with correct orientation)
    nx = ay * bz - az * by;
    ny = az * bx - ax * bz;
    nz = ax * by - ay * bx;
    // B_f = B . (area-weighted normal) = B . cross product / 2 * 2
    // Actually cross product magnitude = 2 * area for triangle
    // So B . cross/2 = B . n_hat * area, but we want B_f = integral B.dS
    // = B . n_hat * area. cross = 2*area*n_hat, so B.cross/2 = B.n*area
    return 0.5 * (Bx * nx + By * ny + Bz * nz);
  } else {
    int local = f - n_tri_faces;
    int va = m_mesh.rect_face_v0[local];
    int vb = m_mesh.rect_face_v1[local];
    int vd = m_mesh.rect_face_v3[local];
    Scalar ax = m_mesh.vert_x[vb] - m_mesh.vert_x[va];
    Scalar ay = m_mesh.vert_y[vb] - m_mesh.vert_y[va];
    Scalar az = m_mesh.vert_z[vb] - m_mesh.vert_z[va];
    Scalar bx = m_mesh.vert_x[vd] - m_mesh.vert_x[va];
    Scalar by = m_mesh.vert_y[vd] - m_mesh.vert_y[va];
    Scalar bz = m_mesh.vert_z[vd] - m_mesh.vert_z[va];
    nx = ay * bz - az * by;
    ny = az * bx - ax * bz;
    nz = ax * by - ay * bx;
    // For a (non-planar) quad, this is approximate but OK for MVP
    return Bx * nx + By * ny + Bz * nz;
  }
}

Scalar dec_field_solver::project_E_on_edge(int e, Scalar Ex, Scalar Ey,
                                            Scalar Ez) const {
  // D_e = integral of E . dl ≈ E . edge_vector
  int v0 = m_mesh.edge_v0[e];
  int v1 = m_mesh.edge_v1[e];
  Scalar dx = m_mesh.vert_x[v1] - m_mesh.vert_x[v0];
  Scalar dy = m_mesh.vert_y[v1] - m_mesh.vert_y[v0];
  Scalar dz = m_mesh.vert_z[v1] - m_mesh.vert_z[v0];
  return Ex * dx + Ey * dy + Ez * dz;
}

void dec_field_solver::dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx,
                                 Scalar my, Scalar mz, Scalar& Bx, Scalar& By,
                                 Scalar& Bz) const {
  // B = (3(m.r_hat)r_hat - m) / r^3  (in units where mu0/4pi = 1)
  Scalar r2 = x * x + y * y + z * z;
  Scalar r = std::sqrt(r2);
  Scalar r5 = r2 * r2 * r;
  Scalar mdotr = mx * x + my * y + mz * z;
  Scalar factor = 3.0 * mdotr / r5;
  Scalar r3 = r2 * r;
  Bx = factor * x - mx / r3;
  By = factor * y - my / r3;
  Bz = factor * z - mz / r3;
}

}  // namespace Aperture
