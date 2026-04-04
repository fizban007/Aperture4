#include "systems/prismatic/dec_field_solver.h"
#include "framework/environment.h"
#include "utils/logger.h"
#include <cmath>

namespace Aperture {

dec_field_solver::dec_field_solver(prismatic_mesh& mesh)
    : m_mesh(mesh),
      m_E_e(mesh.m_N_edges, MemType::host_only),
      m_B_f(mesh.m_N_faces, MemType::host_only),
      m_rhs(mesh.m_N_edges, MemType::host_only),
      m_M2B(mesh.m_N_faces, MemType::host_only),
      m_dE_prev(mesh.m_N_edges, MemType::host_only),
      m_cg_r(mesh.m_N_edges, MemType::host_only),
      m_cg_z(mesh.m_N_edges, MemType::host_only),
      m_cg_p(mesh.m_N_edges, MemType::host_only),
      m_cg_Ap(mesh.m_N_edges, MemType::host_only) {}

void dec_field_solver::init() {
  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Omega", m_Omega);
  sim_env().params().get_value("obliquity", m_obliquity);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("damping_coef", m_damping_coef);
  int64_t cg_iter_tmp = m_cg_max_iter;
  sim_env().params().get_value("cg_max_iter", cg_iter_tmp);
  m_cg_max_iter = cg_iter_tmp;
  double cg_tol_tmp = m_cg_tol;
  sim_env().params().get_value("cg_tol", cg_tol_tmp);
  m_cg_tol = cg_tol_tmp;

  m_E_e.assign(0, m_mesh.m_N_edges, 0.0);
  m_B_f.assign(0, m_mesh.m_N_faces, 0.0);
  m_dE_prev.assign(0, m_mesh.m_N_edges, 0.0);

  set_initial_dipole();

  m_time = 0.0;
  Logger::print_info(
      "DEC field solver initialized: Bp={}, Omega={}, obliquity={}, "
      "CG max_iter={}, tol={}",
      m_Bp, m_Omega, m_obliquity, m_cg_max_iter, m_cg_tol);
}

void dec_field_solver::update(double dt, uint32_t step) {
  int N_edges = m_mesh.m_N_edges;
  int N_faces = m_mesh.m_N_faces;

  // Geometric dual Hodge formulation:
  //   Faraday: ∂B/∂t = -d₁ E              (exact, no Hodge)
  //   Ampere:  ∂E/∂t = ★₁⁻¹ d₁ᵀ ★₂ B     (diagonal Hodge, fully explicit)

  // Step 1: Faraday — B -= dt * d₁ * E
  for (int f = 0; f < N_faces; f++) {
    Scalar curl_E = 0.0;
    for (int j = m_mesh.d1_row_ptr[f]; j < m_mesh.d1_row_ptr[f + 1]; j++) {
      curl_E += m_mesh.d1_val[j] * m_E_e[m_mesh.d1_col_idx[j]];
    }
    m_B_f[f] -= dt * curl_E;
  }

  // Step 2: Ampere — E += dt * ★₁⁻¹ * d₁ᵀ * (★₂ * B)
  for (int e = 0; e < N_edges; e++) {
    Scalar curl_H = 0.0;
    for (int j = m_mesh.d1t_row_ptr[e]; j < m_mesh.d1t_row_ptr[e + 1]; j++) {
      int f = m_mesh.d1t_col_idx[j];
      curl_H += m_mesh.d1t_val[j] * m_mesh.hodge2[f] * m_B_f[f];
    }
    m_E_e[e] += dt * m_mesh.hodge1_inv[e] * curl_H;
  }

  // Step 3: Damping at outer boundary
  apply_damping(dt);

  // Step 4: Inner boundary condition
  m_time += dt;
  apply_inner_bc(m_time);
}

int dec_field_solver::cg_solve(const buffer<Scalar>& rhs, buffer<Scalar>& x,
                                int max_iter, Scalar tol) {
  int N = m_mesh.m_N_edges;

  // r = rhs - M₁ × x (warm-start: x may be nonzero from previous timestep)
  m_mesh.spmv_M1(x, m_cg_Ap);  // Ap = M₁ × x (reuse Ap as scratch)
  for (int i = 0; i < N; i++) {
    m_cg_r[i] = rhs[i] - m_cg_Ap[i];
  }

  // Preconditioned CG with block-Jacobi M₁⁻¹_approx as preconditioner
  // z = M₁⁻¹_approx × r
  m_mesh.spmv_M1inv(m_cg_r, m_cg_z);
  Scalar rz = 0;
  for (int i = 0; i < N; i++) {
    rz += m_cg_r[i] * m_cg_z[i];
    m_cg_p[i] = m_cg_z[i];
  }

  Scalar rhs_norm2 = 0;
  for (int i = 0; i < N; i++) {
    rhs_norm2 += rhs[i] * rhs[i];
  }
  if (rhs_norm2 == 0) return 0;

  Scalar tol2 = tol * tol * rhs_norm2;

  for (int iter = 0; iter < max_iter; iter++) {
    // Ap = M₁ × p
    m_mesh.spmv_M1(m_cg_p, m_cg_Ap);

    // alpha = rz / (p · Ap)
    Scalar pAp = 0;
    for (int i = 0; i < N; i++) {
      pAp += m_cg_p[i] * m_cg_Ap[i];
    }
    if (pAp == 0) return iter;
    Scalar alpha = rz / pAp;

    // x += alpha * p
    // r -= alpha * Ap
    Scalar r_norm2 = 0;
    for (int i = 0; i < N; i++) {
      x[i] += alpha * m_cg_p[i];
      m_cg_r[i] -= alpha * m_cg_Ap[i];
      r_norm2 += m_cg_r[i] * m_cg_r[i];
    }

    if (r_norm2 < tol2) return iter + 1;

    // z = M₁⁻¹_approx × r
    m_mesh.spmv_M1inv(m_cg_r, m_cg_z);
    Scalar rz_new = 0;
    for (int i = 0; i < N; i++) {
      rz_new += m_cg_r[i] * m_cg_z[i];
    }

    Scalar beta = rz_new / rz;
    rz = rz_new;

    // p = z + beta * p
    for (int i = 0; i < N; i++) {
      m_cg_p[i] = m_cg_z[i] + beta * m_cg_p[i];
    }
  }

  return max_iter;
}

void dec_field_solver::apply_damping(double dt) {
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

void dec_field_solver::apply_inner_bc(double time) {
  Scalar mx = m_Bp * std::sin(m_obliquity) * std::cos(m_Omega * time);
  Scalar my = m_Bp * std::sin(m_obliquity) * std::sin(m_Omega * time);
  Scalar mz = m_Bp * std::cos(m_obliquity);

  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);

  // Overwrite B_f on inner boundary faces
  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    if (m_mesh.face_boundary[f] != 1) continue;

    Scalar fx, fy, fz;
    if (f < n_tri_faces) {
      int local = f;
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

  // Overwrite E_e on inner boundary edges
  // E = -(Omega × r) × B for a rotating conductor
  for (int e = 0; e < m_mesh.m_N_edges; e++) {
    if (m_mesh.edge_boundary[e] != 1) continue;

    int v0 = m_mesh.edge_v0[e];
    int v1 = m_mesh.edge_v1[e];
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

    // E_e is the line integral ∫ E · dl (no Hodge conversion needed now)
    m_E_e[e] = project_E_on_edge(e, Ex, Ey, Ez);
  }
}

void dec_field_solver::set_initial_dipole() {
  // Initialize B using the scalar magnetic potential Φ = m·r / r³.
  // This is more accurate than evaluating B at face centroids because
  // Φ ~ 1/r² is smoother than B ~ 1/r³, giving better vertex sampling.
  //
  // For radial faces (triangular, on shells): B_r = -∂Φ/∂r = 2Φ/r
  //   → B_f = (2/r) × (average Φ at vertices) × face_area
  //
  // For tangential faces (rectangular, between shells): use the line
  //   integral of B along the face boundary via the scalar potential:
  //   B_f = -∫_f ∇Φ·dS, computed using vertex Φ values and the
  //   area-weighted normal.

  Scalar mx = m_Bp * std::sin(m_obliquity);
  Scalar my = 0.0;
  Scalar mz = m_Bp * std::cos(m_obliquity);

  // Compute Φ at every vertex: Φ = m·r / r³
  int N_verts = m_mesh.m_N_verts;
  std::vector<Scalar> phi(N_verts);
  for (int v = 0; v < N_verts; v++) {
    Scalar x = m_mesh.vert_x[v];
    Scalar y = m_mesh.vert_y[v];
    Scalar z = m_mesh.vert_z[v];
    Scalar r2 = x * x + y * y + z * z;
    Scalar r = std::sqrt(r2);
    Scalar r3 = r2 * r;
    phi[v] = (mx * x + my * y + mz * z) / r3;
  }

  int n_tri_faces = m_mesh.m_N_tri * (m_mesh.m_N_r + 1);

  for (int f = 0; f < m_mesh.m_N_faces; f++) {
    if (f < n_tri_faces) {
      // Triangular face on shell: normal is radial.
      // B_r = 2Φ/r, so B_f = B_r × face_area = (2Φ_avg/r) × face_area
      int v0 = m_mesh.tri_face_v0[f];
      int v1 = m_mesh.tri_face_v1[f];
      int v2 = m_mesh.tri_face_v2[f];
      Scalar phi_avg = (phi[v0] + phi[v1] + phi[v2]) / 3.0;

      // Shell radius (from centroid)
      Scalar cx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] + m_mesh.vert_x[v2]) / 3.0;
      Scalar cy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] + m_mesh.vert_y[v2]) / 3.0;
      Scalar cz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] + m_mesh.vert_z[v2]) / 3.0;
      Scalar r = std::sqrt(cx * cx + cy * cy + cz * cz);

      // face_area already has the correct sign from the face normal direction
      // (outward for the shell). B_r = 2Φ/r, flux = B_r × signed_area.
      // But face_area is unsigned. The sign comes from the face normal direction.
      // For triangular faces, the normal points outward (radially), and
      // B_r is positive when Φ > 0 (same hemisphere as dipole).
      // The project_B_on_face function handles the sign via the cross product.
      // Here we compute B_r and multiply by the signed area from the cross product.
      Scalar ax = m_mesh.vert_x[v1] - m_mesh.vert_x[v0];
      Scalar ay = m_mesh.vert_y[v1] - m_mesh.vert_y[v0];
      Scalar az = m_mesh.vert_z[v1] - m_mesh.vert_z[v0];
      Scalar bx = m_mesh.vert_x[v2] - m_mesh.vert_x[v0];
      Scalar by = m_mesh.vert_y[v2] - m_mesh.vert_y[v0];
      Scalar bz = m_mesh.vert_z[v2] - m_mesh.vert_z[v0];
      // Area-weighted normal = 0.5 × (edge1 × edge2)
      Scalar nx = 0.5 * (ay * bz - az * by);
      Scalar ny = 0.5 * (az * bx - ax * bz);
      Scalar nz = 0.5 * (ax * by - ay * bx);
      // n · r̂ gives signed area in radial direction
      Scalar n_dot_rhat = (nx * cx + ny * cy + nz * cz) / r;

      m_B_f[f] = (2.0 * phi_avg / r) * n_dot_rhat;
    } else {
      // Rectangular face (tangential): use ∇Φ from vertex values.
      // For a quad with vertices v0, v1, v2, v3, compute ∇Φ using the
      // bilinear gradient and dot with the area-weighted normal.
      // This is equivalent to the midpoint formula but uses vertex Φ values
      // for potentially better accuracy.
      int local = f - n_tri_faces;
      int v0 = m_mesh.rect_face_v0[local];
      int v1 = m_mesh.rect_face_v1[local];
      int v2 = m_mesh.rect_face_v2[local];
      int v3 = m_mesh.rect_face_v3[local];

      // Compute B at the face centroid using the dipole formula
      // (for rectangular faces the scalar potential approach is less clean)
      Scalar fx = (m_mesh.vert_x[v0] + m_mesh.vert_x[v1] +
                   m_mesh.vert_x[v2] + m_mesh.vert_x[v3]) / 4.0;
      Scalar fy = (m_mesh.vert_y[v0] + m_mesh.vert_y[v1] +
                   m_mesh.vert_y[v2] + m_mesh.vert_y[v3]) / 4.0;
      Scalar fz = (m_mesh.vert_z[v0] + m_mesh.vert_z[v1] +
                   m_mesh.vert_z[v2] + m_mesh.vert_z[v3]) / 4.0;
      Scalar Bx, By, Bz;
      dipole_B(fx, fy, fz, mx, my, mz, Bx, By, Bz);
      m_B_f[f] = project_B_on_face(f, Bx, By, Bz);
    }
  }
}

Scalar dec_field_solver::project_B_on_face(int f, Scalar Bx, Scalar By,
                                            Scalar Bz) const {
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
    Scalar nx = ay * bz - az * by;
    Scalar ny = az * bx - ax * bz;
    Scalar nz = ax * by - ay * bx;
    return 0.5 * (Bx * nx + By * ny + Bz * nz);
  }

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
  Scalar nx = ay * bz - az * by;
  Scalar ny = az * bx - ax * bz;
  Scalar nz = ax * by - ay * bx;
  return Bx * nx + By * ny + Bz * nz;
}

Scalar dec_field_solver::project_E_on_edge(int e, Scalar Ex, Scalar Ey,
                                            Scalar Ez) const {
  int v0 = m_mesh.edge_v0[e];
  int v1 = m_mesh.edge_v1[e];
  Scalar dx = m_mesh.vert_x[v1] - m_mesh.vert_x[v0];
  Scalar dy = m_mesh.vert_y[v1] - m_mesh.vert_y[v0];
  Scalar dz = m_mesh.vert_z[v1] - m_mesh.vert_z[v0];
  return Ex * dx + Ey * dy + Ez * dz;
}

void dec_field_solver::dipole_B(Scalar x, Scalar y, Scalar z, Scalar mx,
                                 Scalar my, Scalar mz, Scalar& Bx, Scalar& By,
                                 Scalar& Bz) {
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
