#include "systems/prismatic/prismatic_vertex_recovery.h"

#include "utils/logger.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace Aperture {

namespace {

// ---------------------------------------------------------------------
// Small double-precision vector helpers (host only)
// ---------------------------------------------------------------------
struct v3 {
  double x, y, z;
};
inline v3 operator+(v3 a, v3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline v3 operator-(v3 a, v3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline v3 operator*(double s, v3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline v3 cross(v3 a, v3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(v3 a) { return std::sqrt(dot(a, a)); }

// Gauss-Legendre nodes/weights on [0,1], n = 4 (order 7 — the fit only
// needs the moments of linear fields exactly, so this is ample).
constexpr int NQ = 4;
constexpr double GL_X[NQ] = {0.06943184420297371, 0.33000947820757187,
                             0.6699905217924281, 0.9305681557970262};
constexpr double GL_W[NQ] = {0.1739274225687269, 0.3260725774312731,
                             0.3260725774312731, 0.1739274225687269};

// Vector area A = ∫ dA and first moment M[i][j] = ∫ (x - x_v)_j dA_i of
// one curved face, by tensor quadrature (Duffy for the triangle).
struct face_moments {
  double A[3];
  double M[3][3];
};

// Shell (spherical) triangle at radius r with unit-sphere corners u0,u1,u2.
face_moments tri_face_moments(v3 u0, v3 u1, v3 u2, double r, v3 x_v) {
  face_moments fm{};
  for (int a = 0; a < NQ; a++) {
    for (int b = 0; b < NQ; b++) {
      double l1 = GL_X[a];
      double l2 = GL_X[b] * (1.0 - l1);
      double w = GL_W[a] * GL_W[b] * (1.0 - l1);  // Duffy Jacobian
      double l0 = 1.0 - l1 - l2;
      v3 P = l0 * u0 + l1 * u1 + l2 * u2;
      double Pn = norm(P);
      v3 Nh = (1.0 / Pn) * P;
      v3 dP1 = u1 - u0, dP2 = u2 - u0;
      v3 dN1 = (1.0 / Pn) * (dP1 - dot(Nh, dP1) * Nh);
      v3 dN2 = (1.0 / Pn) * (dP2 - dot(Nh, dP2) * Nh);
      v3 dA = cross(r * dN1, r * dN2);
      v3 X = r * Nh - x_v;
      double dAv[3] = {dA.x, dA.y, dA.z};
      double Xv[3] = {X.x, X.y, X.z};
      for (int i = 0; i < 3; i++) {
        fm.A[i] += w * dAv[i];
        for (int j = 0; j < 3; j++) fm.M[i][j] += w * dAv[i] * Xv[j];
      }
    }
  }
  return fm;
}

// Ruled rectangular face over sphere edge (u0, u1), radii [ra, rb].
face_moments rect_face_moments(v3 u0, v3 u1, double ra, double rb, v3 x_v) {
  face_moments fm{};
  for (int a = 0; a < NQ; a++) {
    for (int b = 0; b < NQ; b++) {
      double u = GL_X[a], z = GL_X[b];
      double w = GL_W[a] * GL_W[b];
      v3 P = (1.0 - u) * u0 + u * u1;
      double Pn = norm(P);
      v3 Nh = (1.0 / Pn) * P;
      v3 dPu = u1 - u0;
      v3 dNu = (1.0 / Pn) * (dPu - dot(Nh, dPu) * Nh);
      double r = ra + (rb - ra) * z;
      v3 dXu = r * dNu;
      v3 dXz = (rb - ra) * Nh;
      v3 dA = cross(dXu, dXz);
      v3 X = r * Nh - x_v;
      double dAv[3] = {dA.x, dA.y, dA.z};
      double Xv[3] = {X.x, X.y, X.z};
      for (int i = 0; i < 3; i++) {
        fm.A[i] += w * dAv[i];
        for (int j = 0; j < 3; j++) fm.M[i][j] += w * dAv[i] * Xv[j];
      }
    }
  }
  return fm;
}

// Fixed 12x11 embedding of [B0(3), G(9, row-major)] with trace(G) = 0.
// Column order: B0_x, B0_y, B0_z, 6 off-diagonal G, 2 traceless diagonal.
struct traceless_embedding {
  double N[12][11] = {};
  traceless_embedding() {
    for (int i = 0; i < 3; i++) N[i][i] = 1.0;
    int col = 3;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (i != j) N[3 + 3 * i + j][col++] = 1.0;
    const double s2 = 1.0 / std::sqrt(2.0), s6 = 1.0 / std::sqrt(6.0);
    N[3 + 0][9] = s2;  N[3 + 4][9] = -s2;                    // diag(1,-1,0)
    N[3 + 0][10] = s6; N[3 + 4][10] = s6; N[3 + 8][10] = -2 * s6;
  }
};

// Cholesky solve of the 11x11 SPD system C X = E (E is 11 x nf).
void cholesky_solve_11(double C[11][11], std::vector<double>& E, int nf) {
  double L[11][11] = {};
  for (int i = 0; i < 11; i++) {
    for (int j = 0; j <= i; j++) {
      double s = C[i][j];
      for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i == j) {
        if (s <= 0.0)
          throw std::runtime_error("vertex recovery: patch fit not SPD");
        L[i][i] = std::sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  // forward + back substitution, column by column of E (11 x nf, row-major)
  for (int c = 0; c < nf; c++) {
    double y[11];
    for (int i = 0; i < 11; i++) {
      double s = E[i * nf + c];
      for (int k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    for (int i = 10; i >= 0; i--) {
      double s = y[i];
      for (int k = i + 1; k < 11; k++) s -= L[k][i] * E[k * nf + c];
      E[i * nf + c] = s / L[i][i];
    }
  }
}

// Condition number of the 11x11 SPD matrix via Jacobi eigenvalues;
// cond of the underlying LSQ matrix is its square root.
double spd_cond_11(double Cin[11][11]) {
  double A[11][11];
  std::memcpy(A, Cin, sizeof(A));
  for (int sweep = 0; sweep < 30; sweep++) {
    double off = 0;
    for (int p = 0; p < 11; p++)
      for (int q = p + 1; q < 11; q++) off += A[p][q] * A[p][q];
    if (off < 1e-24) break;
    for (int p = 0; p < 11; p++) {
      for (int q = p + 1; q < 11; q++) {
        if (std::abs(A[p][q]) < 1e-30) continue;
        double theta = 0.5 * std::atan2(2 * A[p][q], A[q][q] - A[p][p]);
        double c = std::cos(theta), s = std::sin(theta);
        for (int k = 0; k < 11; k++) {
          double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 11; k++) {
          double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  double lo = A[0][0], hi = A[0][0];
  for (int i = 1; i < 11; i++) {
    lo = std::min(lo, A[i][i]);
    hi = std::max(hi, A[i][i]);
  }
  return std::sqrt(hi / std::max(lo, 1e-300));
}

}  // namespace

void prismatic_vertex_recovery::build(const prismatic_mesh& mesh) {
  m_N_vert_s = mesh.m_N_vert_s;
  m_N_r = mesh.m_N_r;
  m_N_verts = mesh.m_N_verts;
  m_r_ref = mesh.radii[1];
  if (m_N_r < 3)
    throw std::runtime_error("vertex recovery: needs N_r >= 3");

  // The interior-weight radial rescaling assumes exactly geometric shells
  // (guaranteed by prismatic_mesh::build; assert in case that changes).
  for (int k = 0; k + 2 <= m_N_r; k++) {
    double q0 = double(mesh.radii[k + 1]) / mesh.radii[k];
    double q1 = double(mesh.radii[k + 2]) / mesh.radii[k + 1];
    if (std::abs(q1 / q0 - 1.0) > 1e-4)
      throw std::runtime_error(
          "vertex recovery: radial shells are not geometric");
  }

  // ---- fan tables --------------------------------------------------
  valence.resize(m_N_vert_s);
  tri_fan.resize(6 * m_N_vert_s);
  edge_fan.resize(6 * m_N_vert_s);
  std::vector<int> nt(m_N_vert_s, 0), ne(m_N_vert_s, 0);
  for (int i = 0; i < 6 * m_N_vert_s; i++) tri_fan[i] = edge_fan[i] = 0;
  for (int t = 0; t < mesh.m_N_tri; t++) {
    for (int a = 0; a < 3; a++) {
      int s = mesh.tri_verts[t * 3 + a];
      if (nt[s] >= 6)
        throw std::runtime_error("vertex recovery: valence > 6");
      tri_fan[s * 6 + nt[s]++] = t;
    }
  }
  // sphere edge endpoints (7D: from the persisted sphere tables — the
  // recovery must build on a sphere-only mesh).
  for (int e = 0; e < mesh.m_N_edge_s; e++) {
    int s0 = mesh.sphere_edge_v0[e], s1 = mesh.sphere_edge_v1[e];
    if (s0 >= m_N_vert_s || s1 >= m_N_vert_s)
      throw std::runtime_error("vertex recovery: bad layer-0 edge endpoint");
    edge_fan[s0 * 6 + ne[s0]++] = e;
    edge_fan[s1 * 6 + ne[s1]++] = e;
  }
  for (int s = 0; s < m_N_vert_s; s++) {
    if (nt[s] != ne[s] || (nt[s] != 5 && nt[s] != 6))
      throw std::runtime_error("vertex recovery: inconsistent fan valence");
    valence[s] = nt[s];
  }

  // ---- weights ------------------------------------------------------
  constexpr int SI = prismatic_recovery_ptrs::stride_int;
  constexpr int SB = prismatic_recovery_ptrs::stride_bnd;
  w_int.resize(m_N_vert_s * 3 * SI);
  w_inner.resize(m_N_vert_s * 3 * SB);
  w_outer.resize(m_N_vert_s * 3 * SB);
  Bv.resize(3 * m_N_verts);

  static const traceless_embedding NT;
  m_max_cond = 0.0;

  // Patch face list for one (sphere vertex, class); mirrors the runtime
  // canonical order in prismatic_recovery_ptrs::compute_vertex_B.
  struct pface {
    bool is_tri;
    double r0, r1;  // tri: r0 = shell radius; rect: [r0, r1]
    int idx;        // triangle or sphere-edge index
  };

  auto solve_patch = [&](int s, const std::vector<pface>& faces, double r_v,
                         Scalar* w_out, int stride) {
    v3 u_v{mesh.sphere_vx[s], mesh.sphere_vy[s], mesh.sphere_vz[s]};
    v3 x_v = r_v * u_v;
    int nf = (int)faces.size();

    std::vector<double> rows(nf * 12), rscale(nf);
    for (int f = 0; f < nf; f++) {
      face_moments fm;
      if (faces[f].is_tri) {
        int t = faces[f].idx;
        int a = mesh.tri_verts[t * 3 + 0], b = mesh.tri_verts[t * 3 + 1],
            c = mesh.tri_verts[t * 3 + 2];
        fm = tri_face_moments(
            {mesh.sphere_vx[a], mesh.sphere_vy[a], mesh.sphere_vz[a]},
            {mesh.sphere_vx[b], mesh.sphere_vy[b], mesh.sphere_vz[b]},
            {mesh.sphere_vx[c], mesh.sphere_vy[c], mesh.sphere_vz[c]},
            faces[f].r0, x_v);
      } else {
        int e = faces[f].idx;
        int s0 = mesh.sphere_edge_v0[e], s1 = mesh.sphere_edge_v1[e];
        fm = rect_face_moments(
            {mesh.sphere_vx[s0], mesh.sphere_vy[s0], mesh.sphere_vz[s0]},
            {mesh.sphere_vx[s1], mesh.sphere_vy[s1], mesh.sphere_vz[s1]},
            faces[f].r0, faces[f].r1, x_v);
      }
      double an = std::sqrt(fm.A[0] * fm.A[0] + fm.A[1] * fm.A[1] +
                            fm.A[2] * fm.A[2]);
      rscale[f] = 1.0 / an;
      for (int i = 0; i < 3; i++) rows[f * 12 + i] = fm.A[i] * rscale[f];
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          rows[f * 12 + 3 + 3 * i + j] = fm.M[i][j] * rscale[f];
    }

    // Rc = rows · NT  (nf x 11)
    std::vector<double> Rc(nf * 11, 0.0);
    for (int f = 0; f < nf; f++)
      for (int c = 0; c < 11; c++) {
        double acc = 0;
        for (int k = 0; k < 12; k++) acc += rows[f * 12 + k] * NT.N[k][c];
        Rc[f * 11 + c] = acc;
      }

    // Normal equations C = Rc^T Rc, E = Rc^T (11 x nf)
    double C[11][11] = {};
    for (int a = 0; a < 11; a++)
      for (int b = a; b < 11; b++) {
        double acc = 0;
        for (int f = 0; f < nf; f++) acc += Rc[f * 11 + a] * Rc[f * 11 + b];
        C[a][b] = C[b][a] = acc;
      }
    m_max_cond = std::max(m_max_cond, spd_cond_11(C));

    std::vector<double> E(11 * nf);
    for (int a = 0; a < 11; a++)
      for (int f = 0; f < nf; f++) E[a * nf + f] = Rc[f * 11 + a];
    cholesky_solve_11(C, E, nf);  // E := (Rc^T Rc)^{-1} Rc^T = pinv(Rc)

    // B0 weights: rows 0..2 of NT·pinv, times the row scaling
    for (int c = 0; c < 3; c++) {
      for (int j = 0; j < stride; j++) w_out[c * stride + j] = Scalar(0.0);
      for (int f = 0; f < nf; f++) {
        double acc = 0;
        for (int k = 0; k < 11; k++) acc += NT.N[c][k] * E[k * nf + f];
        w_out[c * stride + f] = Scalar(acc * rscale[f]);
      }
    }
  };

  for (int s = 0; s < m_N_vert_s; s++) {
    int val = valence[s];
    // interior class, reference shell k = 1
    {
      std::vector<pface> faces;
      for (int ks : {0, 1, 2})
        for (int j = 0; j < val; j++)
          faces.push_back({true, (double)mesh.radii[ks], 0, tri_fan[s * 6 + j]});
      for (int kl : {0, 1})
        for (int j = 0; j < val; j++)
          faces.push_back({false, (double)mesh.radii[kl],
                           (double)mesh.radii[kl + 1], edge_fan[s * 6 + j]});
      solve_patch(s, faces, mesh.radii[1], &w_int[s * 3 * SI], SI);
    }
    // inner boundary (k = 0)
    {
      std::vector<pface> faces;
      for (int ks : {0, 1})
        for (int j = 0; j < val; j++)
          faces.push_back({true, (double)mesh.radii[ks], 0, tri_fan[s * 6 + j]});
      for (int kl : {0, 1})
        for (int j = 0; j < val; j++)
          faces.push_back({false, (double)mesh.radii[kl],
                           (double)mesh.radii[kl + 1], edge_fan[s * 6 + j]});
      solve_patch(s, faces, mesh.radii[0], &w_inner[s * 3 * SB], SB);
    }
    // outer boundary (k = N_r)
    {
      std::vector<pface> faces;
      for (int ks : {m_N_r - 1, m_N_r})
        for (int j = 0; j < val; j++)
          faces.push_back({true, (double)mesh.radii[ks], 0, tri_fan[s * 6 + j]});
      for (int kl : {m_N_r - 1, m_N_r - 2})
        for (int j = 0; j < val; j++)
          faces.push_back({false, (double)mesh.radii[kl],
                           (double)mesh.radii[kl + 1], edge_fan[s * 6 + j]});
      solve_patch(s, faces, mesh.radii[m_N_r], &w_outer[s * 3 * SB], SB);
    }
  }

  Logger::print_info(
      "Vertex recovery built: {} sphere vertices, max patch condition {:.1f}",
      m_N_vert_s, m_max_cond);
}

void prismatic_vertex_recovery::copy_to_device() {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  valence.copy_to_device();
  tri_fan.copy_to_device();
  edge_fan.copy_to_device();
  w_int.copy_to_device();
  w_inner.copy_to_device();
  w_outer.copy_to_device();
  Bv.copy_to_device();
#endif
}

prismatic_recovery_ptrs prismatic_vertex_recovery::host_ptrs() {
  prismatic_recovery_ptrs p;
  p.N_vert_s = m_N_vert_s;
  p.N_r = m_N_r;
  p.N_verts = m_N_verts;
  p.r_ref = m_r_ref;
  p.valence = valence.host_ptr();
  p.tri_fan = tri_fan.host_ptr();
  p.edge_fan = edge_fan.host_ptr();
  p.w_int = w_int.host_ptr();
  p.w_inner = w_inner.host_ptr();
  p.w_outer = w_outer.host_ptr();
  p.Bv = Bv.host_ptr();
  return p;
}

prismatic_recovery_ptrs prismatic_vertex_recovery::dev_ptrs() {
  prismatic_recovery_ptrs p;
  p.N_vert_s = m_N_vert_s;
  p.N_r = m_N_r;
  p.N_verts = m_N_verts;
  p.r_ref = m_r_ref;
  p.valence = valence.dev_ptr();
  p.tri_fan = tri_fan.dev_ptr();
  p.edge_fan = edge_fan.dev_ptr();
  p.w_int = w_int.dev_ptr();
  p.w_inner = w_inner.dev_ptr();
  p.w_outer = w_outer.dev_ptr();
  p.Bv = Bv.dev_ptr();
  return p;
}

}  // namespace Aperture
