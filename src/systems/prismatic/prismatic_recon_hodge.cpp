#include "systems/prismatic/prismatic_recon_hodge.h"

#include "utils/logger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Aperture {

namespace {

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
inline v3 normalized(v3 a) { return (1.0 / norm(a)) * a; }

constexpr int NQ = 4;
constexpr double GX[NQ] = {0.06943184420297371, 0.33000947820757187,
                           0.6699905217924281, 0.9305681557970262};
constexpr double GW[NQ] = {0.1739274225687269, 0.3260725774312731,
                           0.3260725774312731, 0.1739274225687269};

struct moments {
  double A[3] = {};
  double M[3][3] = {};
  void add(v3 dA, v3 rel, double w) {
    double d[3] = {dA.x, dA.y, dA.z}, r[3] = {rel.x, rel.y, rel.z};
    for (int i = 0; i < 3; i++) {
      A[i] += w * d[i];
      for (int j = 0; j < 3; j++) M[i][j] += w * d[i] * r[j];
    }
  }
};

// spherical triangle (unit corners) at radius r
moments sphtri_moments(v3 u0, v3 u1, v3 u2, double r, v3 x_ref) {
  moments fm;
  for (int a = 0; a < NQ; a++)
    for (int b = 0; b < NQ; b++) {
      double l1 = GX[a], l2 = GX[b] * (1 - GX[a]);
      double w = GW[a] * GW[b] * (1 - GX[a]);
      v3 P = (1 - l1 - l2) * u0 + l1 * u1 + l2 * u2;
      double Pn = norm(P);
      v3 Nh = (1.0 / Pn) * P;
      v3 d1 = u1 - u0, d2 = u2 - u0;
      v3 dN1 = (1.0 / Pn) * (d1 - dot(Nh, d1) * Nh);
      v3 dN2 = (1.0 / Pn) * (d2 - dot(Nh, d2) * Nh);
      fm.add(cross(r * dN1, r * dN2), r * Nh - x_ref, w);
    }
  return fm;
}

// ruled patch r in [ra, rb] along the normalized-lerp arc u0 -> u1
moments ruled_moments(v3 u0, v3 u1, double ra, double rb, v3 x_ref) {
  moments fm;
  for (int a = 0; a < NQ; a++) {
    v3 P = (1 - GX[a]) * u0 + GX[a] * u1;
    double Pn = norm(P);
    v3 Nh = (1.0 / Pn) * P;
    v3 dNu = (1.0 / Pn) * ((u1 - u0) - dot(Nh, u1 - u0) * Nh);
    for (int b = 0; b < NQ; b++) {
      double r = ra + (rb - ra) * GX[b];
      fm.add(cross(r * dNu, (rb - ra) * Nh), r * Nh - x_ref,
             GW[a] * GW[b]);
    }
  }
  return fm;
}

// chord vector + moment K[i][j] = int dl_i (x - x_ref)_j along a curve
struct lineint {
  double chord[3] = {};
  double K[3][3] = {};
  void add(v3 dl, v3 rel, double w) {
    double d[3] = {dl.x, dl.y, dl.z}, r[3] = {rel.x, rel.y, rel.z};
    for (int i = 0; i < 3; i++) {
      chord[i] += w * d[i];
      for (int j = 0; j < 3; j++) K[i][j] += w * d[i] * r[j];
    }
  }
};

lineint arc_lineint(v3 u0, v3 u1, double r, v3 x_ref) {
  lineint li;
  for (int a = 0; a < NQ; a++) {
    v3 P = (1 - GX[a]) * u0 + GX[a] * u1;
    double Pn = norm(P);
    v3 Nh = (1.0 / Pn) * P;
    v3 dl = (r / Pn) * ((u1 - u0) - dot(Nh, u1 - u0) * Nh);
    li.add(dl, r * Nh - x_ref, GW[a]);
  }
  return li;
}

lineint radial_lineint(v3 u, double ra, double rb, v3 x_ref) {
  lineint li;
  for (int a = 0; a < NQ; a++) {
    double r = ra + (rb - ra) * GX[a];
    li.add((rb - ra) * u, r * u - x_ref, GW[a]);
  }
  return li;
}

// trace-free embedding (matches prismatic_vertex_recovery)
struct traceless {
  double N[12][11] = {};
  traceless() {
    for (int i = 0; i < 3; i++) N[i][i] = 1.0;
    int col = 3;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (i != j) N[3 + 3 * i + j][col++] = 1.0;
    const double s2 = 1.0 / std::sqrt(2.0), s6 = 1.0 / std::sqrt(6.0);
    N[3][9] = s2; N[7][9] = -s2;
    N[3][10] = s6; N[7][10] = s6; N[11][10] = -2 * s6;
  }
};

void cholesky_solve_11(double C[11][11], std::vector<double>& E, int nf) {
  double L[11][11] = {};
  for (int i = 0; i < 11; i++)
    for (int j = 0; j <= i; j++) {
      double s = C[i][j];
      for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i == j) {
        if (s <= 0.0)
          throw std::runtime_error("recon hodge: fit not SPD");
        L[i][i] = std::sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
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

// generic LSQ: rows (nf x 12) -> full pseudo-inverse weights (12 x nf)
std::vector<double> fit_weights(std::vector<double>& rows, int nf) {
  static const traceless NT;
  std::vector<double> scale(nf);
  for (int f = 0; f < nf; f++) {
    double an = std::sqrt(rows[f * 12 + 0] * rows[f * 12 + 0] +
                          rows[f * 12 + 1] * rows[f * 12 + 1] +
                          rows[f * 12 + 2] * rows[f * 12 + 2]);
    scale[f] = 1.0 / an;
    for (int j = 0; j < 12; j++) rows[f * 12 + j] *= scale[f];
  }
  std::vector<double> Rc(nf * 11, 0.0);
  for (int f = 0; f < nf; f++)
    for (int c = 0; c < 11; c++) {
      double acc = 0;
      for (int k = 0; k < 12; k++) acc += rows[f * 12 + k] * NT.N[k][c];
      Rc[f * 11 + c] = acc;
    }
  double C[11][11] = {};
  for (int a = 0; a < 11; a++)
    for (int b = a; b < 11; b++) {
      double acc = 0;
      for (int f = 0; f < nf; f++) acc += Rc[f * 11 + a] * Rc[f * 11 + b];
      C[a][b] = C[b][a] = acc;
    }
  std::vector<double> E(11 * nf);
  for (int a = 0; a < 11; a++)
    for (int f = 0; f < nf; f++) E[a * nf + f] = Rc[f * 11 + a];
  cholesky_solve_11(C, E, nf);
  std::vector<double> W(12 * nf, 0.0);
  for (int r = 0; r < 12; r++)
    for (int f = 0; f < nf; f++) {
      double acc = 0;
      for (int k = 0; k < 11; k++) acc += NT.N[r][k] * E[k * nf + f];
      W[r * nf + f] = acc * scale[f];
    }
  return W;
}

}  // namespace

void prismatic_recon_hodge::build(const prismatic_mesh& mesh) {
  m_mesh = &mesh;
  m_N_r = mesh.m_N_r;
  m_N_tri = mesh.m_N_tri;
  m_N_vert_s = mesh.m_N_vert_s;
  m_N_edge_s = mesh.m_N_edge_s;
  if (m_N_r < 6)
    throw std::runtime_error("recon hodge: needs N_r >= 6");
  const int kref = 2;
  m_r_ref = mesh.radii[kref];

  auto sv = [&](int s) -> v3 {
    return {mesh.sphere_vx[s], mesh.sphere_vy[s], mesh.sphere_vz[s]};
  };

  // ---- fan tables (azimuth-ordered tri fan for the dual polygons) ----
  valence.resize(m_N_vert_s);
  tri_fan.resize(6 * m_N_vert_s);
  edge_fan.resize(6 * m_N_vert_s);
  nbr_fan.resize(6 * m_N_vert_s);
  tri_anchor.resize(m_N_tri);
  edge_anchor.resize(m_N_edge_s);
  for (int i = 0; i < 6 * m_N_vert_s; i++)
    tri_fan[i] = edge_fan[i] = nbr_fan[i] = 0;

  std::vector<std::vector<int>> vtris(m_N_vert_s), vedges(m_N_vert_s);
  std::vector<std::array<int, 2>> sedges(m_N_edge_s);
  for (int e = 0; e < m_N_edge_s; e++) {
    sedges[e] = {mesh.edge_v0[e], mesh.edge_v1[e]};
    vedges[sedges[e][0]].push_back(e);
    vedges[sedges[e][1]].push_back(e);
  }
  for (int t = 0; t < m_N_tri; t++) {
    tri_anchor[t] = mesh.tri_verts[t * 3];
    for (int j = 0; j < 3; j++) vtris[mesh.tri_verts[t * 3 + j]].push_back(t);
  }
  for (int e = 0; e < m_N_edge_s; e++) edge_anchor[e] = sedges[e][0];

  // circumcenter directions
  std::vector<v3> circ(m_N_tri);
  for (int t = 0; t < m_N_tri; t++) {
    v3 a = sv(mesh.tri_verts[t * 3]), b = sv(mesh.tri_verts[t * 3 + 1]),
       c = sv(mesh.tri_verts[t * 3 + 2]);
    v3 u = normalized(cross(b - a, c - a));
    if (dot(u, a) < 0) u = -1.0 * u;
    circ[t] = u;
  }
  // sphere-edge -> adjacent triangles
  std::vector<std::array<int, 2>> etris(m_N_edge_s, {-1, -1});
  for (int t = 0; t < m_N_tri; t++)
    for (int j = 0; j < 3; j++) {
      int e = mesh.tri_edges_s[t * 3 + j];
      if (etris[e][0] < 0) etris[e][0] = t;
      else etris[e][1] = t;
    }

  for (int s = 0; s < m_N_vert_s; s++) {
    int val = (int)vtris[s].size();
    if (val != (int)vedges[s].size() || (val != 5 && val != 6))
      throw std::runtime_error("recon hodge: bad fan");
    valence[s] = val;
    v3 u = sv(s);
    v3 ax = (std::abs(u.z) < 0.9) ? v3{0, 0, 1} : v3{1, 0, 0};
    v3 t1 = normalized(cross(ax, u));
    v3 t2 = cross(u, t1);
    std::vector<std::pair<double, int>> to;
    for (int t : vtris[s]) {
      v3 d = circ[t] - u;
      to.push_back({std::atan2(dot(d, t2), dot(d, t1)), t});
    }
    std::sort(to.begin(), to.end());
    std::vector<std::pair<double, int>> eo;
    for (int e : vedges[s]) {
      int s2 = sedges[e][0] + sedges[e][1] - s;
      v3 d = sv(s2) - u;
      eo.push_back({std::atan2(dot(d, t2), dot(d, t1)), e});
    }
    std::sort(eo.begin(), eo.end());
    for (int j = 0; j < val; j++) {
      tri_fan[s * 6 + j] = to[j].second;
      edge_fan[s * 6 + j] = eo[j].second;
      nbr_fan[s * 6 + j] =
          sedges[eo[j].second][0] + sedges[eo[j].second][1] - s;
    }
  }

  // ---- per-vertex fits at the reference shell ----
  auto r_mid = [&](int k) {
    return 0.5 * (mesh.radii[k] + mesh.radii[k + 1]);
  };

  // B-side fit rows at (s, kv): faces in w2_cols order
  auto bfit = [&](int s, int kv) {
    int val = valence[s];
    int nf = 5 * val;
    v3 x_v = mesh.radii[kv] * sv(s);
    std::vector<double> rows(nf * 12);
    int m = 0;
    for (int o = -1; o <= 1; o++)
      for (int j = 0; j < val; j++) {
        int t = tri_fan[s * 6 + j];
        moments fm = sphtri_moments(sv(mesh.tri_verts[t * 3]),
                                    sv(mesh.tri_verts[t * 3 + 1]),
                                    sv(mesh.tri_verts[t * 3 + 2]),
                                    mesh.radii[kv + o], x_v);
        for (int i = 0; i < 3; i++) rows[m * 12 + i] = fm.A[i];
        for (int i = 0; i < 9; i++) rows[m * 12 + 3 + i] = fm.M[i / 3][i % 3];
        m++;
      }
    for (int o = -1; o <= 0; o++)
      for (int j = 0; j < val; j++) {
        int e = edge_fan[s * 6 + j];
        moments fm = ruled_moments(sv(sedges[e][0]), sv(sedges[e][1]),
                                   mesh.radii[kv + o], mesh.radii[kv + o + 1],
                                   x_v);
        for (int i = 0; i < 3; i++) rows[m * 12 + i] = fm.A[i];
        for (int i = 0; i < 9; i++) rows[m * 12 + 3 + i] = fm.M[i / 3][i % 3];
        m++;
      }
    return fit_weights(rows, nf);
  };

  // pairing fit rows at (s, kv): dual faces in w1_cols order
  auto pfit = [&](int s, int kv) {
    int val = valence[s];
    int nf = 3 * val + 2 * (1 + val);
    v3 x_v = mesh.radii[kv] * sv(s);
    std::vector<double> rows(nf * 12);
    int m = 0;
    auto put = [&](const moments& fm, v3 tvec) {
      double sgn = (fm.A[0] * tvec.x + fm.A[1] * tvec.y +
                    fm.A[2] * tvec.z) >= 0 ? 1.0 : -1.0;
      for (int i = 0; i < 3; i++) rows[m * 12 + i] = sgn * fm.A[i];
      for (int i = 0; i < 9; i++)
        rows[m * 12 + 3 + i] = sgn * fm.M[i / 3][i % 3];
      m++;
    };
    for (int o = -1; o <= 1; o++)
      for (int j = 0; j < val; j++) {
        int e = edge_fan[s * 6 + j];
        moments fm = ruled_moments(circ[etris[e][0]], circ[etris[e][1]],
                                   r_mid(kv + o - 1), r_mid(kv + o), x_v);
        put(fm, sv(sedges[e][1]) - sv(sedges[e][0]));
      }
    for (int o = -1; o <= 0; o++) {
      // v-edge dual polygons: anchor first, then fan neighbors
      auto vpoly = [&](int s2) {
        moments fm;
        int v2 = valence[s2];
        for (int i = 0; i < v2; i++) {
          moments w = sphtri_moments(
              sv(s2), circ[tri_fan[s2 * 6 + i]],
              circ[tri_fan[s2 * 6 + (i + 1) % v2]], r_mid(kv + o), x_v);
          for (int a = 0; a < 3; a++) {
            fm.A[a] += w.A[a];
            for (int b = 0; b < 3; b++) fm.M[a][b] += w.M[a][b];
          }
        }
        put(fm, sv(s2));
      };
      vpoly(s);
      for (int j = 0; j < val; j++) vpoly(nbr_fan[s * 6 + j]);
    }
    return fit_weights(rows, nf);
  };

  // ---- assemble template rows ----
  constexpr int S2 = recon_hodge_ptrs::W2_STRIDE;
  constexpr int S1 = recon_hodge_ptrs::W1_STRIDE;
  w2_tri.resize(m_N_tri * S2);
  w2_rect.resize(m_N_edge_s * S2);
  w1_h.resize(m_N_edge_s * S1);
  w1_v.resize(m_N_vert_s * S1);
  for (int i = 0; i < m_N_tri * S2; i++) w2_tri[i] = 0;
  for (int i = 0; i < m_N_edge_s * S2; i++) w2_rect[i] = 0;
  for (int i = 0; i < m_N_edge_s * S1; i++) w1_h[i] = 0;
  for (int i = 0; i < m_N_vert_s * S1; i++) w1_v[i] = 0;

  auto apply_line = [&](const lineint& li, const std::vector<double>& W,
                        int nf, Scalar* out, double sgn) {
    for (int f = 0; f < nf; f++) {
      double acc = 0;
      for (int i = 0; i < 3; i++) acc += li.chord[i] * W[i * nf + f];
      for (int i = 0; i < 9; i++)
        acc += li.K[i / 3][i % 3] * W[(3 + i) * nf + f];
      out[f] = Scalar(sgn * acc);
    }
  };

  // cache one fit per sphere vertex (all elements anchored at kref)
  std::vector<std::vector<double>> bW(m_N_vert_s), pW(m_N_vert_s);
  for (int s = 0; s < m_N_vert_s; s++) {
    bW[s] = bfit(s, kref);
    pW[s] = pfit(s, kref);
  }

  for (int t = 0; t < m_N_tri; t++) {
    int s = tri_anchor[t];
    v3 x_v = mesh.radii[kref] * sv(s);
    lineint li = radial_lineint(circ[t], r_mid(kref - 1), r_mid(kref), x_v);
    apply_line(li, bW[s], 5 * valence[s], &w2_tri[t * S2], 1.0);
  }
  for (int e = 0; e < m_N_edge_s; e++) {
    int s = edge_anchor[e];
    v3 x_v = mesh.radii[kref] * sv(s);
    lineint li = arc_lineint(circ[etris[e][0]], circ[etris[e][1]],
                             r_mid(kref), x_v);
    // sign: dual arc must run along the rect face normal (stored order)
    int fi = kref * m_N_edge_s + e;
    int sa = mesh.rect_face_v0[fi] % m_N_vert_s;
    int sb = mesh.rect_face_v1[fi] % m_N_vert_s;
    v3 nf3 = cross(sv(sb) - sv(sa), normalized(sv(sa) + sv(sb)));
    v3 ch = r_mid(kref) * (normalized(circ[etris[e][1]]) -
                           normalized(circ[etris[e][0]]));
    double sgn = dot(ch, nf3) >= 0 ? 1.0 : -1.0;
    apply_line(li, bW[s], 5 * valence[s], &w2_rect[e * S2], sgn);
  }
  for (int e = 0; e < m_N_edge_s; e++) {
    int s = edge_anchor[e];
    v3 x_v = mesh.radii[kref] * sv(s);
    lineint li = arc_lineint(sv(sedges[e][0]), sv(sedges[e][1]),
                             mesh.radii[kref], x_v);
    apply_line(li, pW[s], 5 * valence[s] + 2, &w1_h[e * S1], 1.0);
  }
  for (int s = 0; s < m_N_vert_s; s++) {
    v3 x_v = mesh.radii[kref] * sv(s);
    lineint li = radial_lineint(sv(s), mesh.radii[kref],
                                mesh.radii[kref + 1], x_v);
    apply_line(li, pW[s], 5 * valence[s] + 2, &w1_v[s * S1], 1.0);
  }

  // ---- verify the r_ref/r_k template scaling law directly ----
  // (the whole per-sphere-element storage rests on it: rows must scale
  // exactly as 1/r across shells of the geometric grid)
  {
    int kchk = kref + 2;
    if (kchk > m_N_r - 1) kchk = m_N_r - 1;
    int t = 0, s = tri_anchor[t];
    auto Wchk = bfit(s, kchk);
    v3 x_v = mesh.radii[kchk] * sv(s);
    lineint li = radial_lineint(circ[t], r_mid(kchk - 1), r_mid(kchk), x_v);
    int nf = 5 * valence[s];
    double max_rel = 0.0, scale = mesh.radii[kref] / mesh.radii[kchk];
    double ref_mag = 0.0;
    for (int f = 0; f < nf; f++) ref_mag = std::max(
        ref_mag, std::abs((double)w2_tri[t * recon_hodge_ptrs::W2_STRIDE + f]));
    for (int f = 0; f < nf; f++) {
      double acc = 0;
      for (int i = 0; i < 3; i++) acc += li.chord[i] * Wchk[i * nf + f];
      for (int i = 0; i < 9; i++)
        acc += li.K[i / 3][i % 3] * Wchk[(3 + i) * nf + f];
      double templ = (double)w2_tri[t * recon_hodge_ptrs::W2_STRIDE + f] *
                     scale;
      max_rel = std::max(max_rel, std::abs(acc - templ) / ref_mag);
    }
    if (max_rel > 1e-4)
      throw std::runtime_error(
          "recon hodge: template scaling law violated (non-geometric "
          "shells?)");
    Logger::print_info(
        "Reconstruction Hodge built: ref shell {} (r={:.3f}), scaling "
        "check {:.2e}", kref, m_r_ref, max_rel);
  }
}

void prismatic_recon_hodge::copy_to_device() {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  valence.copy_to_device();
  tri_fan.copy_to_device();
  edge_fan.copy_to_device();
  nbr_fan.copy_to_device();
  tri_anchor.copy_to_device();
  edge_anchor.copy_to_device();
  w2_tri.copy_to_device();
  w2_rect.copy_to_device();
  w1_h.copy_to_device();
  w1_v.copy_to_device();
#endif
}

recon_hodge_ptrs prismatic_recon_hodge::host_ptrs() const {
  recon_hodge_ptrs p;
  p.N_r = m_N_r; p.N_tri = m_N_tri;
  p.N_vert_s = m_N_vert_s; p.N_edge_s = m_N_edge_s;
  p.r_ref = m_r_ref;
  p.radii = m_mesh->radii.host_ptr();
  p.valence = valence.host_ptr();
  p.tri_fan = tri_fan.host_ptr();
  p.edge_fan = edge_fan.host_ptr();
  p.nbr_fan = nbr_fan.host_ptr();
  p.tri_anchor = tri_anchor.host_ptr();
  p.edge_anchor = edge_anchor.host_ptr();
  p.w2_tri = w2_tri.host_ptr();
  p.w2_rect = w2_rect.host_ptr();
  p.w1_h = w1_h.host_ptr();
  p.w1_v = w1_v.host_ptr();
  return p;
}

recon_hodge_ptrs prismatic_recon_hodge::dev_ptrs() const {
  recon_hodge_ptrs p;
  p.N_r = m_N_r; p.N_tri = m_N_tri;
  p.N_vert_s = m_N_vert_s; p.N_edge_s = m_N_edge_s;
  p.r_ref = m_r_ref;
  p.radii = m_mesh->radii.dev_ptr();
  p.valence = valence.dev_ptr();
  p.tri_fan = tri_fan.dev_ptr();
  p.edge_fan = edge_fan.dev_ptr();
  p.nbr_fan = nbr_fan.dev_ptr();
  p.tri_anchor = tri_anchor.dev_ptr();
  p.edge_anchor = edge_anchor.dev_ptr();
  p.w2_tri = w2_tri.dev_ptr();
  p.w2_rect = w2_rect.dev_ptr();
  p.w1_h = w1_h.dev_ptr();
  p.w1_v = w1_v.dev_ptr();
  return p;
}

}  // namespace Aperture
