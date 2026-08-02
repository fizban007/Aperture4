#include "systems/prismatic/prismatic_whitney_hodge.h"
#include "systems/physics/spherical_metric.hpp"
#include "utils/hdf_wrapper.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <vector>

namespace Aperture {

namespace {

// 3-point (edge-midpoint) triangle rule, degree 2.  Row q gives the
// barycentric coordinates of the midpoint of local edge q = (q, q+1).
constexpr double TRI_L[3][3] = {
    {0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}};
constexpr double TRI_W = 1.0 / 3.0;
// 2-point Gauss in ζ.
const double GZ[2] = {0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
constexpr double GW = 0.5;

double wrap_pi(double x) {
  const double pi = 3.14159265358979323846;
  while (x > pi) x -= 2.0 * pi;
  while (x < -pi) x += 2.0 * pi;
  return x;
}

// File-local inverse spatial metric, axis-regular forms.  Kept out of
// spherical_metric.hpp deliberately: that header is on the validated
// flat-space paper path and stays untouched (user request 2026-08-01).
// All sin θ cancellations are done analytically except the genuine
// 1/sin²θ in γ^φφ, which is only ever evaluated at quadrature points
// strictly off the axis (edge midpoints of a triangle with a polar
// vertex all have θ ≥ O(h)).
struct inv_metric3 {
  double gu_rr, gu_rph, gu_thth, gu_phph;
};

inv_metric3 inverse_metric(const flat_spherical_metric&, double r, double sth,
                           double cth) {
  return {1.0, 0.0, 1.0 / (r * r), 1.0 / (r * r * sth * sth)};
}

// Metric_KS::gu* are already the axis-regular forms: gu11 is
// algebraically identical to P/(γ_rr ρ²) with the sin²θ cancelled
// (verified against the lab's regular expression), gu13 = a/ρ²,
// gu22 = 1/ρ²; only gu33 keeps the true 1/sin²θ.
inv_metric3 inverse_metric(const ks_spherical_metric& m, double r, double sth,
                           double cth) {
  return {(double)Metric_KS::gu11(m.a, r, sth, cth),
          (double)Metric_KS::gu13(m.a, r, sth, cth),
          (double)Metric_KS::gu22(m.a, r, sth, cth),
          (double)Metric_KS::gu33(m.a, r, sth, cth)};
}

// Per-sphere-triangle affine-chart data, reused across all radial layers.
struct tri_chart {
  double th[3];        // vertex θ
  double lam_c[3][3];  // λ_i(u) = lam_c[0][i] + lam_c[1][i]·θ + lam_c[2][i]·φ
  double grad[3][2];   // ∇λ_i in (θ, φ)
  double Ac;           // coordinate area of the (θ, φ) triangle
  double sgnA;         // orientation sign of the (θ, φ) triangle
  double es[3];        // tri_edge_signs (local i→j vs global v0→v1)
  // Quadrature-point data (z-independent): θ_q and the horizontal edge
  // 1-form 2-vectors w2[q][j] = λ_a ∇λ_b − λ_b ∇λ_a at point q, plus the
  // signed RT0 rotations rot[q][j] = −rot90(w2)·sgnA·es (the −1 is the
  // rect-face orientation relative to the tri_edge_signs RT0 basis).
  double th_q[3];
  double w2[3][3][2];   // [q][j][θφ], already × es[j]
  double rot[3][3][2];  // [q][j][θφ], already × (−sgnA·es[j])
  double lam_q[3][3];   // [q][i]
};

}  // namespace

template <typename Metric>
void prismatic_whitney_hodge::build(const prismatic_mesh_metric& mesh,
                                    const Metric& met) {
  timer::stamp("whitney_build");
  const int N_r = mesh.m_N_r;
  const int Ntri = mesh.m_N_tri;
  const int Nes = mesh.m_N_edge_s;
  const int Nvs = mesh.m_N_vert_s;
  const int Ne = int(mesh.m_N_edges);
  const int Nf = int(mesh.m_N_faces);
  const int Nh = (N_r + 1) * Nes;       // horizontal edge count
  const int Ntf = (N_r + 1) * Ntri;     // tri face count
  m_N_edges = Ne;
  m_N_faces = Nf;

  const int* tvs = mesh.tri_verts.host_ptr();
  const int* te = mesh.tri_edges_s.host_ptr();
  const int* tes = mesh.tri_edge_signs.host_ptr();
  const Scalar* svz = mesh.sphere_vz.host_ptr();
  const Scalar* sph = mesh.sphere_phi.host_ptr();
  const Scalar* radii = mesh.radii.host_ptr();

  // ---- sphere adjacency (local; independent of compute_metric's copy) ----
  std::vector<std::array<int, 2>> edge_tri(Nes, {-1, -1});
  std::vector<int> vert_fan_cnt(Nvs, 0);
  std::vector<std::array<int, 6>> vert_fan(Nvs);
  for (int t = 0; t < Ntri; t++) {
    for (int j = 0; j < 3; j++) {
      int e = te[t * 3 + j];
      if (edge_tri[e][0] < 0)
        edge_tri[e][0] = t;
      else
        edge_tri[e][1] = t;
      int s = tvs[t * 3 + j];
      vert_fan[s][vert_fan_cnt[s]++] = t;
    }
  }

  // ---- per-triangle affine chart + quadrature-point basis data ----
  std::vector<tri_chart> charts(Ntri);
#pragma omp parallel for schedule(static)
  for (int t = 0; t < Ntri; t++) {
    tri_chart& c = charts[t];
    double ph[3];
    for (int i = 0; i < 3; i++) {
      int s = tvs[t * 3 + i];
      double ct = (double)svz[s];
      if (ct > 1.0) ct = 1.0;
      if (ct < -1.0) ct = -1.0;
      c.th[i] = std::acos(ct);
      ph[i] = (double)sph[s];
      c.es[i] = (double)tes[t * 3 + i];
    }
    // Unwrap φ about vertex 0 (matches the lab).
    for (int i = 1; i < 3; i++) ph[i] = ph[0] + wrap_pi(ph[i] - ph[0]);

    // λ coefficients: invert M = [[1, θ_i, φ_i]].
    double m[3][3] = {{1.0, c.th[0], ph[0]},
                      {1.0, c.th[1], ph[1]},
                      {1.0, c.th[2], ph[2]}};
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    double inv = 1.0 / det;
    // lam_c[r][i] = (M⁻¹)[r][i]; cofactor expansion.
    c.lam_c[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv;
    c.lam_c[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
    c.lam_c[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
    c.lam_c[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv;
    c.lam_c[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
    c.lam_c[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
    c.lam_c[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv;
    c.lam_c[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
    c.lam_c[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
    // But note: with M[v][*] = (1, θ_v, φ_v) and λ_i(u) = Σ_r lam_c[r][i]·u_r
    // (u = (1, θ, φ)), the defining property is λ_i(vertex v) = δ_iv, i.e.
    // M · lam_c = I, so lam_c really is M⁻¹ with [row = u-component,
    // col = i], which the cofactors above produce.
    for (int i = 0; i < 3; i++) {
      c.grad[i][0] = c.lam_c[1][i];
      c.grad[i][1] = c.lam_c[2][i];
    }
    double cross = (c.th[1] - c.th[0]) * (ph[2] - ph[0]) -
                   (ph[1] - ph[0]) * (c.th[2] - c.th[0]);
    c.Ac = 0.5 * std::abs(cross);
    c.sgnA = (cross >= 0) ? 1.0 : -1.0;

    for (int q = 0; q < 3; q++) {
      double thq = 0.0;
      for (int i = 0; i < 3; i++) {
        c.lam_q[q][i] = TRI_L[q][i];
        thq += TRI_L[q][i] * c.th[i];
      }
      c.th_q[q] = thq;
      for (int j = 0; j < 3; j++) {
        int a = j, b = (j + 1) % 3;
        double w2t = TRI_L[q][a] * c.grad[b][0] - TRI_L[q][b] * c.grad[a][0];
        double w2p = TRI_L[q][a] * c.grad[b][1] - TRI_L[q][b] * c.grad[a][1];
        c.w2[q][j][0] = w2t * c.es[j];
        c.w2[q][j][1] = w2p * c.es[j];
        // rot90(w2) = (−w2_φ, w2_θ), then × sgnA (outward RT0), then the
        // global −1 rect-face orientation, then the edge sign.
        c.rot[q][j][0] = -(-w2p) * c.sgnA * c.es[j];
        c.rot[q][j][1] = -(w2t)*c.sgnA * c.es[j];
      }
    }
  }

  // ---- global DOF indices of prism (k, t) ----
  auto prism_edges = [&](int k, int t, int idx[9]) {
    for (int j = 0; j < 3; j++) {
      idx[j] = k * Nes + te[t * 3 + j];
      idx[3 + j] = (k + 1) * Nes + te[t * 3 + j];
      idx[6 + j] = Nh + k * Nvs + tvs[t * 3 + j];
    }
  };
  auto prism_faces = [&](int k, int t, int idx[5]) {
    idx[0] = k * Ntri + t;
    idx[1] = (k + 1) * Ntri + t;
    for (int j = 0; j < 3; j++) idx[2 + j] = Ntf + k * Nes + te[t * 3 + j];
  };

  // ---- CSR sparsity: per-row column enumeration ----
  Logger::print_info("whitney_hodge: building sparsity ({} edges, {} faces)",
                     Ne, Nf);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  constexpr MemType mem = MemType::host_device;
#else
  constexpr MemType mem = MemType::host_only;
#endif
  m1_row_ptr.set_memtype(mem);
  m2_row_ptr.set_memtype(mem);
  m1_row_ptr.resize(Ne + 1);
  m2_row_ptr.resize(Nf + 1);

  std::vector<int> m1_cols_tmp;  // filled row-by-row below
  std::vector<int> m2_cols_tmp;
  std::vector<int> c1_cols_tmp;
  std::vector<int> c1t_cols_tmp;
  {
    // First pass: count + collect per row (serial, cheap relative to fill).
    std::vector<int> cols;
    cols.reserve(64);
    int* rp = m1_row_ptr.host_ptr();
    rp[0] = 0;
    for (int e = 0; e < Ne; e++) {
      cols.clear();
      int idx[9];
      if (e < Nh) {
        int k = e / Nes, es_ = e % Nes;
        for (int kk = k - 1; kk <= k; kk++) {
          if (kk < 0 || kk >= N_r) continue;
          for (int a = 0; a < 2; a++) {
            int t = edge_tri[es_][a];
            if (t < 0) continue;
            prism_edges(kk, t, idx);
            cols.insert(cols.end(), idx, idx + 9);
          }
        }
      } else {
        int li = e - Nh;
        int k = li / Nvs, s = li % Nvs;
        for (int a = 0; a < vert_fan_cnt[s]; a++) {
          prism_edges(k, vert_fan[s][a], idx);
          cols.insert(cols.end(), idx, idx + 9);
        }
      }
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
      m1_cols_tmp.insert(m1_cols_tmp.end(), cols.begin(), cols.end());
      rp[e + 1] = int(m1_cols_tmp.size());
    }

    // C1 pattern: edge rows × faces of the prisms containing the edge.
    c1_row_ptr.set_memtype(mem);
    c1_row_ptr.resize(Ne + 1);
    int* rpc = c1_row_ptr.host_ptr();
    rpc[0] = 0;
    int fidx[5];
    for (int e = 0; e < Ne; e++) {
      cols.clear();
      if (e < Nh) {
        int k = e / Nes, es_ = e % Nes;
        for (int kk = k - 1; kk <= k; kk++) {
          if (kk < 0 || kk >= N_r) continue;
          for (int a = 0; a < 2; a++) {
            int t = edge_tri[es_][a];
            if (t < 0) continue;
            prism_faces(kk, t, fidx);
            cols.insert(cols.end(), fidx, fidx + 5);
          }
        }
      } else {
        int li = e - Nh;
        int k = li / Nvs, s = li % Nvs;
        for (int a = 0; a < vert_fan_cnt[s]; a++) {
          prism_faces(k, vert_fan[s][a], fidx);
          cols.insert(cols.end(), fidx, fidx + 5);
        }
      }
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
      c1_cols_tmp.insert(c1_cols_tmp.end(), cols.begin(), cols.end());
      rpc[e + 1] = int(c1_cols_tmp.size());
    }

    // C1ᵀ pattern: face rows × edges of the prisms containing the face.
    c1t_row_ptr.set_memtype(mem);
    c1t_row_ptr.resize(Nf + 1);
    int* rpct = c1t_row_ptr.host_ptr();
    rpct[0] = 0;
    int eidx9[9];
    for (int f = 0; f < Nf; f++) {
      cols.clear();
      if (f < Ntf) {
        int k = f / Ntri, t = f % Ntri;
        for (int kk = k - 1; kk <= k; kk++) {
          if (kk < 0 || kk >= N_r) continue;
          prism_edges(kk, t, eidx9);
          cols.insert(cols.end(), eidx9, eidx9 + 9);
        }
      } else {
        int ri = f - Ntf;
        int k = ri / Nes, es_ = ri % Nes;
        for (int a = 0; a < 2; a++) {
          int t = edge_tri[es_][a];
          if (t < 0) continue;
          prism_edges(k, t, eidx9);
          cols.insert(cols.end(), eidx9, eidx9 + 9);
        }
      }
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
      c1t_cols_tmp.insert(c1t_cols_tmp.end(), cols.begin(), cols.end());
      rpct[f + 1] = int(c1t_cols_tmp.size());
    }

    int* rp2 = m2_row_ptr.host_ptr();
    rp2[0] = 0;
    for (int f = 0; f < Nf; f++) {
      cols.clear();
      if (f < Ntf) {
        int k = f / Ntri, t = f % Ntri;
        for (int kk = k - 1; kk <= k; kk++) {
          if (kk < 0 || kk >= N_r) continue;
          prism_faces(kk, t, fidx);
          cols.insert(cols.end(), fidx, fidx + 5);
        }
      } else {
        int ri = f - Ntf;
        int k = ri / Nes, es_ = ri % Nes;
        for (int a = 0; a < 2; a++) {
          int t = edge_tri[es_][a];
          if (t < 0) continue;
          prism_faces(k, t, fidx);
          cols.insert(cols.end(), fidx, fidx + 5);
        }
      }
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
      m2_cols_tmp.insert(m2_cols_tmp.end(), cols.begin(), cols.end());
      rp2[f + 1] = int(m2_cols_tmp.size());
    }
  }
  const size_t nnz1 = m1_cols_tmp.size();
  const size_t nnz2 = m2_cols_tmp.size();
  const size_t nnzc = c1_cols_tmp.size();
  const size_t nnzct = c1t_cols_tmp.size();
  Logger::print_info("whitney_hodge: M1 nnz = {}, M2 nnz = {}, C1 nnz = {}",
                     nnz1, nnz2, nnzc);

  auto alloc_i = [mem](buffer<int>& b, size_t n) {
    b.set_memtype(mem);
    b.resize(n);
  };
  auto alloc_s = [mem](buffer<Scalar>& b, size_t n) {
    b.set_memtype(mem);
    b.resize(n);
  };
  alloc_i(m1_col_idx, nnz1);
  alloc_i(m2_col_idx, nnz2);
  alloc_i(c1_col_idx, nnzc);
  alloc_i(c1t_col_idx, nnzct);
  std::copy(m1_cols_tmp.begin(), m1_cols_tmp.end(), m1_col_idx.host_ptr());
  std::copy(m2_cols_tmp.begin(), m2_cols_tmp.end(), m2_col_idx.host_ptr());
  std::copy(c1_cols_tmp.begin(), c1_cols_tmp.end(), c1_col_idx.host_ptr());
  std::copy(c1t_cols_tmp.begin(), c1t_cols_tmp.end(),
            c1t_col_idx.host_ptr());
  m1_cols_tmp.clear();
  m1_cols_tmp.shrink_to_fit();
  m2_cols_tmp.clear();
  m2_cols_tmp.shrink_to_fit();
  c1_cols_tmp.clear();
  c1_cols_tmp.shrink_to_fit();
  c1t_cols_tmp.clear();
  c1t_cols_tmp.shrink_to_fit();

  // Double-precision accumulation, cast to Scalar at the end.
  std::vector<double> v1(nnz1, 0.0), v1a(nnz1, 0.0), v2a(nnz2, 0.0);
  std::vector<double> vc1(nnzc, 0.0), vc1t(nnzct, 0.0);

  const int* m1_rp = m1_row_ptr.host_ptr();
  const int* m1_ci = m1_col_idx.host_ptr();
  const int* m2_rp = m2_row_ptr.host_ptr();
  const int* m2_ci = m2_col_idx.host_ptr();
  const int* c1_rp = c1_row_ptr.host_ptr();
  const int* c1_ci = c1_col_idx.host_ptr();
  const int* c1t_rp = c1t_row_ptr.host_ptr();
  const int* c1t_ci = c1t_col_idx.host_ptr();
  auto scatter = [](const int* rp, const int* ci, std::vector<double>& v,
                    int row, int col, double val) {
    const int lo = rp[row], hi = rp[row + 1];
    const int* first = ci + lo;
    const int* last = ci + hi;
    const int* it = std::lower_bound(first, last, col);
    v[lo + int(it - first)] += val;
  };

  // ---- fill: loop prisms, even layers then odd (rows of layer k touch
  // only shells k, k+1 / layer k, so prisms two layers apart are
  // write-disjoint and each parity pass is safe to parallelize) ----
  Logger::print_info("whitney_hodge: assembling {} prisms", N_r * Ntri);
  for (int parity = 0; parity < 2; parity++) {
#pragma omp parallel for schedule(dynamic)
    for (int k = parity; k < N_r; k += 2) {
      const double r0 = (double)radii[k], r1 = (double)radii[k + 1];
      const double dr = r1 - r0;
      double W[9][3], Bd[5][3];
      double M1l[9][9], M1al[9][9], M2al[5][5], C1l[9][5];
      int eidx[9], fidx[5];
      for (int t = 0; t < Ntri; t++) {
        const tri_chart& c = charts[t];
        for (auto& row : M1l)
          for (auto& x : row) x = 0.0;
        for (auto& row : M1al)
          for (auto& x : row) x = 0.0;
        for (auto& row : M2al)
          for (auto& x : row) x = 0.0;
        for (auto& row : C1l)
          for (auto& x : row) x = 0.0;

        for (int iz = 0; iz < 2; iz++) {
          const double z = GZ[iz];
          const double r = r0 + z * dr;
          for (int q = 0; q < 3; q++) {
            const double th = c.th_q[q];
            const double sth = std::sin(th), cth = std::cos(th);
            const double sqg = (double)met.sqrt_gamma(r, sth, cth);
            const double al = (double)met.alpha(r, sth, cth);
            const inv_metric3 gu = inverse_metric(met, r, sth, cth);
            const double gu_rr = gu.gu_rr;
            const double gu_rp = gu.gu_rph;
            const double gu_tt = gu.gu_thth;
            const double gu_pp = gu.gu_phph;
            const double g_rr = (double)met.g_rr(r, sth, cth);
            const double g_rp = (double)met.g_rph(r, sth, cth);
            const double g_tt = (double)met.g_thth(r, sth, cth);
            const double g_pp = (double)met.g_phph(r, sth, cth);

            // 1-form basis, covariant (r, θ, φ).
            for (int j = 0; j < 3; j++) {
              W[j][0] = 0.0;
              W[j][1] = c.w2[q][j][0] * (1.0 - z);
              W[j][2] = c.w2[q][j][1] * (1.0 - z);
              W[3 + j][0] = 0.0;
              W[3 + j][1] = c.w2[q][j][0] * z;
              W[3 + j][2] = c.w2[q][j][1] * z;
              W[6 + j][0] = c.lam_q[q][j] / dr;
              W[6 + j][1] = 0.0;
              W[6 + j][2] = 0.0;
            }
            // 2-form densitized proxy Bd^i = √γ B^i.
            Bd[0][0] = (1.0 - z) / c.Ac;
            Bd[0][1] = Bd[0][2] = 0.0;
            Bd[1][0] = z / c.Ac;
            Bd[1][1] = Bd[1][2] = 0.0;
            for (int j = 0; j < 3; j++) {
              Bd[2 + j][0] = 0.0;
              Bd[2 + j][1] = c.rot[q][j][0] / dr;
              Bd[2 + j][2] = c.rot[q][j][1] / dr;
            }

            const double jac = GW * TRI_W * dr * c.Ac;
            const double w1 = jac * sqg;
            const double w2f = jac * al / sqg;
            for (int a = 0; a < 9; a++) {
              for (int b = a; b < 9; b++) {
                const double s =
                    gu_rr * W[a][0] * W[b][0] + gu_tt * W[a][1] * W[b][1] +
                    gu_pp * W[a][2] * W[b][2] +
                    gu_rp * (W[a][0] * W[b][2] + W[a][2] * W[b][0]);
                M1l[a][b] += w1 * s;
                M1al[a][b] += w1 * al * s;
              }
            }
            for (int a = 0; a < 5; a++) {
              for (int b = a; b < 5; b++) {
                const double s =
                    g_rr * Bd[a][0] * Bd[b][0] + g_tt * Bd[a][1] * Bd[b][1] +
                    g_pp * Bd[a][2] * Bd[b][2] +
                    g_rp * (Bd[a][0] * Bd[b][2] + Bd[a][2] * Bd[b][0]);
                M2al[a][b] += w2f * s;
              }
            }
            // Shift coupling C1[e,f] = ⟨β×B_f, W_e⟩₁.  With β = β^r ∂_r:
            // (β×B)_θ = −β^r Bd^φ, (β×B)_φ = +β^r Bd^θ, so the raised
            // pairing against the covariant W gives
            //   √γ β^r [ (γ^{rφ}W_r + γ^{φφ}W_φ)·Bd^θ − γ^{θθ}W_θ·Bd^φ ].
            const double br = (double)met.beta_r(r, sth, cth);
            if (br != 0.0) {
              const double wc = jac * sqg * br;
              for (int a = 0; a < 9; a++) {
                for (int b = 0; b < 5; b++) {
                  C1l[a][b] += wc * ((gu_rp * W[a][0] + gu_pp * W[a][2]) *
                                         Bd[b][1] -
                                     gu_tt * W[a][1] * Bd[b][2]);
                }
              }
            }
          }
        }

        prism_edges(k, t, eidx);
        prism_faces(k, t, fidx);
        for (int a = 0; a < 9; a++) {
          for (int b = a; b < 9; b++) {
            scatter(m1_rp, m1_ci, v1, eidx[a], eidx[b], M1l[a][b]);
            scatter(m1_rp, m1_ci, v1a, eidx[a], eidx[b], M1al[a][b]);
            if (b != a) {
              scatter(m1_rp, m1_ci, v1, eidx[b], eidx[a], M1l[a][b]);
              scatter(m1_rp, m1_ci, v1a, eidx[b], eidx[a], M1al[a][b]);
            }
          }
        }
        for (int a = 0; a < 5; a++) {
          for (int b = a; b < 5; b++) {
            scatter(m2_rp, m2_ci, v2a, fidx[a], fidx[b], M2al[a][b]);
            if (b != a)
              scatter(m2_rp, m2_ci, v2a, fidx[b], fidx[a], M2al[a][b]);
          }
        }
        for (int a = 0; a < 9; a++) {
          for (int b = 0; b < 5; b++) {
            scatter(c1_rp, c1_ci, vc1, eidx[a], fidx[b], C1l[a][b]);
            scatter(c1t_rp, c1t_ci, vc1t, fidx[b], eidx[a], C1l[a][b]);
          }
        }
      }
    }
  }

  // ---- store, Jacobi diagonal, sanity checks ----
  alloc_s(m1_val, nnz1);
  alloc_s(m1a_val, nnz1);
  alloc_s(m2a_val, nnz2);
  alloc_s(c1_val, nnzc);
  alloc_s(c1t_val, nnzct);
  alloc_s(m1_jacobi, Ne);
  for (size_t i = 0; i < nnz1; i++) {
    m1_val[i] = (Scalar)v1[i];
    m1a_val[i] = (Scalar)v1a[i];
  }
  for (size_t i = 0; i < nnz2; i++) m2a_val[i] = (Scalar)v2a[i];
  for (size_t i = 0; i < nnzc; i++) c1_val[i] = (Scalar)vc1[i];
  for (size_t i = 0; i < nnzct; i++) c1t_val[i] = (Scalar)vc1t[i];

  double min_diag1 = 1e300, min_diag2 = 1e300;
  for (int e = 0; e < Ne; e++) {
    double d = 0.0;
    for (int j = m1_rp[e]; j < m1_rp[e + 1]; j++) {
      if (m1_ci[j] == e) d = v1[j];
    }
    min_diag1 = std::min(min_diag1, d);
    m1_jacobi[e] = (Scalar)(d > 0 ? 1.0 / d : 0.0);
  }
  for (int f = 0; f < Nf; f++) {
    double d = 0.0;
    for (int j = m2_rp[f]; j < m2_rp[f + 1]; j++) {
      if (m2_ci[j] == f) d = v2a[j];
    }
    min_diag2 = std::min(min_diag2, d);
  }

  // Symmetry defect (max |M − Mᵀ| / max |M|): the pattern is symmetric by
  // construction, so look up the transposed entry per nonzero.
  auto sym_defect = [&](const int* rp, const int* ci,
                        const std::vector<double>& v, int n) {
    double dmax = 0.0, vmax = 0.0;
    for (int row = 0; row < n; row++) {
      for (int j = rp[row]; j < rp[row + 1]; j++) {
        int col = ci[j];
        vmax = std::max(vmax, std::abs(v[j]));
        if (col < row) continue;
        const int lo = rp[col], hi = rp[col + 1];
        const int* it = std::lower_bound(ci + lo, ci + hi, row);
        double vt = (it != ci + hi && *it == row) ? v[lo + int(it - (ci + lo))]
                                                  : 0.0;
        dmax = std::max(dmax, std::abs(v[j] - vt));
      }
    }
    return vmax > 0 ? dmax / vmax : 0.0;
  };
  double sd1 = sym_defect(m1_rp, m1_ci, v1, Ne);
  double sd1a = sym_defect(m1_rp, m1_ci, v1a, Ne);
  double sd2 = sym_defect(m2_rp, m2_ci, v2a, Nf);

  timer::show_duration_since_stamp("whitney_hodge build", "ms",
                                   "whitney_build");
  Logger::print_info(
      "whitney_hodge: sym defect M1 {:.2e}, M1a {:.2e}, M2a {:.2e}; "
      "min diag M1 {:.3e}, M2a {:.3e}",
      sd1, sd1a, sd2, min_diag1, min_diag2);
  if (sd1 > 1e-12 || sd1a > 1e-12 || sd2 > 1e-12) {
    Logger::err(
        "whitney_hodge: mass matrices are NOT symmetric — energy stability "
        "is lost.  Refusing to mark the operator ready.");
    return;
  }
  if (min_diag1 <= 0 || min_diag2 <= 0) {
    Logger::err("whitney_hodge: non-positive diagonal — matrix is not SPD.");
    return;
  }

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  copy_to_device();
#endif
  m_ready = true;
}

void prismatic_whitney_hodge::dump(const std::string& path) const {
  // The solver may dump before the data exporter has created the output
  // directory (init order = registration order).
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  auto file = hdf_create(path);
  file.write(m1_row_ptr.host_ptr(), m1_row_ptr.size(), "m1_row_ptr");
  file.write(m1_col_idx.host_ptr(), m1_col_idx.size(), "m1_col_idx");
  file.write(m1_val.host_ptr(), m1_val.size(), "m1_val");
  file.write(m1a_val.host_ptr(), m1a_val.size(), "m1a_val");
  file.write(m1_jacobi.host_ptr(), m1_jacobi.size(), "m1_jacobi");
  file.write(m2_row_ptr.host_ptr(), m2_row_ptr.size(), "m2_row_ptr");
  file.write(m2_col_idx.host_ptr(), m2_col_idx.size(), "m2_col_idx");
  file.write(m2a_val.host_ptr(), m2a_val.size(), "m2a_val");
  file.write(c1_row_ptr.host_ptr(), c1_row_ptr.size(), "c1_row_ptr");
  file.write(c1_col_idx.host_ptr(), c1_col_idx.size(), "c1_col_idx");
  file.write(c1_val.host_ptr(), c1_val.size(), "c1_val");
  file.write(c1t_row_ptr.host_ptr(), c1t_row_ptr.size(), "c1t_row_ptr");
  file.write(c1t_col_idx.host_ptr(), c1t_col_idx.size(), "c1t_col_idx");
  file.write(c1t_val.host_ptr(), c1t_val.size(), "c1t_val");
  file.write(m_N_edges, "N_edges");
  file.write(m_N_faces, "N_faces");
  Logger::print_info("whitney_hodge: dumped CSR operators to {}", path);
}

prismatic_whitney_hodge_ptrs prismatic_whitney_hodge::host_ptrs() const {
  prismatic_whitney_hodge_ptrs p{};
  p.m1_row_ptr = m1_row_ptr.host_ptr();
  p.m1_col_idx = m1_col_idx.host_ptr();
  p.m1_val = m1_val.host_ptr();
  p.m1a_val = m1a_val.host_ptr();
  p.m1_jacobi = m1_jacobi.host_ptr();
  p.m2_row_ptr = m2_row_ptr.host_ptr();
  p.m2_col_idx = m2_col_idx.host_ptr();
  p.m2a_val = m2a_val.host_ptr();
  p.c1_row_ptr = c1_row_ptr.host_ptr();
  p.c1_col_idx = c1_col_idx.host_ptr();
  p.c1_val = c1_val.host_ptr();
  p.c1t_row_ptr = c1t_row_ptr.host_ptr();
  p.c1t_col_idx = c1t_col_idx.host_ptr();
  p.c1t_val = c1t_val.host_ptr();
  p.N_edges = m_N_edges;
  p.N_faces = m_N_faces;
  return p;
}

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
prismatic_whitney_hodge_ptrs prismatic_whitney_hodge::dev_ptrs() const {
  prismatic_whitney_hodge_ptrs p{};
  p.m1_row_ptr = m1_row_ptr.dev_ptr();
  p.m1_col_idx = m1_col_idx.dev_ptr();
  p.m1_val = m1_val.dev_ptr();
  p.m1a_val = m1a_val.dev_ptr();
  p.m1_jacobi = m1_jacobi.dev_ptr();
  p.m2_row_ptr = m2_row_ptr.dev_ptr();
  p.m2_col_idx = m2_col_idx.dev_ptr();
  p.m2a_val = m2a_val.dev_ptr();
  p.c1_row_ptr = c1_row_ptr.dev_ptr();
  p.c1_col_idx = c1_col_idx.dev_ptr();
  p.c1_val = c1_val.dev_ptr();
  p.c1t_row_ptr = c1t_row_ptr.dev_ptr();
  p.c1t_col_idx = c1t_col_idx.dev_ptr();
  p.c1t_val = c1t_val.dev_ptr();
  p.N_edges = m_N_edges;
  p.N_faces = m_N_faces;
  return p;
}

void prismatic_whitney_hodge::copy_to_device() {
  m1_row_ptr.copy_to_device();
  m1_col_idx.copy_to_device();
  m1_val.copy_to_device();
  m1a_val.copy_to_device();
  m1_jacobi.copy_to_device();
  m2_row_ptr.copy_to_device();
  m2_col_idx.copy_to_device();
  m2a_val.copy_to_device();
  c1_row_ptr.copy_to_device();
  c1_col_idx.copy_to_device();
  c1_val.copy_to_device();
  c1t_row_ptr.copy_to_device();
  c1t_col_idx.copy_to_device();
  c1t_val.copy_to_device();
}
#endif

template void prismatic_whitney_hodge::build<flat_spherical_metric>(
    const prismatic_mesh_metric&, const flat_spherical_metric&);
template void prismatic_whitney_hodge::build<ks_spherical_metric>(
    const prismatic_mesh_metric&, const ks_spherical_metric&);

}  // namespace Aperture
