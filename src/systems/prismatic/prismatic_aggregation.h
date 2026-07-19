#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

namespace Aperture {

// =========================================================================
// Phase 7D (F9) — TRUE coarse-cochain aggregation maps.
//
// Downsampled output = the chain-map restriction of the fine DEC field
// to a level-(L−j) icosphere × every-R-th shell:
//   coarse sphere edge = signed sum of its 2^j fine sub-edges,
//   coarse tri face    = sum of its 4^j fine children,
//   coarse v-edge      = R stacked fine v-edges,
//   coarse rect face   = (signed 2^j chain) × R fine layers,
//   coarse vertex      = hat partition-of-unity restriction (gnomonic
//                        barycentric in the parent coarse tri × radial
//                        tent) of the fine vertex moments.
// Aggregation is the transpose of refinement, so it COMMUTES with the
// discrete d: the dump is a bona fide level-(L−j) DEC field (coarse
// Faraday/Gauss consistency is exact — pinned by unit test on integer
// cochains).
//
// The coarse topology is DERIVED from the fine tri_verts alone: the
// subdivision pushes each parent's children contiguously with the
// corner children first, so
//   parent verts    a = tv[4T][0], b = tv[4T+1][1], c = tv[4T+2][2]
//   edge midpoints  m_ab = tv[4T][1], m_ac = tv[4T][2], m_bc = tv[4T+1][2]
// recursively down to level L−j.  Coarse edges are created in the same
// canonical order as a direct level-(L−j) build (per tri: (v0,v1),
// (v1,v2), (v0,v2), first-come) — verified equal to an independent
// build_sphere_only by unit test.
// =========================================================================
struct prismatic_coarse_aggregator {
  int j = 0;   // angular levels
  int R = 1;   // radial shell stride
  int L_out = 0;
  int N_r_c = 0;                      // coarse layers = N_r / R
  int n_tri_c = 0, n_edge_c = 0, n_vert_c = 0;

  // Coarse sphere topology (vertex ids are FINE ids — level prefixes).
  std::vector<int> tri_verts_c;                  // [n_tri_c * 3]
  std::vector<std::array<int, 2>> edges_c;       // canonical (a < b)

  // Coarse edge -> signed fine sub-edge chain (CSR).
  std::vector<int> chain_off;                    // [n_edge_c + 1]
  std::vector<int> chain_edge;
  std::vector<int8_t> chain_sign;

  // Fine sphere vertex -> 3 (coarse vertex, weight) hat contributions.
  std::vector<std::array<int, 3>> vw_idx;        // [N_vert_s fine]
  std::vector<std::array<double, 3>> vw_w;

  const prismatic_mesh* mesh = nullptr;

  void build(const prismatic_mesh& m, int j_in, int R_in) {
    mesh = &m;
    j = j_in;
    R = R_in;
    if (j < 0 || j > m.m_L) {
      throw std::invalid_argument("aggregator: 0 <= j <= L required");
    }
    if (R < 1 || m.m_N_r % R != 0) {
      throw std::invalid_argument(
          "aggregator: radial stride must divide N_r");
    }
    L_out = m.m_L - j;
    N_r_c = m.m_N_r / R;

    // ---- Derive tri tables + per-level edge-midpoint maps ----
    const int NT_f = m.m_N_tri;
    std::vector<std::vector<int>> tv(j + 1);
    tv[0].assign(m.tri_verts.host_ptr(), m.tri_verts.host_ptr() + NT_f * 3);
    // mid[lev]: midpoint map at the level with tv[lev+1] tris (i.e. the
    // parents of tv[lev]); key = canonical parent-edge endpoints.
    std::vector<std::map<std::pair<int, int>, int>> mid(j);
    for (int lev = 0; lev < j; ++lev) {
      const auto& f = tv[lev];
      const int nt = int(f.size() / 3) / 4;
      auto& c = tv[lev + 1];
      c.resize(nt * 3);
      for (int T = 0; T < nt; ++T) {
        const int a = f[(4 * T + 0) * 3 + 0];
        const int b = f[(4 * T + 1) * 3 + 1];
        const int cc = f[(4 * T + 2) * 3 + 2];
        const int m_ab = f[(4 * T + 0) * 3 + 1];
        const int m_ac = f[(4 * T + 0) * 3 + 2];
        const int m_bc = f[(4 * T + 1) * 3 + 2];
        c[T * 3 + 0] = a;
        c[T * 3 + 1] = b;
        c[T * 3 + 2] = cc;
        auto key = [](int u, int w) {
          return std::make_pair(u < w ? u : w, u < w ? w : u);
        };
        mid[lev][key(a, b)] = m_ab;
        mid[lev][key(a, cc)] = m_ac;
        mid[lev][key(b, cc)] = m_bc;
      }
    }
    tri_verts_c = tv[j];
    n_tri_c = int(tri_verts_c.size() / 3);

    // ---- Coarse edges (canonical creation order) ----
    std::map<std::pair<int, int>, int> edge_map_c;
    edges_c.clear();
    auto get_or_create = [&](int u, int w) {
      auto key = std::make_pair(u < w ? u : w, u < w ? w : u);
      auto it = edge_map_c.find(key);
      if (it != edge_map_c.end()) return it->second;
      int idx = int(edges_c.size());
      edges_c.push_back({key.first, key.second});
      edge_map_c[key] = idx;
      return idx;
    };
    for (int T = 0; T < n_tri_c; ++T) {
      const int v0 = tri_verts_c[T * 3 + 0];
      const int v1 = tri_verts_c[T * 3 + 1];
      const int v2 = tri_verts_c[T * 3 + 2];
      get_or_create(v0, v1);
      get_or_create(v1, v2);
      get_or_create(v0, v2);
    }
    n_edge_c = int(edges_c.size());

    // Coarse vertex ids are a PREFIX of the fine ids.
    n_vert_c = 0;
    for (auto v : tri_verts_c) n_vert_c = std::max(n_vert_c, v + 1);

    // ---- Fine edge lookup + chains ----
    std::map<std::pair<int, int>, int> fine_edge_map;
    const int* fe0 = m.sphere_edge_v0.host_ptr();
    const int* fe1 = m.sphere_edge_v1.host_ptr();
    for (int e = 0; e < m.m_N_edge_s; ++e) {
      fine_edge_map[{fe0[e], fe1[e]}] = e;
    }
    chain_off.assign(n_edge_c + 1, 0);
    chain_edge.clear();
    chain_sign.clear();
    // Recursive expansion: traversal u -> w; at the finest level look
    // the edge up (sign + when u is the canonical v0).
    std::function<void(int, int, int)> expand = [&](int u, int w, int lev) {
      if (lev == j) {
        auto it = fine_edge_map.find(
            {u < w ? u : w, u < w ? w : u});
        if (it == fine_edge_map.end()) {
          throw std::runtime_error("aggregator: fine sub-edge not found");
        }
        chain_edge.push_back(it->second);
        chain_sign.push_back(u < w ? int8_t(+1) : int8_t(-1));
        return;
      }
      // mid[] is indexed with 0 = finest parents; level `lev` counted
      // from the coarse side: the midpoint map for coarse level
      // (L_out + lev) parents is mid[j - 1 - lev].
      auto& mm = mid[j - 1 - lev];
      auto it = mm.find({u < w ? u : w, u < w ? w : u});
      if (it == mm.end()) {
        throw std::runtime_error("aggregator: missing midpoint");
      }
      expand(u, it->second, lev + 1);
      expand(it->second, w, lev + 1);
    };
    for (int E = 0; E < n_edge_c; ++E) {
      expand(edges_c[E][0], edges_c[E][1], 0);
      chain_off[E + 1] = int(chain_edge.size());
    }

    // ---- Fine-vertex hat weights ----
    const int NVs = m.m_N_vert_s;
    vw_idx.assign(NVs, {0, 0, 0});
    vw_w.assign(NVs, {0.0, 0.0, 0.0});
    const Scalar* svx = m.sphere_vx.host_ptr();
    const Scalar* svy = m.sphere_vy.host_ptr();
    const Scalar* svz = m.sphere_vz.host_ptr();
    for (int s = 0; s < NVs; ++s) {
      // Parent coarse tri via the first fine fan tri.
      const int t_f = m.sph_vert_tris[m.sph_vert_tri_offset[s]];
      const int T = t_f >> (2 * j);
      const int A = tri_verts_c[T * 3 + 0];
      const int B = tri_verts_c[T * 3 + 1];
      const int C = tri_verts_c[T * 3 + 2];
      // Gnomonic barycentric of s in (A, B, C), normalized to sum 1.
      double p[3] = {svx[s], svy[s], svz[s]};
      double va[3] = {svx[A], svy[A], svz[A]};
      double vb[3] = {svx[B], svy[B], svz[B]};
      double vc[3] = {svx[C], svy[C], svz[C]};
      auto det3 = [](const double* x, const double* y, const double* z) {
        return x[0] * (y[1] * z[2] - y[2] * z[1]) -
               x[1] * (y[0] * z[2] - y[2] * z[0]) +
               x[2] * (y[0] * z[1] - y[1] * z[0]);
      };
      double c0 = det3(p, vb, vc);
      double c1 = det3(va, p, vc);
      double c2 = det3(va, vb, p);
      double sum = c0 + c1 + c2;
      vw_idx[s] = {A, B, C};
      vw_w[s] = {c0 / sum, c1 / sum, c2 / sum};
    }
  }

  // ---- Coarse sizes -----------------------------------------------------
  int n_h_c() const { return (N_r_c + 1) * n_edge_c; }
  int n_v_c() const { return N_r_c * n_vert_c; }
  int n_trif_c() const { return (N_r_c + 1) * n_tri_c; }
  int n_rect_c() const { return N_r_c * n_edge_c; }
  int n_vertc_c() const { return (N_r_c + 1) * n_vert_c; }

  // =======================================================================
  // Aggregation.  `val(g)` returns the FINE global cochain value for a
  // contribution this rank owns and 0.0 when `owned(g)` is false — the
  // caller supplies both (identity/always-true single-rank; partition
  // ownership + local layout lookup distributed).  Output arrays are
  // ACCUMULATED into (caller zeroes; distributed callers MPI_SUM-reduce
  // the partials, which needs no slab alignment at all).
  // =======================================================================
  template <typename F>
  void agg_h_edges(F&& val, double* out) const {
    const int NEf = mesh->m_N_edge_s;
    for (int K = 0; K <= N_r_c; ++K) {
      const int kf = K * R;
      for (int E = 0; E < n_edge_c; ++E) {
        double s = 0;
        for (int c = chain_off[E]; c < chain_off[E + 1]; ++c) {
          s += double(chain_sign[c]) * val(gidx_t(kf) * NEf + chain_edge[c]);
        }
        out[K * n_edge_c + E] += s;
      }
    }
  }

  template <typename F>
  void agg_v_edges(F&& val, double* out) const {
    const int NVf = mesh->m_N_vert_s;
    for (int K = 0; K < N_r_c; ++K) {
      for (int S = 0; S < n_vert_c; ++S) {
        double s = 0;
        for (int i = 0; i < R; ++i) {
          s += val(gidx_t(K * R + i) * NVf + S);
        }
        out[K * n_vert_c + S] += s;
      }
    }
  }

  template <typename F>
  void agg_tri_faces(F&& val, double* out) const {
    const int NTf = mesh->m_N_tri;
    const int nchild = 1 << (2 * j);
    for (int K = 0; K <= N_r_c; ++K) {
      const int kf = K * R;
      for (int T = 0; T < n_tri_c; ++T) {
        double s = 0;
        for (int c = 0; c < nchild; ++c) {
          s += val(gidx_t(kf) * NTf + T * nchild + c);
        }
        out[K * n_tri_c + T] += s;
      }
    }
  }

  template <typename F>
  void agg_rect_faces(F&& val, double* out) const {
    const int NEf = mesh->m_N_edge_s;
    for (int K = 0; K < N_r_c; ++K) {
      for (int E = 0; E < n_edge_c; ++E) {
        double s = 0;
        for (int c = chain_off[E]; c < chain_off[E + 1]; ++c) {
          for (int i = 0; i < R; ++i) {
            s += double(chain_sign[c]) *
                 val(gidx_t(K * R + i) * NEf + chain_edge[c]);
          }
        }
        out[K * n_edge_c + E] += s;
      }
    }
  }

  // Vertex moments: 3D hat partition-of-unity (angular barycentric ×
  // radial tent).  Total charge is conserved (weights sum to 1).
  template <typename F>
  void agg_vertices(F&& val, double* out) const {
    const int NVf = mesh->m_N_vert_s;
    const int N_r_f = mesh->m_N_r;
    for (int k = 0; k <= N_r_f; ++k) {
      const int K0 = k / R;
      const double frac = double(k - K0 * R) / R;
      for (int s = 0; s < NVf; ++s) {
        const double v = val(gidx_t(k) * NVf + s);
        if (v == 0.0) continue;
        for (int c = 0; c < 3; ++c) {
          const double w = vw_w[s][c];
          const int S = vw_idx[s][c];
          out[K0 * n_vert_c + S] += v * w * (1.0 - frac);
          if (frac > 0.0) {
            out[(K0 + 1) * n_vert_c + S] += v * w * frac;
          }
        }
      }
    }
  }
};

}  // namespace Aperture
