// Phase 7D (F9) — coarse-cochain aggregation unit tests.
//
// 1. The DERIVED coarse topology (tri tables from the fine child
//    arrangement, edges in canonical creation order) equals an
//    independently built level-(L−j) sphere-only mesh.
// 2. The aggregation COMMUTES WITH THE DISCRETE d exactly: on
//    integer-valued fine cochains, the coarse curl of the aggregated E
//    equals the aggregation of the fine curls, bit-for-bit — the
//    property value sampling never had.  (Integer values make float
//    sums exact, isolating the topological identity from FP
//    reordering.)
// 3. The vertex hat restriction is a partition of unity (angular
//    weights sum to 1 per fine vertex; radial tents sum to 1 per fine
//    shell), so the total charge of aggregated moments is conserved.
#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_aggregation.h"
#include "systems/prismatic/prismatic_mesh.h"
#include <cmath>
#include <vector>

using namespace Aperture;

namespace {

constexpr int TL = 2;
constexpr int TN_r = 8;

// Deterministic small-integer cochain values (exact in float).
double int_val(int g, int salt) { return double(((g * 37 + salt) % 21) - 10); }

}  // namespace

TEST_CASE("aggregator: derived coarse topology equals a direct build",
          "[prismatic][aggregation]") {
  prismatic_mesh fine;
  fine.build_sphere_only(TL, TN_r, 1.0, 2.0);

  for (int j : {1, 2}) {
    prismatic_coarse_aggregator agg;
    agg.build(fine, j, 2);

    prismatic_mesh coarse;
    coarse.build_sphere_only(TL - j, TN_r, 1.0, 2.0);
    REQUIRE(agg.n_tri_c == coarse.m_N_tri);
    REQUIRE(agg.n_edge_c == coarse.m_N_edge_s);
    REQUIRE(agg.n_vert_c == coarse.m_N_vert_s);
    for (int t = 0; t < agg.n_tri_c * 3; ++t) {
      REQUIRE(agg.tri_verts_c[t] == coarse.tri_verts[t]);
    }
    for (int e = 0; e < agg.n_edge_c; ++e) {
      REQUIRE(agg.edges_c[e][0] == coarse.sphere_edge_v0[e]);
      REQUIRE(agg.edges_c[e][1] == coarse.sphere_edge_v1[e]);
    }
    // Chains: 2^j fine edges per coarse edge, all distinct.
    for (int e = 0; e < agg.n_edge_c; ++e) {
      REQUIRE(agg.chain_off[e + 1] - agg.chain_off[e] == (1 << j));
    }
  }
}

TEST_CASE("aggregator: chain map commutes with d exactly (integer "
          "cochains)",
          "[prismatic][aggregation]") {
  prismatic_mesh fine;
  fine.build(TL, TN_r, 1.0, 2.0);  // full build: fine d1 needed

  const int j = 1, R = 2;
  prismatic_coarse_aggregator agg;
  agg.build(fine, j, R);
  prismatic_mesh coarse;
  coarse.build_sphere_only(TL - j, TN_r / R, 1.0, 2.0);

  // Integer fine E cochain (combined [h|v]).
  const int N_h_f = (fine.m_N_r + 1) * fine.m_N_edge_s;
  std::vector<Scalar> E(fine.m_N_edges);
  for (int e = 0; e < fine.m_N_edges; ++e) E[e] = Scalar(int_val(e, 7));

  // Fine curls per fine face (d1 E).
  std::vector<double> curl_f(fine.m_N_faces, 0.0);
  const int* rp = fine.d1_row_ptr.host_ptr();
  const int* ci = fine.d1_col_idx.host_ptr();
  const Scalar* dv = fine.d1_val.host_ptr();
  for (int f = 0; f < fine.m_N_faces; ++f) {
    for (int q = rp[f]; q < rp[f + 1]; ++q) {
      curl_f[f] += double(dv[q]) * double(E[ci[q]]);
    }
  }

  // Aggregate E (h + v blocks) and the fine curls (tri + rect blocks).
  std::vector<double> E_c(agg.n_h_c() + agg.n_v_c(), 0.0);
  agg.agg_h_edges([&](int g) { return double(E[g]); }, E_c.data());
  agg.agg_v_edges([&](int g) { return double(E[N_h_f + g]); },
                  E_c.data() + agg.n_h_c());
  std::vector<double> curl_c(agg.n_trif_c() + agg.n_rect_c(), 0.0);
  const int N_trif_f = (fine.m_N_r + 1) * fine.m_N_tri;
  agg.agg_tri_faces([&](int g) { return curl_f[g]; }, curl_c.data());
  agg.agg_rect_faces([&](int g) { return curl_f[N_trif_f + g]; },
                     curl_c.data() + agg.n_trif_c());

  // Coarse curl of the aggregated E, using the independently built
  // coarse topology, must equal the aggregated fine curl EXACTLY.
  // Coarse tri faces:
  for (int K = 0; K <= agg.N_r_c; ++K) {
    for (int T = 0; T < agg.n_tri_c; ++T) {
      double c = 0;
      for (int jj = 0; jj < 3; ++jj) {
        const int e = coarse.tri_edges_s[T * 3 + jj];
        const int sgn = coarse.tri_edge_signs[T * 3 + jj];
        c += double(sgn) * E_c[K * agg.n_edge_c + e];
      }
      REQUIRE(c == curl_c[K * agg.n_tri_c + T]);
    }
  }
  // Coarse rect faces: bottom h +1, right v(b) +1, top h −1, left v(a) −1.
  for (int K = 0; K < agg.N_r_c; ++K) {
    for (int E_i = 0; E_i < agg.n_edge_c; ++E_i) {
      const int a = agg.edges_c[E_i][0], b = agg.edges_c[E_i][1];
      double c = E_c[K * agg.n_edge_c + E_i] -
                 E_c[(K + 1) * agg.n_edge_c + E_i] +
                 E_c[agg.n_h_c() + K * agg.n_vert_c + b] -
                 E_c[agg.n_h_c() + K * agg.n_vert_c + a];
      REQUIRE(c == curl_c[agg.n_trif_c() + K * agg.n_edge_c + E_i]);
    }
  }
}

TEST_CASE("aggregator: vertex hat restriction is a partition of unity "
          "and conserves total charge",
          "[prismatic][aggregation]") {
  prismatic_mesh fine;
  fine.build_sphere_only(TL, TN_r, 1.0, 2.0);
  prismatic_coarse_aggregator agg;
  agg.build(fine, 2, 2);

  for (int s = 0; s < fine.m_N_vert_s; ++s) {
    const double w = agg.vw_w[s][0] + agg.vw_w[s][1] + agg.vw_w[s][2];
    REQUIRE(std::abs(w - 1.0) < 1e-12);
    for (int c = 0; c < 3; ++c) {
      REQUIRE(agg.vw_idx[s][c] < agg.n_vert_c);
    }
  }

  // Total charge: random-ish fine rho aggregates to the same total.
  const int NVf = fine.m_N_vert_s;
  std::vector<double> rho((fine.m_N_r + 1) * NVf);
  double total_f = 0;
  for (size_t i = 0; i < rho.size(); ++i) {
    rho[i] = std::sin(0.037 * double(i)) + 0.2;
    total_f += rho[i];
  }
  std::vector<double> rho_c(agg.n_vertc_c(), 0.0);
  agg.agg_vertices([&](int g) { return rho[g]; }, rho_c.data());
  double total_c = 0;
  for (double v : rho_c) total_c += v;
  REQUIRE(total_c == Catch::Approx(total_f).epsilon(1e-12));
}
