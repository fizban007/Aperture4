// Vacuum test of the ADJOINT-PAIRED frame-drag scheme:
//   Faraday: dB/dt = -d1 (E + W (Bd + B0))                    [unchanged]
//   Ampere : dE/dt = h1inv d1t (h2 Bd + F) ,  F = W^T h1inv^-1 E
// which restores the dropped beta x E term as the EXACT energy adjoint of
// the existing W under the diagonal Hodge.  Semi-discretely
//   U_full = 1/2 E' h1inv^-1 E + 1/2 Bd' h2 Bd + E' h1inv^-1 W Bd
// is conserved identically (every spurious quadratic cancels), so the
// grid-scale instability of the one-sided scheme cannot exist.
//
// argv: L N_r r_max n_steps dt noise_mode adj
//   noise_mode=1: homogeneous system + 1e-6 noise (linear stability test)
//   adj=0: original one-sided scheme; adj=1: explicit adjoint term;
//   adj>=2: adj Picard sweeps toward the midpoint (time-centred) term.
#define private public
#include "systems/prismatic/dec_solver_dist.h"
#undef private
#include "systems/prismatic/dec_solver_geometry.hpp"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Aperture;
using policy = prismatic_exec_policy_host;
using core_t = dec_solver_dist<policy>;

int main(int argc, char** argv) {
  const int L = (argc > 1) ? atoi(argv[1]) : 4;
  const int N_r = (argc > 2) ? atoi(argv[2]) : 32;
  const double r_max = (argc > 3) ? atof(argv[3]) : 3.0;
  const int n_steps = (argc > 4) ? atoi(argv[4]) : 20000;
  const double dt_in = (argc > 5) ? atof(argv[5]) : 0.0;
  const bool noise_mode = (argc > 6) && atoi(argv[6]) != 0;
  const int adj = (argc > 7) ? atoi(argv[7]) : 2;
  const double lapse_c = (argc > 8) ? atof(argv[8]) : 0.0;
  const Scalar Omega = 0.25, Bp = 1000.0;
  const Scalar w0 = 0.05;
  const int lt_p = 3;

  prismatic_mesh mesh;
  mesh.build(L, N_r, 1.0, Scalar(r_max));
  auto topo = icosphere_topology::build_from_mesh(mesh);
  auto part = prismatic_partition::single_rank(L, N_r);
  part.set_topology(&topo);
  auto mp_b = prismatic_mesh_partition::build(part, topo);
  core_t core;
  core.build(mesh, mp_b);
  core.build_frame_drag(w0, Scalar(1.0), lt_p);
  if (lapse_c > 0) core.build_lapse(Scalar(lapse_c), Scalar(1.0));

  const int ne = core.n_edges_local(), nfl = core.n_faces_local();
  const int es = core.e_split(), bs = core.b_split();
  auto lp = core.get_lp(exec_tags::host{});
  auto mp = mesh.host_ptrs();

  // ---- Transposed W blocks (face-major CSR), from the solver's own
  // d1t sparsity + fd values ------------------------------------------
  const Scalar* wht = core.m_fd_h_tri_val.host_ptr();
  const Scalar* whr = core.m_fd_h_rect_val.host_ptr();
  const Scalar* wvr = core.m_fd_v_rect_val.host_ptr();
  const int n_tri = lp.n_owned_tri, n_rect = lp.n_owned_rect;
  // tri faces <- h edges
  std::vector<int> tt_row(n_tri + 1, 0), tt_col;
  std::vector<Scalar> tt_val;
  // rect faces <- edges (h edges as-is; v edges stored as n_he + e)
  std::vector<int> tr_row(n_rect + 1, 0), tr_col;
  std::vector<Scalar> tr_val;
  {
    for (int e = 0; e < lp.n_owned_he; e++) {
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++)
        tt_row[lp.d1t_h_tri_col[j] + 1]++;
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++)
        tr_row[lp.d1t_h_rect_col[j] + 1]++;
    }
    for (int e = 0; e < lp.n_owned_ve; e++)
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++)
        tr_row[lp.d1t_v_rect_col[j] + 1]++;
    for (int f = 0; f < n_tri; f++) tt_row[f + 1] += tt_row[f];
    for (int f = 0; f < n_rect; f++) tr_row[f + 1] += tr_row[f];
    tt_col.resize(tt_row[n_tri]); tt_val.resize(tt_row[n_tri]);
    tr_col.resize(tr_row[n_rect]); tr_val.resize(tr_row[n_rect]);
    std::vector<int> pt(tt_row.begin(), tt_row.end() - 1);
    std::vector<int> pr(tr_row.begin(), tr_row.end() - 1);
    for (int e = 0; e < lp.n_owned_he; e++) {
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
        int f = lp.d1t_h_tri_col[j];
        tt_col[pt[f]] = e; tt_val[pt[f]] = wht[j]; pt[f]++;
      }
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
        int f = lp.d1t_h_rect_col[j];
        tr_col[pr[f]] = e; tr_val[pr[f]] = whr[j]; pr[f]++;
      }
    }
    for (int e = 0; e < lp.n_owned_ve; e++)
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
        int f = lp.d1t_v_rect_col[j];
        tr_col[pr[f]] = lp.n_owned_he + e; tr_val[pr[f]] = wvr[j]; pr[f]++;
      }
  }

  buffer<Scalar> E, B, B0, J, Eeff;
  for (auto* b : {&E, &J, &Eeff}) { b->set_memtype(MemType::host_only); b->resize(ne); }
  for (auto* b : {&B, &B0}) { b->set_memtype(MemType::host_only); b->resize(nfl); }
  J.assign(0);
  B.assign(0);
  B0.assign(0);
  if (!noise_mode) core.fill_dipole_B(B0, Scalar(0), Scalar(0), Bp);

  auto corot = [&](Scalar x, Scalar y, Scalar z, Scalar& ex, Scalar& ey,
                   Scalar& ez) {
    Scalar bx, by, bz;
    dipole_B_impl(x, y, z, Scalar(0), Scalar(0), Bp, bx, by, bz);
    Scalar r = std::sqrt(x * x + y * y + z * z);
    Scalar om = Omega - frame_drag_omega(r, w0, Scalar(1.0), lt_p);
    om /= gr_lapse(r, Scalar(lapse_c), Scalar(1.0));
    Scalar vx = -om * y, vy = om * x;
    ex = -(vy * bz);
    ey = vx * bz;
    ez = -(vx * by - vy * bx);
  };
  const int NQ = 20;
  if (noise_mode) {
    srand(12345);
    for (int e = 0; e < ne; e++)
      E[e] = Scalar(1e-6) * (Scalar(rand()) / RAND_MAX - Scalar(0.5));
    for (int f = 0; f < nfl; f++)
      B[f] = Scalar(1e-6) * (Scalar(rand()) / RAND_MAX - Scalar(0.5));
  } else {
    for (int e = 0; e < lp.n_owned_he; e++) {
      gidx_t g = lp.h_edge_l2g[e];
      gidx_t v0, v1;
      h_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, bx2, by2, bz2;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx2, by2, bz2);
      double acc = 0;
      for (int i = 0; i < NQ; i++) {
        Scalar t = (i + 0.5) / NQ;
        Scalar x, y, z, dlx, dly, dlz;
        h_edge_sphere_sample(r0, ax, ay, az, bx2, by2, bz2, t, x, y, z, dlx,
                             dly, dlz);
        Scalar ex, ey, ez;
        corot(x, y, z, ex, ey, ez);
        acc += (ex * dlx + ey * dly + ez * dlz) / NQ;
      }
      E[e] = Scalar(acc);
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      gidx_t g = lp.v_edge_l2g[e];
      gidx_t v0, v1;
      v_edge_vertex_ids(mp, g, v0, v1);
      Scalar r0, ax, ay, az, r1, bx2, by2, bz2;
      vertex_unit(mp, v0, r0, ax, ay, az);
      vertex_unit(mp, v1, r1, bx2, by2, bz2);
      double acc = 0;
      for (int i = 0; i < NQ; i++) {
        Scalar t = (i + 0.5) / NQ;
        Scalar rt = (Scalar(1) - t) * r0 + t * r1;
        Scalar x = rt * ax, y = rt * ay, z = rt * az;
        Scalar dr = r1 - r0;
        Scalar ex, ey, ez;
        corot(x, y, z, ex, ey, ez);
        acc += (ex * ax + ey * ay + ez * az) * dr / NQ;
      }
      E[es + e] = Scalar(acc);
    }
  }
  std::vector<Scalar> E_mt(ne, Scalar(0));
  if (!noise_mode)
    for (int e = 0; e < ne; e++) E_mt[e] = E[e];

  double dt = dt_in;
  if (dt <= 0) dt = 0.25 * std::sqrt(4.0 * M_PI / mp.N_vert_s);
  std::printf("# L=%d N_r=%d r_max=%g dt=%g steps=%d noise=%d adj=%d lapse=%g\n",
              L, N_r, r_max, dt, n_steps, int(noise_mode), adj, lapse_c);

  dec_inner_bc_params par;
  par.Bp = noise_mode ? Scalar(0) : Bp;
  par.Omega = noise_mode ? Scalar(0) : Omega;
  par.obliquity = 0;
  par.use_deutsch = false;
  par.overwrite_b = true;
  par.omega_lt0 = w0;
  par.lt_r_star = 1.0;
  par.lt_p = lt_p;
  par.lapse_compactness = Scalar(lapse_c);

  const int NEs = mp.N_edge_s;
  auto energy = [&](double& U, double& Ufull) {
    U = 0;
    for (int e = 0; e < lp.n_owned_he; e++)
      U += 0.5 * double(E[e]) * double(E[e]) / lp.h_edge_hodge1_inv[e];
    for (int e = 0; e < lp.n_owned_ve; e++)
      U += 0.5 * double(E[es + e]) * double(E[es + e]) /
           lp.v_edge_hodge1_inv[e];
    for (int f = 0; f < n_tri; f++)
      U += 0.5 * lp.tri_face_hodge2[f] * double(B[f]) * double(B[f]);
    for (int f = 0; f < n_rect; f++)
      U += 0.5 * lp.rect_face_hodge2[f] * double(B[bs + f]) * double(B[bs + f]);
    // cross term E' h1inv^-1 W Bd
    Ufull = U;
    for (int e = 0; e < lp.n_owned_he; e++) {
      double WB = 0;
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++)
        WB += double(wht[j]) * double(B[lp.d1t_h_tri_col[j]]);
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++)
        WB += double(whr[j]) * double(B[bs + lp.d1t_h_rect_col[j]]);
      Ufull += double(E[e]) * WB / lp.h_edge_hodge1_inv[e];
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      double WB = 0;
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++)
        WB += double(wvr[j]) * double(B[bs + lp.d1t_v_rect_col[j]]);
      Ufull += double(E[es + e]) * WB / lp.v_edge_hodge1_inv[e];
    }
  };

  auto report = [&](int step) {
    double U, Uf;
    energy(U, Uf);
    std::printf("%7d", step);
    for (int k = 0; k <= 12; k++) {
      double s = 0;
      for (int e = k * NEs; e < (k + 1) * NEs; e++)
        s += std::abs(double(E[e]) - double(E_mt[e]));
      std::printf(" %10.4g", s / NEs);
    }
    std::printf("  U=%.6g Ufull=%.6g\n", U, Uf);
    std::fflush(stdout);
  };

  const int damp_len = std::max(6, N_r / 5);
  std::vector<Scalar> Eold(ne), Es(ne), Ftri(n_tri), Frect(n_rect);

  // Manual Ampere with the adjoint face term, from Eold -> E, F built
  // from Euse.
  auto ampere_adj = [&](const std::vector<Scalar>& Eo, const Scalar* Euse) {
    for (int e = 0; e < lp.n_owned_he; e++)
      Es[e] = Euse[e] / lp.h_edge_hodge1_inv[e];
    for (int e = 0; e < lp.n_owned_ve; e++)
      Es[lp.n_owned_he + e] = Euse[es + e] / lp.v_edge_hodge1_inv[e];
    for (int f = 0; f < n_tri; f++) {
      double a = 0;
      for (int j = tt_row[f]; j < tt_row[f + 1]; j++)
        a += double(tt_val[j]) * double(Es[tt_col[j]]);
      Ftri[f] = Scalar(a);
    }
    for (int f = 0; f < n_rect; f++) {
      double a = 0;
      for (int j = tr_row[f]; j < tr_row[f + 1]; j++)
        a += double(tr_val[j]) * double(Es[tr_col[j]]);
      Frect[f] = Scalar(a);
    }
    for (int e = 0; e < lp.n_owned_he; e++) {
      Scalar curl_H = 0;
      for (int j = lp.d1t_h_tri_row[e]; j < lp.d1t_h_tri_row[e + 1]; j++) {
        int f = lp.d1t_h_tri_col[j];
        curl_H += lp.d1t_h_tri_val[j] *
                  (lp.tri_face_hodge2[f] * B[f] + Ftri[f]);
      }
      for (int j = lp.d1t_h_rect_row[e]; j < lp.d1t_h_rect_row[e + 1]; j++) {
        int f = lp.d1t_h_rect_col[j];
        curl_H += lp.d1t_h_rect_val[j] *
                  (lp.rect_face_hodge2[f] * B[bs + f] + Frect[f]);
      }
      E[e] = Eo[e] + Scalar(dt) * lp.h_edge_hodge1_inv[e] * curl_H;
    }
    for (int e = 0; e < lp.n_owned_ve; e++) {
      Scalar curl_H = 0;
      for (int j = lp.d1t_v_rect_row[e]; j < lp.d1t_v_rect_row[e + 1]; j++) {
        int f = lp.d1t_v_rect_col[j];
        curl_H += lp.d1t_v_rect_val[j] *
                  (lp.rect_face_hodge2[f] * B[bs + f] + Frect[f]);
      }
      E[es + e] = Eo[es + e] +
                  Scalar(dt) * lp.v_edge_hodge1_inv[e] * curl_H;
    }
  };

  report(0);
  std::vector<Scalar> Emid(ne), Bold(nfl);
  buffer<Scalar> Bmid_b;
  Bmid_b.set_memtype(MemType::host_only);
  Bmid_b.resize(nfl);
  for (int step = 1; step <= n_steps; step++) {
    if (adj >= 2) {
      // Time-symmetric Faraday: W acts on the midpoint (B^{n-1/2}+B^{n+1/2})/2,
      // reached by Picard sweeps (the wave-part curl E is unchanged leapfrog).
      for (int f = 0; f < nfl; f++) Bold[f] = B[f];
      core.frame_drag_eff_E(E, B, B0, Eeff);
      core.faraday(Eeff, B, dt);          // pass 0: candidate B^{n+1/2}
      for (int it = 1; it < adj; it++) {
        for (int f = 0; f < nfl; f++)
          Bmid_b[f] = Scalar(0.5) * (Bold[f] + B[f]);
        core.frame_drag_eff_E(E, Bmid_b, B0, Eeff);
        for (int f = 0; f < nfl; f++) B[f] = Bold[f];
        core.faraday(Eeff, B, dt);
      }
    } else {
      core.frame_drag_eff_E(E, B, B0, Eeff);
      core.faraday(Eeff, B, dt);
    }
    if (adj == 0) {
      core.ampere(E, B, J, dt);
    } else {
      for (int e = 0; e < ne; e++) Eold[e] = E[e];
      // pass 0: F from Eold (explicit); passes >0: F from midpoint
      ampere_adj(Eold, Eold.data());
      for (int it = 1; it < adj; it++) {
        for (int e = 0; e < ne; e++)
          Emid[e] = Scalar(0.5) * (Eold[e] + E[e]);
        ampere_adj(Eold, Emid.data());
      }
    }
    core.apply_damping(E, B, dt, damp_len, 1.0, 3.0);
    double t = step * dt;
    core.apply_inner_bc(E, B, B0, par, t, t);
    if (step % 500 == 0) report(step);
    if (step % 500 == 0) {
      double m = 0;
      for (int e = 0; e < NEs; e++)
        m = std::max(m, std::abs(double(E[3 * NEs + e])));
      if (m > 1e6) { std::printf("# DIVERGED\n"); break; }
    }
  }
  return 0;
}
