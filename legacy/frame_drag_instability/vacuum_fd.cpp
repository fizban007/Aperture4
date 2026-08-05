// Vacuum leapfrog evolution of the fake-GR frame-drag scheme, single rank,
// host policy.  Initializes the analytic MT corotation state (E = MT corot,
// B = dipole via B0, Bdelta = 0), evolves faraday/ampere with the W term and
// the corotating inner BC, J = 0, and tracks the per-shell mean |E - E_MT|
// on h-edges.  A field-solver-only (numerical) instability grows here; a
// plasma-driven one cannot.
#include "systems/prismatic/dec_solver_dist.h"
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
  const double dt = (argc > 5) ? atof(argv[5]) : 0.0;
  const bool noise_mode = (argc > 6) && atoi(argv[6]) != 0;
  const Scalar Omega = 0.25, Bp = 1000.0;
  const Scalar w0 = 0.05;  // omega_lt0 = 0.4 * C * Omega, C = 0.5
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

  const int ne = core.n_edges_local(), nfl = core.n_faces_local();
  const int es = core.e_split();
  buffer<Scalar> E, B, B0, J, Eeff;
  for (auto* b : {&E, &J, &Eeff}) { b->set_memtype(MemType::host_only); b->resize(ne); }
  for (auto* b : {&B, &B0}) { b->set_memtype(MemType::host_only); b->resize(nfl); }
  J.assign(0);
  B.assign(0);
  B0.assign(0);
  if (!noise_mode) core.fill_dipole_B(B0, Scalar(0), Scalar(0), Bp);

  // Analytic MT corotation E on every edge (10-pt midpoint quadrature).
  auto lp = core.get_lp(exec_tags::host{});
  auto mp = mesh.host_ptrs();
  auto corot = [&](Scalar x, Scalar y, Scalar z, Scalar& ex, Scalar& ey,
                   Scalar& ez) {
    Scalar bx, by, bz;
    dipole_B_impl(x, y, z, Scalar(0), Scalar(0), Bp, bx, by, bz);
    Scalar r = std::sqrt(x * x + y * y + z * z);
    Scalar om = Omega - frame_drag_omega(r, w0, Scalar(1.0), lt_p);
    Scalar vx = -om * y, vy = om * x;
    ex = -(vy * bz);
    ey = vx * bz;
    ez = -(vx * by - vy * bx);
  };
  const int NQ = 20;
  if (noise_mode) {
    // homogeneous system: seed white noise, zero BC, watch for growing modes
    srand(12345);
    for (int e = 0; e < ne; e++)
      E[e] = Scalar(1e-6) * (Scalar(rand()) / RAND_MAX - Scalar(0.5));
    for (int f = 0; f < nfl; f++)
      B[f] = Scalar(1e-6) * (Scalar(rand()) / RAND_MAX - Scalar(0.5));
  } else
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
  if (!noise_mode)
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
  std::vector<Scalar> E_mt(ne, Scalar(0));
  if (!noise_mode)
    for (int e = 0; e < ne; e++) E_mt[e] = E[e];

  double dt_use = dt;
  if (dt_use <= 0) {
    // production ratio: dt = 0.28 * min angular edge arc
    double h_ang = std::sqrt(4.0 * M_PI / mp.N_vert_s);
    dt_use = 0.25 * h_ang;
  }
  std::printf("# L=%d N_r=%d r_max=%g dt=%g steps=%d\n", L, N_r, r_max,
              dt_use, n_steps);

  dec_inner_bc_params par;
  par.Bp = noise_mode ? Scalar(0) : Bp;
  par.Omega = noise_mode ? Scalar(0) : Omega;
  par.obliquity = 0;
  par.use_deutsch = false;
  par.overwrite_b = true;
  par.omega_lt0 = w0;
  par.lt_r_star = 1.0;
  par.lt_p = lt_p;
  par.lapse_compactness = 0;  // shift-only arm

  // shell edge ranges: h-edges are k*N_edge_s .. (k+1)*N_edge_s
  const int NEs = mp.N_edge_s;
  const int n_shells_rep = 12;
  auto report = [&](int step) {
    std::printf("%7d", step);
    for (int k = 0; k <= n_shells_rep; k++) {
      double s = 0;
      for (int e = k * NEs; e < (k + 1) * NEs; e++)
        s += std::abs(double(E[e]) - double(E_mt[e]));
      std::printf(" %10.4g", s / NEs);
    }
    std::printf("\n");
    std::fflush(stdout);
  };

  // damping on the outer shells (production: damping_length 44 of 204;
  // scale to N_r)
  const int damp_len = std::max(6, N_r / 5);

  report(0);
  for (int step = 1; step <= n_steps; step++) {
    core.frame_drag_eff_E(E, B, B0, Eeff);
    core.faraday(Eeff, B, dt_use);
    core.ampere(E, B, J, dt_use);
    core.apply_damping(E, B, dt_use, damp_len, 1.0, 3.0);
    double t = step * dt_use;
    core.apply_inner_bc(E, B, B0, par, t, t);
    if (step % 500 == 0) report(step);
    // divergence guard
    if (step % 500 == 0) {
      double m = 0;
      for (int e = 0; e < NEs; e++) m = std::max(m, std::abs(double(E[3 * NEs + e])));
      if (m > 1e6) { std::printf("# DIVERGED\n"); break; }
    }
  }
  return 0;
}
