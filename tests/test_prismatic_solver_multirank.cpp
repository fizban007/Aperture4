// Multi-process MPI test for the distributed DEC solver (4.1b B2).
//
// Run under mpirun with 20*K ranks, e.g.:
//   mpirun -n 20 ./test_prismatic_solver_multirank      (angular only)
//   mpirun -n 80 ./test_prismatic_solver_multirank      (combined 20 x 4)
//
// Every rank builds the global mesh, seeds the same deterministic
// synthetic cochains, and then runs BOTH
//   (a) a single-rank reference core (identity layout, no exchanges) —
//       bit-identical on every rank, and
//   (b) its own slice of the distributed run, with real MPI halo
//       exchanges through prismatic_halo_exchanger at the plan's sync
//       points.
// Scenario 1: explicit leapfrog + damping + rotating-dipole inner BC
// (exercises the l2g quadrature path).  Scenario 2: semi-implicit with
// the ghost refresh inside every Picard iteration + damping + PEC.
//
// Acceptance (PHASE_4_1B_PLAN.md): max relative difference on owned
// cells < 1e-4 across all ranks (measured: bit-exact — same arithmetic
// per owned cell, exchanges copy exact values).
//
// NOTE (single-GPU workstations): the solver core here runs the HOST
// exec policy, but prismatic_mesh::build still allocates host_device
// buffers in CUDA builds, so every MPI process creates a CUDA context
// (~300 MB).  Oversubscribing N ranks onto one GPU therefore needs
// ~N * 0.3 GB free device memory — 80 ranks wants ~25 GB.  On
// rank-per-GPU clusters (Frontier) this does not apply.

#include "systems/prismatic/dec_solver_dist.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_halo_exchanger.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <vector>

using namespace Aperture;

namespace {

using core_t = dec_solver_dist<prismatic_exec_policy_host>;
using exchanger_t = prismatic_halo_exchanger<prismatic_exec_policy_host>;

constexpr int TL = 2;
constexpr int TN_r = 8;
constexpr double TDT = 0.01;

struct fields {
  buffer<Scalar> E, B, J, B0;
  buffer<Scalar> tmpE, tmpB, dE, dB, dE2, dB2;

  void alloc(const core_t& c) {
    auto a = [](buffer<Scalar>& b, int n) {
      b.set_memtype(MemType::host_only);
      b.resize(n);
      b.assign(Scalar(0));
    };
    a(E, c.n_edges_local());
    a(J, c.n_edges_local());
    a(tmpE, c.n_edges_local());
    a(dE, c.n_edges_local());
    a(dE2, c.n_edges_local());
    a(B, c.n_faces_local());
    a(B0, c.n_faces_local());
    a(tmpB, c.n_faces_local());
    a(dB, c.n_faces_local());
    a(dB2, c.n_faces_local());
  }
  void seed(const core_t& c, const std::vector<Scalar>& Eg,
            const std::vector<Scalar>& Bg, const std::vector<Scalar>& Jg) {
    c.edge_from_global(Eg.data(), E);
    c.face_from_global(Bg.data(), B);
    c.edge_from_global(Jg.data(), J);
  }
};

// Scenario 1: explicit + damping + inner dipole BC.
void run_explicit(core_t& core, exchanger_t& ex, fields& f,
                  int n_steps) {
  dec_inner_bc_params par;
  par.Bp = 100.0;
  par.Omega = 0.25;
  par.obliquity = 0.3;
  double time = 0.0;
  for (int s = 0; s < n_steps; s++) {
    ex.exchange_edge(f.E, core.e_split());
    core.faraday(f.E, f.B, TDT);
    ex.exchange_face(f.B, core.b_split());
    core.ampere(f.E, f.B, f.J, TDT);
    core.apply_damping(f.E, f.B, TDT, 3, Scalar(0.5), Scalar(3.0));
    core.apply_inner_bc(f.E, f.B, f.B0, par, time + TDT, time + 0.5 * TDT);
    time += TDT;
  }
}

// Scenario 3: full Deutsch retarded IC computed per-rank on owned cells
// through the l2g quadrature path, then explicit evolution with the
// Deutsch analytic inner BC (leapfrog staggering: B seeded at -dt/2).
void run_deutsch(core_t& core, exchanger_t& ex, fields& f,
                 int n_steps) {
  dec_inner_bc_params par;
  par.Bp = 1.0;
  par.Omega = 0.2;
  par.obliquity = 1.0471975511965976;
  par.use_deutsch = true;
  core.set_initial_deutsch(f.E, f.B, par.Bp, par.Omega, par.obliquity,
                           Scalar(0), Scalar(-0.5 * TDT));
  double time = 0.0;
  for (int s = 0; s < n_steps; s++) {
    ex.exchange_edge(f.E, core.e_split());
    core.faraday(f.E, f.B, TDT);
    ex.exchange_face(f.B, core.b_split());
    core.ampere(f.E, f.B, f.J, TDT);
    core.apply_damping(f.E, f.B, TDT, 3, Scalar(0.5), Scalar(3.0));
    core.apply_inner_bc(f.E, f.B, f.B0, par, time + TDT, time + 0.5 * TDT);
    time += TDT;
  }
}

// Scenario 2: semi-implicit + damping + PEC, ghost refresh inside every
// Picard iteration.
void run_semi(core_t& core, exchanger_t& ex, fields& f,
              int n_steps) {
  const Scalar beta = 0.55, alpha = Scalar(1) - beta;
  const int iters = 4;
  for (int s = 0; s < n_steps; s++) {
    ex.exchange_edge(f.E, core.e_split());
    ex.exchange_face(f.B, core.b_split());
    core.compute_rhs(f.E, f.B, f.J, f.dE, f.dB);
    core.euler_predict(f.E, f.dE, f.tmpE, f.B, f.dB, f.tmpB, TDT);
    core.apply_damping(f.tmpE, f.tmpB, TDT, 3, Scalar(0.5), Scalar(3.0));
    core.apply_pec_bc(f.tmpE, f.tmpB);
    for (int it = 0; it < iters; it++) {
      ex.exchange_edge(f.tmpE, core.e_split());
      ex.exchange_face(f.tmpB, core.b_split());
      core.compute_rhs(f.tmpE, f.tmpB, f.J, f.dE2, f.dB2);
      core.picard_combine(f.E, f.dE, f.dE2, f.tmpE, f.B, f.dB, f.dB2,
                          f.tmpB, TDT, alpha, beta);
      core.apply_damping(f.tmpE, f.tmpB, TDT, 3, Scalar(0.5), Scalar(3.0));
      core.apply_pec_bc(f.tmpE, f.tmpB);
    }
    for (int e = 0; e < core.n_edges_local(); e++) f.E[e] = f.tmpE[e];
    for (int b = 0; b < core.n_faces_local(); b++) f.B[b] = f.tmpB[b];
    core.apply_pec_bc(f.E, f.B);
  }
}

// Max |dist - ref| / max|ref| over this rank's OWNED cells.  The
// reference core has the identity (global) layout, so its buffers are
// directly global-indexed: overlay the distributed owned values onto a
// copy of the reference and diff.
Scalar owned_rel_diff(core_t& dist_core, fields& dist_f, fields& ref_f) {
  std::vector<Scalar> Eg(ref_f.E.host_ptr(),
                         ref_f.E.host_ptr() + ref_f.E.size());
  std::vector<Scalar> Bg(ref_f.B.host_ptr(),
                         ref_f.B.host_ptr() + ref_f.B.size());
  Scalar scale = Scalar(0);
  for (auto v : Eg) scale = std::max(scale, std::abs(v));
  for (auto v : Bg) scale = std::max(scale, std::abs(v));

  std::vector<Scalar> Eo = Eg, Bo = Bg;
  dist_core.edge_owned_to_global(dist_f.E, Eo.data());
  dist_core.face_owned_to_global(dist_f.B, Bo.data());
  Scalar diff = Scalar(0);
  for (size_t i = 0; i < Eg.size(); i++)
    diff = std::max(diff, std::abs(Eo[i] - Eg[i]));
  for (size_t i = 0; i < Bg.size(); i++)
    diff = std::max(diff, std::abs(Bo[i] - Bg[i]));
  return diff / scale;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Optional first argument: angular rank count A (Phase 7A canonical
  // path-ordered decomposition, prismatic_mpi_comm::create(world, A, K)
  // + prismatic_partition::combined).  Without it, the legacy 20-face
  // identity wiring is exercised (create(world, K) + combined_ico_face).
  int A_arg = 0;
  if (argc > 1) A_arg = std::atoi(argv[1]);
  const bool canonical = A_arg > 0;
  const int A = canonical ? A_arg : 20;
  // Optional second argument: ranks_per_node for the 7E node-tile rank
  // remap (canonical mode only) — results must stay bit-exact, only
  // which process plays which logical rank changes.
  int rpn = 0;
  if (argc > 2) rpn = std::atoi(argv[2]);

  if (world_size % A != 0 ||
      (canonical && prismatic_partition::min_patch_level_for(A) < 0)) {
    if (world_rank == 0)
      std::fprintf(stderr, "SKIP: needs A*K ranks with valid A=%d (got %d)\n",
                   A, world_size);
    MPI_Finalize();
    return 0;
  }
  const int K = world_size / A;

  prismatic_mpi_comm mcomm =
      canonical ? prismatic_mpi_comm::create(MPI_COMM_WORLD, A, K, rpn)
                : prismatic_mpi_comm::create(MPI_COMM_WORLD, K);

  prismatic_mesh mesh;
  mesh.build(TL, TN_r, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  // Deterministic synthetic global cochains (same on every rank).
  std::vector<Scalar> Eg(mesh.m_N_edges), Bg(mesh.m_N_faces),
      Jg(mesh.m_N_edges);
  for (int e = 0; e < mesh.m_N_edges; e++) {
    Eg[e] = std::sin(Scalar(0.013) * e) + Scalar(0.37);
    Jg[e] = Scalar(0.05) * std::sin(Scalar(0.011) * e + Scalar(0.3));
  }
  for (int f = 0; f < mesh.m_N_faces; f++)
    Bg[f] = std::cos(Scalar(0.007) * f) - Scalar(0.21);

  // Single-rank reference (identity layout, inactive exchanger).
  auto ref_part = prismatic_partition::single_rank(TL, TN_r);
  ref_part.set_topology(&topo);
  auto ref_mp = prismatic_mesh_partition::build(ref_part, topo);
  core_t ref_core;
  ref_core.build(mesh, ref_mp);
  exchanger_t no_ex;  // inactive

  // This rank's slice of the combined A x K decomposition.
  auto part = canonical
                  ? prismatic_partition::combined(TL, TN_r, A, K,
                                                  mcomm.world_rank())
                  : prismatic_partition::combined_ico_face(
                        TL, TN_r, K, mcomm.radial_rank(),
                        mcomm.angular_rank());
  part.set_topology(&topo);
  auto mp = prismatic_mesh_partition::build(part, topo);
  core_t core;
  core.build(mesh, mp);
  // Both exchange paths: packed (device-direct structure, here on host
  // pointers) and the legacy full-buffer host-staged fallback.  They
  // must both reproduce the single-rank reference bit-exactly.
  exchanger_t ex_packed, ex_staged;
  ex_packed.init(mp, mcomm, /*device_direct=*/true);
  ex_staged.init(mp, mcomm, /*device_direct=*/false);

  const Scalar tol = Scalar(1e-4);
  bool all_ok = true;

  struct scenario {
    const char* name;
    void (*run)(core_t&, exchanger_t&, fields&, int);
    int n_steps;
  };
  struct mode {
    const char* name;
    exchanger_t* ex;
  };
  for (auto sc : {scenario{"explicit + inner dipole BC", run_explicit, 10},
                  scenario{"semi-implicit + PEC", run_semi, 6},
                  scenario{"Deutsch IC + Deutsch BC", run_deutsch, 8}}) {
    for (auto md : {mode{"packed", &ex_packed}, mode{"staged", &ex_staged}}) {
      fields ref_f, dist_f;
      ref_f.alloc(ref_core);
      dist_f.alloc(core);
      ref_f.seed(ref_core, Eg, Bg, Jg);
      dist_f.seed(core, Eg, Bg, Jg);

      sc.run(ref_core, no_ex, ref_f, sc.n_steps);
      sc.run(core, *md.ex, dist_f, sc.n_steps);

      Scalar rel = owned_rel_diff(core, dist_f, ref_f);
      Scalar rel_max = 0;
      MPI_Allreduce(&rel, &rel_max, 1, mpi_scalar_type(), MPI_MAX,
                    MPI_COMM_WORLD);
      bool ok = rel_max < tol;
      if (world_rank == 0)
        std::printf(
            "%-28s [%s] (%s %dx%d, %2d steps): max rel diff = %.3e  %s\n",
            sc.name, md.name, canonical ? "canonical" : "legacy", A, K,
            sc.n_steps, double(rel_max), ok ? "PASS" : "FAIL");
      if (!ok) all_ok = false;
    }
  }

  // ---- Phase 7B reduce roundtrip ---------------------------------------
  // exchange fills every ghost with the owner's value; reduce ships the
  // ghosts back and accumulates, then zeroes them.  Owned slot ->
  // original * (1 + copies), where copies = occurrences of the local
  // index in this rank's send lists over both axes; ghost slots -> 0.
  for (auto md : {mode{"packed", &ex_packed}, mode{"staged", &ex_staged}}) {
    fields f;
    f.alloc(core);
    f.seed(core, Eg, Bg, Jg);

    std::vector<Scalar> expE(f.E.host_ptr(),
                             f.E.host_ptr() + f.E.size());
    std::vector<Scalar> expB(f.B.host_ptr(),
                             f.B.host_ptr() + f.B.size());
    // Expected result: owned slot += one original value per send-list
    // occurrence (the contribution each ghosting peer ships back equals
    // the owner's exchanged value); ghost slots end zero.
    auto add_copies = [&](cochain_type t, int off, std::vector<Scalar>& exp,
                          const std::vector<Scalar>& orig) {
      for (const halo_plan* pl :
           {&mp.radial_plan_local(t), &mp.angular_plan_local(t)}) {
        for (auto const& pe : pl->peers) {
          for (int l : pe.send_global_idx) exp[l + off] += orig[l + off];
        }
      }
      for (const halo_plan* pl :
           {&mp.radial_plan_local(t), &mp.angular_plan_local(t)}) {
        for (auto const& pe : pl->peers) {
          for (int l : pe.recv_global_idx) exp[l + off] = Scalar(0);
        }
      }
    };
    const std::vector<Scalar> origE = expE, origB = expB;
    add_copies(cochain_type::h_edge, 0, expE, origE);
    add_copies(cochain_type::v_edge, core.e_split(), expE, origE);
    add_copies(cochain_type::tri_face, 0, expB, origB);
    add_copies(cochain_type::rect_face, core.b_split(), expB, origB);

    md.ex->exchange_edge(f.E, core.e_split());
    md.ex->exchange_face(f.B, core.b_split());
    md.ex->reduce_edge(f.E, core.e_split());
    md.ex->reduce_face(f.B, core.b_split());

    Scalar diff = 0;
    for (size_t i = 0; i < expE.size(); ++i)
      diff = std::max(diff, std::abs(f.E[int(i)] - expE[i]));
    for (size_t i = 0; i < expB.size(); ++i)
      diff = std::max(diff, std::abs(f.B[int(i)] - expB[i]));
    Scalar diff_max = 0;
    MPI_Allreduce(&diff, &diff_max, 1, mpi_scalar_type(), MPI_MAX,
                  MPI_COMM_WORLD);
    const bool ok = diff_max < Scalar(1e-5);
    if (world_rank == 0)
      std::printf("reduce roundtrip            [%s] (%s %dx%d): max abs diff "
                  "= %.3e  %s\n",
                  md.name, canonical ? "canonical" : "legacy", A, K,
                  double(diff_max), ok ? "PASS" : "FAIL");
    if (!ok) all_ok = false;
  }

  if (world_rank == 0)
    std::printf("\n%s\n", all_ok ? "ALL PASS" : "FAILURE");
  MPI_Finalize();
  return all_ok ? 0 : 1;
}
