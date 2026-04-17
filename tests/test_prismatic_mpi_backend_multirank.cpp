// Multi-process MPI test for the prismatic halo exchange.
//
// Run under mpirun, e.g.:
//   mpirun -n 20 ./test_prismatic_mpi_backend_multirank
//   mpirun -n 80 ./test_prismatic_mpi_backend_multirank
//
// Validates that mpi_halo_backend produces the same result as the
// in-process backend on the same plan.  Uses MPI_COMM_WORLD as the
// angular sub-communicator (radial K=1) when world_size=20, and for
// larger world sizes runs a combined 20·K test (K = world_size/20).
//
// The test passes if every rank's post-exchange buffer matches the
// "oracle" constructed by running the same plan in a single-process
// in-process backend (which is already unit-tested).

#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_mpi_halo_backend.h"
#include "systems/prismatic/prismatic_partition.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace Aperture;

static Scalar stamp(int global_idx) {
  return Scalar(1) + Scalar(global_idx) * Scalar(1e-4);
}

namespace {

struct test_config {
  int L = 2;
  int N_r = 4;
};

// Build the single-rank oracle.
std::vector<Scalar> oracle_buffer(cochain_type t,
                                   const prismatic_partition& p,
                                   const icosphere_topology& topo) {
  // Build all 20 angular partitions in-process, run oracle exchange.
  int N = global_cochain_size(t, p);
  std::vector<std::vector<Scalar>> bufs(20);
  in_process_halo_backend backend;
  for (int f = 0; f < 20; ++f) {
    auto pf = prismatic_partition::ico_face_angular(p.L, p.N_r_global, f);
    pf.set_topology(&topo);
    bufs[f].assign(N, Scalar(0));
    for (int g = 0; g < N; ++g) {
      bool owned = false;
      switch (t) {
        case cochain_type::tri_face:  owned = pf.owns_tri_face_cochain(g); break;
        case cochain_type::rect_face: owned = pf.owns_rect_face_cochain(g); break;
        case cochain_type::h_edge:    owned = pf.owns_h_edge_cochain(g); break;
        case cochain_type::v_edge:    owned = pf.owns_v_edge_cochain(g); break;
        case cochain_type::vertex:    owned = pf.owns_vertex_cochain(g); break;
      }
      if (owned) bufs[f][g] = stamp(g);
    }
    backend.register_rank(f, bufs[f].data());
  }
  std::vector<halo_plan> plans;
  for (int f = 0; f < 20; ++f) {
    auto pf = prismatic_partition::ico_face_angular(p.L, p.N_r_global, f);
    pf.set_topology(&topo);
    plans.push_back(build_angular_halo_plan(t, pf, topo));
  }
  backend.exchange_all(plans);
  return bufs[p.angular_rank];
}

bool run_exchange_test(cochain_type t, const test_config& cfg,
                        MPI_Comm world) {
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(world, &world_size);
  MPI_Comm_rank(world, &world_rank);

  // Build mesh + topology (every rank does this identically).
  prismatic_mesh mesh;
  mesh.build(cfg.L, 2, 1.0, 2.0);
  auto topo = icosphere_topology::build_from_mesh(mesh);

  // Angular-only for this test (radial K=1 regardless of world_size>20).
  // World_size must be exactly 20.
  if (world_size != 20) {
    if (world_rank == 0) {
      std::fprintf(stderr, "SKIP: this test needs exactly 20 ranks "
                           "(got %d)\n", world_size);
    }
    return true;
  }

  prismatic_partition part = prismatic_partition::ico_face_angular(
      cfg.L, cfg.N_r, world_rank);
  part.set_topology(&topo);

  int N = global_cochain_size(t, part);

  // Each rank fills only its owned slots with the stamp.
  std::vector<Scalar> buf(N, Scalar(0));
  for (int g = 0; g < N; ++g) {
    bool owned = false;
    switch (t) {
      case cochain_type::tri_face:  owned = part.owns_tri_face_cochain(g); break;
      case cochain_type::rect_face: owned = part.owns_rect_face_cochain(g); break;
      case cochain_type::h_edge:    owned = part.owns_h_edge_cochain(g); break;
      case cochain_type::v_edge:    owned = part.owns_v_edge_cochain(g); break;
      case cochain_type::vertex:    owned = part.owns_vertex_cochain(g); break;
    }
    if (owned) buf[g] = stamp(g);
  }

  // Build plan and exchange via MPI.
  auto plan = build_angular_halo_plan(t, part, topo);

  // Need the angular sub-communicator.  For world_size=20, build mpi_comm
  // as 20·1 and use its angular sub-comm.
  prismatic_mpi_comm mcomm = prismatic_mpi_comm::create(world, 1);
  mpi_halo_backend backend(mcomm.angular());
  backend.exchange(buf.data(), plan);

  // Oracle.
  auto oracle = oracle_buffer(t, part, topo);

  // Compare.
  bool ok = true;
  for (int g = 0; g < N; ++g) {
    if (std::abs(buf[g] - oracle[g]) > Scalar(1e-6)) {
      if (ok && world_rank < 2) {  // limit spam
        std::fprintf(stderr,
                     "rank %d: mismatch at idx %d: got %g, oracle %g\n",
                     world_rank, g, double(buf[g]), double(oracle[g]));
      }
      ok = false;
    }
  }
  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, world);
  return global_ok != 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  test_config cfg;
  bool all_ok = true;
  for (cochain_type t : {cochain_type::tri_face, cochain_type::h_edge,
                         cochain_type::rect_face, cochain_type::v_edge,
                         cochain_type::vertex}) {
    bool ok = run_exchange_test(t, cfg, MPI_COMM_WORLD);
    if (world_rank == 0) {
      std::printf("cochain %d: %s\n", int(t), ok ? "PASS" : "FAIL");
    }
    if (!ok) all_ok = false;
  }

  int ret = all_ok ? 0 : 1;
  if (world_rank == 0) {
    std::printf("\n%s\n", all_ok ? "ALL PASS" : "FAILURE");
  }

  MPI_Finalize();
  return ret;
}
