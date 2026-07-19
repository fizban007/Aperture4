#include "catch2/catch_all.hpp"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_mpi_halo_backend.h"
#include <mpi.h>
#include <vector>

using namespace Aperture;
using Catch::Approx;

// =========================================================================
// Single-process MPI tests.  We don't need mpirun here — all tests use
// MPI_COMM_SELF (size 1) or the degenerate single_rank() factory.
// Multi-process correctness is validated by a separate mpirun-driven
// test binary (test_prismatic_mpi_backend_multirank) compared against
// the in-process backend as ground truth.
// =========================================================================

TEST_CASE("single_rank factory yields is_single_rank()==true",
          "[prismatic][mpi_backend]") {
  auto comm = prismatic_mpi_comm::single_rank();
  REQUIRE(comm.is_single_rank());
  REQUIRE(comm.n_angular_ranks() == 1);
  REQUIRE(comm.n_radial_ranks() == 1);
}

TEST_CASE("mpi_halo_backend with empty plan is a no-op",
          "[prismatic][mpi_backend]") {
  prismatic_mpi_comm::ensure_mpi_initialized();
  mpi_halo_backend backend(MPI_COMM_SELF);

  std::vector<Scalar> buf(16, Scalar(42));
  halo_plan empty_plan;  // no peers
  backend.exchange(buf.data(), empty_plan);

  for (Scalar v : buf) REQUIRE(v == Scalar(42));
}

TEST_CASE("mpi_halo_backend: self-send on MPI_COMM_SELF",
          "[prismatic][mpi_backend]") {
  // Smallest multi-peer exchange without mpirun: single rank that sends
  // some indices to itself and receives them into different slots.
  // Validates the pack / Isend+Irecv / Waitall / unpack path.
  prismatic_mpi_comm::ensure_mpi_initialized();
  mpi_halo_backend backend(MPI_COMM_SELF);

  std::vector<Scalar> buf(16, Scalar(0));
  for (int i = 0; i < 8; ++i) buf[i] = Scalar(i + 1);  // owned slots

  halo_plan plan;
  halo_plan::peer_entry pe;
  pe.peer_rank = 0;  // self
  pe.send_global_idx = {0, 1, 2, 3};
  pe.recv_global_idx = {8, 9, 10, 11};
  plan.peers.push_back(pe);

  backend.exchange(buf.data(), plan);

  // Data copied from owned slots [0..3] into ghost slots [8..11].
  REQUIRE(buf[8]  == Approx(Scalar(1)));
  REQUIRE(buf[9]  == Approx(Scalar(2)));
  REQUIRE(buf[10] == Approx(Scalar(3)));
  REQUIRE(buf[11] == Approx(Scalar(4)));
  // Owned slots unchanged.
  for (int i = 0; i < 8; ++i) REQUIRE(buf[i] == Approx(Scalar(i + 1)));
  // Unused slots remain zero.
  for (int i = 12; i < 16; ++i) REQUIRE(buf[i] == Approx(Scalar(0)));
}

TEST_CASE("create() requires world_size == 20 * n_radial",
          "[prismatic][mpi_backend]") {
  prismatic_mpi_comm::ensure_mpi_initialized();
  int self_size = 0;
  MPI_Comm_size(MPI_COMM_SELF, &self_size);
  REQUIRE(self_size == 1);
  REQUIRE_THROWS(prismatic_mpi_comm::create(MPI_COMM_SELF, 1));   // 1 != 20*1
  REQUIRE_THROWS(prismatic_mpi_comm::create(MPI_COMM_SELF, 2));   // 1 != 20*2
}

TEST_CASE("mpi_scalar_type returns a usable MPI datatype",
          "[prismatic][mpi_backend]") {
  MPI_Datatype dt = mpi_scalar_type();
  int size = 0;
  MPI_Type_size(dt, &size);
  REQUIRE(size == int(sizeof(Scalar)));
}

TEST_CASE("node tiling: shapes and coordinate coverage",
          "[prismatic][mpi][tiling]") {
  int a_t, k_t;
  // 8-GCD node on 80x100: squarest valid tile is 4x2.
  REQUIRE(prismatic_mpi_comm::node_tile_shape(80, 100, 8, a_t, k_t));
  REQUIRE(a_t == 4);
  REQUIRE(k_t == 2);
  // 320x25: k_t must divide 25 -> only 8x1 works.
  REQUIRE(prismatic_mpi_comm::node_tile_shape(320, 25, 8, a_t, k_t));
  REQUIRE(a_t == 8);
  REQUIRE(k_t == 1);
  // 4x2 with 8 ranks per node: the whole grid is one tile.
  REQUIRE(prismatic_mpi_comm::node_tile_shape(4, 2, 8, a_t, k_t));
  REQUIRE(a_t == 4);
  REQUIRE(k_t == 2);
  // Unsatisfiable: 3 ranks per node never factors into a 20x2 grid.
  REQUIRE_FALSE(prismatic_mpi_comm::node_tile_shape(20, 2, 3, a_t, k_t));
  // rpn <= 1: no tiling.
  REQUIRE_FALSE(prismatic_mpi_comm::node_tile_shape(20, 2, 1, a_t, k_t));

  // Coverage + compactness: every (ang, rad) hit exactly once, and each
  // consecutive rpn block spans exactly one a_t x k_t patch.
  for (auto [A, K, rpn] : {std::array<int, 3>{20, 4, 8},
                           std::array<int, 3>{16, 4, 4},
                           std::array<int, 3>{80, 2, 8}}) {
    REQUIRE(prismatic_mpi_comm::node_tile_shape(A, K, rpn, a_t, k_t));
    std::vector<int> hits(A * K, 0);
    for (int node = 0; node < A * K / rpn; ++node) {
      int amin = 1 << 30, amax = -1, kmin = 1 << 30, kmax = -1;
      for (int i = 0; i < rpn; ++i) {
        int ang, rad;
        prismatic_mpi_comm::node_tile_coords(A, a_t, k_t, node * rpn + i,
                                             ang, rad);
        REQUIRE(ang >= 0);
        REQUIRE(ang < A);
        REQUIRE(rad >= 0);
        REQUIRE(rad < K);
        hits[rad * A + ang]++;
        amin = std::min(amin, ang); amax = std::max(amax, ang);
        kmin = std::min(kmin, rad); kmax = std::max(kmax, rad);
      }
      REQUIRE(amax - amin + 1 == a_t);   // contiguous angular band
      REQUIRE(kmax - kmin + 1 == k_t);   // contiguous radial band
    }
    for (int h : hits) REQUIRE(h == 1);
  }
}
