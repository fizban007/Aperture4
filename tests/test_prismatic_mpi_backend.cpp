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
