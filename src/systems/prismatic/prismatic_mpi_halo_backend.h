#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_halo_plan.h"
#include <mpi.h>
#include <vector>

namespace Aperture {

// =========================================================================
// MPI halo exchange backend.
//
// Executes a halo_plan over a given MPI communicator using Isend/Irecv
// + Waitall.  The plan's peer_rank fields are interpreted as ranks in
// the provided communicator (per the convention set up by
// prismatic_mpi_comm: angular plan peers == ico_face index == rank in
// comm_angular; radial plan peers == radial_rank == rank in comm_radial).
//
// Data layout: buffers are currently global-sized (same convention as
// the Phase 2 in-process backend) — every rank has a buffer of size
// N_global and only the owned + ghost slots carry meaningful values.
// Phase 4 will introduce a local-indexing layer; the plan's
// {send,recv}_global_idx will become per-rank local offsets, and this
// backend will be updated in one place.
//
// Currently uses blocking-style pack -> Isend/Irecv -> Waitall ->
// unpack.  Phase 3 lights the critical path end-to-end; latency-hiding
// with neighborhood collectives is a Phase 4 or later optimization.
// =========================================================================
class mpi_halo_backend {
 public:
  explicit mpi_halo_backend(MPI_Comm comm) : m_comm(comm) {}

  // Execute a halo exchange on the given buffer.  Must be called
  // collectively by all ranks in m_comm — peers expect matching
  // Isend / Irecv posts.  A no-op if the plan has zero peers.
  void exchange(Scalar* data, const halo_plan& plan, int tag = 0);

  MPI_Comm comm() const { return m_comm; }

 private:
  MPI_Comm m_comm;

  // Scratch reused across calls (grown on demand).
  std::vector<std::vector<Scalar>> m_send_bufs;
  std::vector<std::vector<Scalar>> m_recv_bufs;
  std::vector<MPI_Request> m_requests;
};

// MPI datatype matching Aperture's Scalar type (float or double).
MPI_Datatype mpi_scalar_type();

}  // namespace Aperture
