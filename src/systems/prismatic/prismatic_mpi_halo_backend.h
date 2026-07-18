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

  // Per-peer slice of pre-packed contiguous message buffers (offsets /
  // counts into the flattened send and recv arrays).  Built once from a
  // halo_plan by the caller that owns the packing.
  struct packed_peer {
    int peer_rank = -1;
    int send_off = 0, send_cnt = 0;
    int recv_off = 0, recv_cnt = 0;
  };

  // Exchange ALREADY-PACKED messages: post Irecv(recv_msgs + off) /
  // Isend(send_msgs + off) per peer and Waitall.  The pointers may be
  // host or device memory — device requires a GPU-aware MPI (see
  // mpi_gpu_direct_available); packing/unpacking is the caller's job
  // (prismatic_halo_exchanger does it in device kernels).
  void exchange_packed(const Scalar* send_msgs, Scalar* recv_msgs,
                       const std::vector<packed_peer>& peers, int tag = 0);

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

// True when the MPI library accepts device pointers in point-to-point
// calls (GPU-aware / CUDA-aware MPI).  Queried once at first call; on
// OpenMPI this uses the runtime MPIX_Query_cuda_support(), which also
// covers the case of a CUDA-capable build whose accelerator component
// failed to load.  Always false in host-only builds.
bool mpi_gpu_direct_available();

}  // namespace Aperture
