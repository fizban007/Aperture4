#include "systems/prismatic/prismatic_mpi_halo_backend.h"
#include <type_traits>

namespace Aperture {

MPI_Datatype mpi_scalar_type() {
  // Aperture's Scalar is typically float (single-precision GPU-friendly).
  // Support double for completeness.
  if (std::is_same<Scalar, float>::value) return MPI_FLOAT;
  if (std::is_same<Scalar, double>::value) return MPI_DOUBLE;
  // Fall back: treat as raw bytes (works as long as all ranks agree).
  return MPI_BYTE;
}

void mpi_halo_backend::exchange(Scalar* data, const halo_plan& plan, int tag) {
  const int n_peers = int(plan.peers.size());
  if (n_peers == 0) return;

  const MPI_Datatype dt = mpi_scalar_type();

  // Grow scratch.
  if (int(m_send_bufs.size()) < n_peers) m_send_bufs.resize(n_peers);
  if (int(m_recv_bufs.size()) < n_peers) m_recv_bufs.resize(n_peers);
  m_requests.assign(2 * n_peers, MPI_REQUEST_NULL);

  // Pack sends and post Irecv/Isend.
  for (int i = 0; i < n_peers; ++i) {
    auto const& pe = plan.peers[i];

    // Pack send buffer.
    auto& sb = m_send_bufs[i];
    sb.resize(pe.send_global_idx.size());
    for (size_t j = 0; j < pe.send_global_idx.size(); ++j) {
      sb[j] = data[pe.send_global_idx[j]];
    }

    // Size recv buffer (values filled by MPI).
    auto& rb = m_recv_bufs[i];
    rb.resize(pe.recv_global_idx.size());

    if (!rb.empty()) {
      MPI_Irecv(rb.data(), int(rb.size()), dt, pe.peer_rank, tag, m_comm,
                &m_requests[2 * i]);
    }
    if (!sb.empty()) {
      MPI_Isend(sb.data(), int(sb.size()), dt, pe.peer_rank, tag, m_comm,
                &m_requests[2 * i + 1]);
    }
  }

  // Complete all pending requests (some may be MPI_REQUEST_NULL — Waitall
  // handles those gracefully).
  MPI_Waitall(int(m_requests.size()), m_requests.data(),
              MPI_STATUSES_IGNORE);

  // Unpack receives.
  for (int i = 0; i < n_peers; ++i) {
    auto const& pe = plan.peers[i];
    auto const& rb = m_recv_bufs[i];
    for (size_t j = 0; j < pe.recv_global_idx.size(); ++j) {
      data[pe.recv_global_idx[j]] = rb[j];
    }
  }
}

}  // namespace Aperture
