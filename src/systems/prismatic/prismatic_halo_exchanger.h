#pragma once

#include "core/buffer.hpp"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_mpi_halo_backend.h"
#include <memory>

namespace Aperture {

// =========================================================================
// Phase 4.1b B2 — halo sync-point executor.
//
// Binds the partition bundle's LOCAL-indexed halo plans to the two MPI
// sub-communicators (see prismatic_mpi_comm: angular peers are ico-face
// indices in comm_angular, radial peers are slab indices in comm_radial)
// and exchanges the ghost slots of the combined [h|v] / [tri|rect]
// local-layout field buffers used by dec_solver_dist.
//
// A default-constructed exchanger is INACTIVE: every exchange is a
// no-op.  This is the single-rank mode — the same step code runs
// unchanged with ghost-free identity layouts.
//
// All exchange methods are collective on the sub-communicators: every
// rank must reach the same sync points in the same order (the lockstep
// step sequence guarantees this).  Radial exchanges run before angular
// ones; the order is actually immaterial — no d1/d1^T stencil ever
// needs a corner (diagonal) ghost, because rect faces are angularly
// owned by their sphere-edge (see test_prismatic_dec_dist).
//
// Two data paths, selected by init(..., device_direct):
//
//   device_direct == true (default) — PACKED path: the plans' per-peer
//     index lists are flattened once at init into ExecPolicy-resident
//     index buffers (block offsets baked in), and each sync point runs
//     gather-pack kernel -> MPI on the packed messages -> scatter-
//     unpack kernel.  Only the message-sized packed buffers ever move;
//     with a GPU-aware MPI (mpi_gpu_direct_available) the messages are
//     posted directly from device memory, otherwise just the packed
//     messages stage through the host.  Under the host exec policy the
//     same code runs with host pointers, so the packed path is testable
//     on CPU (test_prismatic_solver_multirank runs both modes).
//
//   device_direct == false — legacy HOST-STAGED path (config fallback
//     "halo_device_direct = false" for debugging): full-buffer
//     copy_to_host, per-cochain host exchange through mpi_halo_backend,
//     full-buffer copy_to_device.
//
// The exchange message structure (one Irecv/Isend pair per peer per
// cochain per axis, tag = cochain id, radial round then angular round)
// is identical in both paths, so any mix of paths across ranks is
// wire-compatible — a single misbehaving rank can be flipped to the
// host path in isolation when debugging.
// =========================================================================
template <typename ExecPolicy>
class prismatic_halo_exchanger {
 public:
  prismatic_halo_exchanger() = default;

  void init(const prismatic_mesh_partition& mp,
            const prismatic_mpi_comm& comm, bool device_direct = true) {
    m_mp = &mp;
    m_active = !comm.is_single_rank();
    m_device_direct = device_direct;
    if (!m_active) return;
    m_radial = std::make_unique<mpi_halo_backend>(comm.radial());
    m_angular = std::make_unique<mpi_halo_backend>(comm.angular());
    if (m_device_direct) {
      const int e_split = mp.layout(cochain_type::h_edge).local_size();
      const int b_split = mp.layout(cochain_type::tri_face).local_size();
      const std::pair<cochain_type, int> blocks[] = {
          {cochain_type::h_edge, 0},
          {cochain_type::v_edge, e_split},
          {cochain_type::tri_face, 0},
          {cochain_type::rect_face, b_split}};
      for (auto [t, off] : blocks) {
        build_packed(0, t, mp.radial_plan_local(t), off);
        build_packed(1, t, mp.angular_plan_local(t), off);
      }
    }
  }

  bool active() const { return m_active; }
  bool device_direct() const { return m_device_direct; }

  // Combined local-layout edge buffer [h_edge | v_edge], split at
  // e_split (== dec_solver_dist::e_split()).
  void exchange_edge(buffer<Scalar>& buf, int e_split) {
    if (!m_active) return;
    if (m_device_direct) {
      exchange_packed_round(buf, cochain_type::h_edge);
      exchange_packed_round(buf, cochain_type::v_edge);
      return;
    }
    stage_in(buf);
    exchange(cochain_type::h_edge, buf.host_ptr());
    exchange(cochain_type::v_edge, buf.host_ptr() + e_split);
    stage_out(buf);
  }

  // Combined local-layout face buffer [tri_face | rect_face], split at
  // b_split (== dec_solver_dist::b_split()).
  void exchange_face(buffer<Scalar>& buf, int b_split) {
    if (!m_active) return;
    if (m_device_direct) {
      exchange_packed_round(buf, cochain_type::tri_face);
      exchange_packed_round(buf, cochain_type::rect_face);
      return;
    }
    stage_in(buf);
    exchange(cochain_type::tri_face, buf.host_ptr());
    exchange(cochain_type::rect_face, buf.host_ptr() + b_split);
    stage_out(buf);
  }

  // ---- packed (device-direct) path -------------------------------------
  // Public because CUDA/HIP require extended __device__ lambdas to live
  // in public methods; treat as implementation detail.

  // One per (axis, cochain type): flattened plan indices with the
  // combined-buffer block offset baked in, plus the contiguous message
  // buffers the MPI calls see.
  struct packed_plan {
    buffer<int> send_idx, recv_idx;
    buffer<Scalar> send_msg, recv_msg;
    std::vector<mpi_halo_backend::packed_peer> peers;
    int total_send = 0, total_recv = 0;
  };

  void exchange_packed_round(buffer<Scalar>& buf, cochain_type t) {
    // Radial round, then angular — same order as the host-staged path.
    run_packed(buf, m_packed[0][int(t)], *m_radial, int(t));
    run_packed(buf, m_packed[1][int(t)], *m_angular, int(t));
  }

  void run_packed(buffer<Scalar>& buf, packed_plan& pp,
                  mpi_halo_backend& backend, int tag) {
    if (pp.peers.empty()) return;

    if (pp.total_send > 0) {
      ExecPolicy::launch(
          [n = pp.total_send] LAMBDA(auto data, auto idx, auto msg) {
            ExecPolicy::loop(0, n,
                             [&] LAMBDA(int j) { msg[j] = data[idx[j]]; });
          },
          buf, pp.send_idx, pp.send_msg);
      // The MPI layer reads the packed buffer from the host thread (or
      // the NIC does, GPU-direct); the pack kernel must be complete.
      ExecPolicy::sync();
    }

    const bool dev_msgs =
        buf.mem_type() != MemType::host_only && mpi_gpu_direct_available();
    const Scalar* sptr;
    Scalar* rptr;
    if (dev_msgs) {
      sptr = pp.send_msg.dev_ptr();
      rptr = pp.recv_msg.dev_ptr();
    } else {
      pp.send_msg.copy_to_host();  // packed messages only; no-op host_only
      sptr = pp.send_msg.host_ptr();
      rptr = pp.recv_msg.host_ptr();
    }
    backend.exchange_packed(sptr, rptr, pp.peers, tag);
    if (!dev_msgs) pp.recv_msg.copy_to_device();

    if (pp.total_recv > 0) {
      ExecPolicy::launch(
          [n = pp.total_recv] LAMBDA(auto data, auto idx, auto msg) {
            ExecPolicy::loop(0, n,
                             [&] LAMBDA(int j) { data[idx[j]] = msg[j]; });
          },
          buf, pp.recv_idx, pp.recv_msg);
      // Same-stream launches serialize, so downstream kernels see the
      // fresh ghosts without an explicit sync here.
    }
  }

 private:
  // ---- legacy host-staged path -----------------------------------------
  void exchange(cochain_type t, Scalar* base) {
    m_radial->exchange(base, m_mp->radial_plan_local(t), int(t));
    m_angular->exchange(base, m_mp->angular_plan_local(t), int(t));
  }

  void stage_in(buffer<Scalar>& buf) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (buf.mem_type() != MemType::host_only) buf.copy_to_host();
#endif
  }
  void stage_out(buffer<Scalar>& buf) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (buf.mem_type() != MemType::host_only) buf.copy_to_device();
#endif
  }

  void build_packed(int axis, cochain_type t, const halo_plan& plan,
                    int block_off) {
    auto& pp = m_packed[axis][int(t)];
    pp.total_send = plan.total_send();
    pp.total_recv = plan.total_recv();
    pp.peers.clear();
    if (plan.peers.empty()) return;

    const auto mem = ExecPolicy::data_mem_type();
    for (auto* b : {&pp.send_idx, &pp.recv_idx}) b->set_memtype(mem);
    for (auto* b : {&pp.send_msg, &pp.recv_msg}) b->set_memtype(mem);
    pp.send_idx.resize(std::max(pp.total_send, 1));
    pp.recv_idx.resize(std::max(pp.total_recv, 1));
    pp.send_msg.resize(std::max(pp.total_send, 1));
    pp.recv_msg.resize(std::max(pp.total_recv, 1));

    int soff = 0, roff = 0;
    for (auto const& pe : plan.peers) {
      mpi_halo_backend::packed_peer m;
      m.peer_rank = pe.peer_rank;
      m.send_off = soff;
      m.send_cnt = int(pe.send_global_idx.size());
      m.recv_off = roff;
      m.recv_cnt = int(pe.recv_global_idx.size());
      for (int j = 0; j < m.send_cnt; ++j)
        pp.send_idx[soff + j] = pe.send_global_idx[j] + block_off;
      for (int j = 0; j < m.recv_cnt; ++j)
        pp.recv_idx[roff + j] = pe.recv_global_idx[j] + block_off;
      soff += m.send_cnt;
      roff += m.recv_cnt;
      pp.peers.push_back(m);
    }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    pp.send_idx.copy_to_device();
    pp.recv_idx.copy_to_device();
#endif
  }

  const prismatic_mesh_partition* m_mp = nullptr;
  bool m_active = false;
  bool m_device_direct = true;
  std::unique_ptr<mpi_halo_backend> m_radial;
  std::unique_ptr<mpi_halo_backend> m_angular;
  // Indexed [axis][int(cochain_type)]; axis 0 = radial, 1 = angular.
  // Slot `vertex` is unused (no vertex cochain in the field state).
  packed_plan m_packed[2][5];
};

}  // namespace Aperture
