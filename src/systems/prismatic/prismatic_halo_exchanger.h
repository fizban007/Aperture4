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
// Device-resident buffers are staged through the host with full-buffer
// copies — correctness-first; packed device staging / GPU-aware MPI is
// a Phase 5 optimization.
// =========================================================================
class prismatic_halo_exchanger {
 public:
  prismatic_halo_exchanger() = default;

  void init(const prismatic_mesh_partition& mp,
            const prismatic_mpi_comm& comm) {
    m_mp = &mp;
    m_active = !comm.is_single_rank();
    if (m_active) {
      m_radial = std::make_unique<mpi_halo_backend>(comm.radial());
      m_angular = std::make_unique<mpi_halo_backend>(comm.angular());
    }
  }

  bool active() const { return m_active; }

  // Exchange one cochain type.  `base` points at the cochain's block
  // (i.e. buf + split for the second block of a combined buffer).
  void exchange(cochain_type t, Scalar* base) {
    if (!m_active) return;
    m_radial->exchange(base, m_mp->radial_plan_local(t), int(t));
    m_angular->exchange(base, m_mp->angular_plan_local(t), int(t));
  }

  // Combined local-layout edge buffer [h_edge | v_edge], split at
  // e_split (== dec_solver_dist::e_split()).
  void exchange_edge(buffer<Scalar>& buf, int e_split) {
    if (!m_active) return;
    stage_in(buf);
    exchange(cochain_type::h_edge, buf.host_ptr());
    exchange(cochain_type::v_edge, buf.host_ptr() + e_split);
    stage_out(buf);
  }

  // Combined local-layout face buffer [tri_face | rect_face], split at
  // b_split (== dec_solver_dist::b_split()).
  void exchange_face(buffer<Scalar>& buf, int b_split) {
    if (!m_active) return;
    stage_in(buf);
    exchange(cochain_type::tri_face, buf.host_ptr());
    exchange(cochain_type::rect_face, buf.host_ptr() + b_split);
    stage_out(buf);
  }

 private:
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

  const prismatic_mesh_partition* m_mp = nullptr;
  bool m_active = false;
  std::unique_ptr<mpi_halo_backend> m_radial;
  std::unique_ptr<mpi_halo_backend> m_angular;
};

}  // namespace Aperture
