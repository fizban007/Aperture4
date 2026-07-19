#pragma once

#include "core/buffer.hpp"
#include "systems/prismatic/prismatic_halo_plan.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_halo_backend.h"
#include <mpi.h>
#include <vector>

namespace Aperture {

// =========================================================================
// Phase 6 — replicated-field synchronization for the particle path.
//
// The particle kernels (push, gathers, Whitney deposit, vertex-recovery
// B fit, injector criteria) are global-indexed and consume the full
// global mesh, which every rank holds.  Distributed PIC therefore runs
// on GLOBAL-SIZED field replicas:
//
//   replicate_*: allgather the owned slots of the solver's local E/B
//     into a global replica on every rank (fields are small next to
//     the particle buffers, which are what distributed PIC shards).
//
//   reduce_*: MPI_Allreduce(SUM) a global deposit replica (J, rho, ...)
//     — each global slot receives contributions only from ranks whose
//     particles touch it, everyone else contributes the cleared 0 —
//     then pull this rank's owned+ghost slots into the solver-facing
//     local buffer.  After a reduce the replica equals the exact
//     global deposit on every rank (consumed by the injectors), and
//     the local buffer feeds Ampere / the sph gather.
//
// Host-staged (full-buffer copies around the collectives) —
// correctness-first, mirroring the pre-optimization halo exchanger.
// The owner-sum neighbor reduction from PARALLELIZATION_PLAN Phase 6
// remains the contained optimization if these collectives show up in
// profiles.
// =========================================================================
class prismatic_field_sync {
 public:
  void build(const prismatic_mesh_partition& mp, MPI_Comm world) {
    m_world = world;
    m_he.build(mp.layout(cochain_type::h_edge), world);
    m_ve.build(mp.layout(cochain_type::v_edge), world);
    m_tri.build(mp.layout(cochain_type::tri_face), world);
    m_rect.build(mp.layout(cochain_type::rect_face), world);
    m_vert.build(mp.layout(cochain_type::vertex), world);
    m_n_h_glob = mp.layout(cochain_type::h_edge).global_size();
    m_n_tri_glob = mp.layout(cochain_type::tri_face).global_size();
    m_e_split = mp.layout(cochain_type::h_edge).local_size();
    m_b_split = mp.layout(cochain_type::tri_face).local_size();
  }

  // local combined [h|v] edge buffer -> global replica (all ranks).
  void replicate_edge(buffer<Scalar>& local, buffer<Scalar>& global) {
    stage_host(local);
    m_he.replicate(local.host_ptr(), global.host_ptr(), m_world);
    m_ve.replicate(local.host_ptr() + m_e_split,
                   global.host_ptr() + m_n_h_glob, m_world);
    push_dev(global);
  }

  // local combined [tri|rect] face buffer -> global replica.
  void replicate_face(buffer<Scalar>& local, buffer<Scalar>& global) {
    stage_host(local);
    m_tri.replicate(local.host_ptr(), global.host_ptr(), m_world);
    m_rect.replicate(local.host_ptr() + m_b_split,
                     global.host_ptr() + m_n_tri_glob, m_world);
    push_dev(global);
  }

  // Global-sum a deposit replica; refresh it on every rank and pull the
  // owned+ghost slots into the local buffer.
  void reduce_edge(buffer<Scalar>& global, buffer<Scalar>& local) {
    allreduce(global);
    m_he.pull(global.host_ptr(), local.host_ptr());
    m_ve.pull(global.host_ptr() + m_n_h_glob,
              local.host_ptr() + m_e_split);
    push_dev(local);
  }

  void reduce_vertex(buffer<Scalar>& global, buffer<Scalar>& local) {
    allreduce(global);
    m_vert.pull(global.host_ptr(), local.host_ptr());
    push_dev(local);
  }

 private:
  void allreduce(buffer<Scalar>& global) {
    stage_host(global);
    MPI_Allreduce(MPI_IN_PLACE, global.host_ptr(), int(global.size()),
                  mpi_scalar_type(), MPI_SUM, m_world);
    push_dev(global);
  }

  static void stage_host(buffer<Scalar>& buf) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (buf.mem_type() != MemType::host_only) buf.copy_to_host();
#endif
  }
  static void push_dev(buffer<Scalar>& buf) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    if (buf.mem_type() != MemType::host_only) buf.copy_to_device();
#endif
  }

  // One cochain block: allgather maps (owned slots of every rank, in
  // rank order) and this rank's local->global map for pulls.
  struct block {
    int n_owned = 0;
    int local_size = 0;
    std::vector<int> counts, displs;
    std::vector<int> gidx_all;
    std::vector<Scalar> scratch;
    std::vector<int> l2g;

    void build(const distributed_cochain_layout& L, MPI_Comm comm) {
      n_owned = L.owned_size();
      local_size = L.local_size();
      int ws = 0;
      MPI_Comm_size(comm, &ws);
      counts.resize(ws);
      MPI_Allgather(&n_owned, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      displs.resize(ws);
      int total = 0;
      for (int r = 0; r < ws; ++r) {
        displs[r] = total;
        total += counts[r];
      }
      gidx_all.resize(total);
      scratch.resize(total);
      l2g.resize(local_size);
      for (int l = 0; l < local_size; ++l) l2g[l] = L.to_global(l);
      MPI_Allgatherv(l2g.data(), n_owned, MPI_INT, gidx_all.data(),
                     counts.data(), displs.data(), MPI_INT, comm);
    }

    // Owned locals are [0, n_owned) of the block in ascending-global
    // order; scatter every rank's contribution into the global block.
    void replicate(const Scalar* local_block, Scalar* global_block,
                   MPI_Comm comm) {
      MPI_Allgatherv(local_block, n_owned, mpi_scalar_type(),
                     scratch.data(), counts.data(), displs.data(),
                     mpi_scalar_type(), comm);
      for (size_t i = 0; i < gidx_all.size(); ++i) {
        global_block[gidx_all[i]] = scratch[i];
      }
    }

    void pull(const Scalar* global_block, Scalar* local_block) const {
      for (int l = 0; l < local_size; ++l) {
        local_block[l] = global_block[l2g[l]];
      }
    }
  };

  MPI_Comm m_world = MPI_COMM_NULL;
  block m_he, m_ve, m_tri, m_rect, m_vert;
  size_t m_n_h_glob = 0, m_n_tri_glob = 0;
  int m_e_split = 0, m_b_split = 0;
};

}  // namespace Aperture
