#pragma once

#include "framework/environment.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_field_sync.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "utils/logger.h"

namespace Aperture {

// =========================================================================
// Phase 6 — global E/B replicas for the particle path.
//
// Registers "E_ptc"/"B_ptc" (global-sized) and refreshes them from the
// solver's local "E"/"B" totals at the START of every step, so the
// particle-side consumers (injectors, prismatic_ptc_updater) see
// exactly what a single-rank run would: the totals from the end of the
// previous step (or the initial condition on step 0 — ICs are set in
// main before env.run()).
//
// MUST be registered FIRST in a distributed PIC main:
//   replicator -> injector(s) -> ptc_updater -> dec_field_solver -> output
// It also registers the local-sized "E"/"B"/"J", making registration
// robust to system order with respect to the solver.
//
// Also owns the prismatic_field_sync instance; prismatic_ptc_updater
// fetches it (env.get_system) for the J/rho deposit reductions.
// =========================================================================
template <typename ExecPolicy>
class prismatic_field_replicator : public system_t {
 public:
  static std::string name() { return "prismatic_field_replicator"; }

  prismatic_field_replicator(prismatic_mesh& mesh,
                             const prismatic_mesh_partition* mp,
                             const prismatic_mpi_comm* comm)
      : m_mesh(mesh), m_mp(mp), m_comm(comm) {
    if (mp == nullptr || comm == nullptr || comm->is_single_rank()) {
      Logger::print_err(
          "prismatic_field_replicator requires a distributed partition; "
          "do not register it in single-rank runs");
      std::abort();
    }
  }

  void register_data_components() override {
    auto mem = ExecPolicy::data_mem_type();
    // Local solver-facing fields (idempotent with the solver's own
    // registration; whichever registers first creates them).
    m_E_loc = sim_env().template register_data<prismatic_edge_field>(
        "E", *m_mp, mem);
    m_B_loc = sim_env().template register_data<prismatic_face_field>(
        "B", *m_mp, mem);
    // Global replicas for the particle-side consumers.
    m_E_rep = sim_env().template register_data<prismatic_edge_field>(
        "E_ptc", m_mesh, mem);
    m_B_rep = sim_env().template register_data<prismatic_face_field>(
        "B_ptc", m_mesh, mem);
    m_E_rep->skip_output(true);
    m_B_rep->skip_output(true);
  }

  void init() override {
    m_sync.build(*m_mp, MPI_COMM_WORLD);
    Logger::print_info("Field replicator: global E/B replicas ({} + {})",
                       m_mesh.m_N_edges, m_mesh.m_N_faces);
  }

  void update(double dt, uint32_t step) override {
    m_sync.replicate_edge(m_E_loc->data(), m_E_rep->data());
    m_sync.replicate_face(m_B_loc->data(), m_B_rep->data());
  }

  prismatic_field_sync& sync() { return m_sync; }

 private:
  prismatic_mesh& m_mesh;
  const prismatic_mesh_partition* m_mp = nullptr;
  const prismatic_mpi_comm* m_comm = nullptr;
  prismatic_field_sync m_sync;

  nonown_ptr<prismatic_edge_field> m_E_loc;
  nonown_ptr<prismatic_face_field> m_B_loc;
  nonown_ptr<prismatic_edge_field> m_E_rep;
  nonown_ptr<prismatic_face_field> m_B_rep;
};

using prismatic_field_replicator_t =
    prismatic_field_replicator<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
