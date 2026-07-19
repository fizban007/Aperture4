#pragma once

#include "core/typedefs_and_constants.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_field_sync.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename ExecPolicy>
class prismatic_ptc_updater : public system_t {
 public:
  static std::string name() { return "prismatic_ptc_updater"; }

  // Phase 6: pass the solver's partition + comm to run distributed.
  // Particles are sharded by cell ownership (ico-face x radial slab);
  // the kernels stay global-indexed and consume the "E_ptc"/"B_ptc"
  // replicas maintained by prismatic_field_replicator (which MUST be
  // registered before any particle system).  Deposits go to global
  // "J_ptc"/"rho_ptc"/... replicas, then are summed across ranks and
  // pulled into the local "J"/"rho"/... consumed by the solver and the
  // sph output.  After each push, particles whose cell left this
  // rank's owned region migrate via MPI_Alltoallv.
  prismatic_ptc_updater(prismatic_mesh& mesh,
                        const prismatic_mesh_partition* mp = nullptr,
                        const prismatic_mpi_comm* comm = nullptr);
  ~prismatic_ptc_updater() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  nonown_ptr<prismatic_particle_data> particles() { return m_ptc; }

  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

  // Distributed only: move particles whose cell is owned by another
  // rank to that rank.  Collective on MPI_COMM_WORLD; called at the
  // end of every update().  Public for the CUDA extended-lambda rule.
  void migrate();

 private:
  prismatic_mesh& m_mesh;
  const prismatic_mesh_partition* m_mp = nullptr;
  const prismatic_mpi_comm* m_comm = nullptr;
  bool m_distributed = false;

  // What the kernels consume: the global fields single-rank, the
  // global replicas distributed.
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  // Local solver-facing deposit targets (distributed only).
  nonown_ptr<prismatic_edge_field> m_J_loc;
  nonown_ptr<prismatic_vertex_field> m_rho_loc;
  nonown_ptr<prismatic_vertex_field> m_rho_abs_loc;
  nonown_ptr<prismatic_vertex_field> m_gw_loc;
  // Owned by the replicator system; fetched in init().
  prismatic_field_sync* m_sync = nullptr;

  // Cell-ownership bounds (owned tris are one contiguous ico-face
  // range, owned layers one contiguous slab) and the global slab map
  // for computing destination ranks.
  int m_tri_lo = 0, m_tri_hi = 0;
  int m_layer_lo = 0, m_layer_hi = 0;
  int m_slab_base = 0, m_slab_rem = 0;
  int m_world_rank = 0, m_world_size = 1;

  // Migration scratch: per-rank counts/cursors, packed send components
  // (device-packed, host-staged through MPI), receive staging.
  buffer<int> m_mig_count, m_mig_cursor;
  buffer<Scalar> m_snd_s[8];
  buffer<uint32_t> m_snd_cell, m_snd_flag;
  buffer<uint64_t> m_snd_id;
  std::vector<Scalar> m_rcv_s[8];
  std::vector<uint32_t> m_rcv_cell, m_rcv_flag;
  std::vector<uint64_t> m_rcv_id;

  nonown_ptr<prismatic_particle_data> m_ptc;
  // Shared RNG pool (consumed by prismatic_ptc_injector, mirroring the
  // base ptc_updater's registration of "rng_states").
  nonown_ptr<rng_states_t<typename ExecPolicy::exec_tag>> m_rng_states;

  // Optional diagnostic deposits (config "deposit_diagnostics",
  // default true): |q| w  ("rho_abs", the multiplicity numerator) and
  // gamma |q| w ("gamma_wsum", for the mean Lorentz factor).
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  nonown_ptr<prismatic_vertex_field> m_gamma_wsum;
  bool m_deposit_diagnostics = true;

  // C0 second-order B-gather (see prismatic_vertex_recovery.h); the
  // primal Whitney gather pitch-angle-scatters particles off face jumps.
  // Config "use_recovery_gather" (default true) selects it; E-gather and
  // deposition always stay primal Whitney.
  prismatic_vertex_recovery m_recovery;
  bool m_use_recovery_gather = true;

  // Absorb particles beyond this radius (config "ptc_absorb_radius").
  // <= 0 (default) disables the check; particles are then only absorbed
  // implicitly at the domain edges [r_min, r_max].  Magnetosphere runs
  // should set this to the damping-layer entrance.
  Scalar m_absorb_radius = Scalar(0);

  Scalar m_charge_e = -1.0;
  Scalar m_mass_e = 1.0;
  int m_sort_interval = 100;
  bool m_use_gca = false;
  bool m_include_curvature = false;
  // Hybrid switch: Boris when omega_c dt / gamma < this (B-null regions).
  Scalar m_gca_switch_wc = Scalar(0.5);
  // Synchrotron-locking option: zero mu when (re)captured by GCA.
  bool m_gca_zero_mu = false;
};

using prismatic_ptc_updater_t = prismatic_ptc_updater<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
