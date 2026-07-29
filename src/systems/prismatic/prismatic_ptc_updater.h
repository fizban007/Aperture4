#pragma once

#include "core/typedefs_and_constants.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_halo_exchanger.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_ptc_mesh_local.h"
#include "systems/prismatic/prismatic_vertex_recovery.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

// =========================================================================
// Phase 7C — fully-local particle updater.
//
// The particle kernels run on a prismatic_ptc_mesh_local: local sphere
// tables, tensor→layout maps into the SAME local field buffers the
// solver uses ("E"/"B" totals in, "J"/"rho"/... deposits out), and
// LOCAL particle cells.  Single-rank runs use the identity partition
// (identity tables and maps — bit-exact with the old global path);
// distributed runs REQUIRE a canonical A·K comm and a pic-depth
// mesh_partition bundle shared with the field solver.
//
// Distributed step structure:
//   sync_fields(step)   — exchange E/B pic halos, refresh the recovery
//                         Bv on owned vertex slots and exchange it as 3
//                         scalar vertex halos.  Collective + idempotent
//                         per step: the injector (which runs BEFORE the
//                         updater but reads the same fields) calls it
//                         too, whoever comes first does the work.
//   clear local J/rho/rho_abs/gamma_wsum (owned + ghost slots)
//   push + Whitney deposit (local kernels, ghost slots absorb the
//                         off-rank stencil ends)
//   reduce() J/rho/...  — ghost deposits fold into their owners
//                         (radial→angular relay for corners);
//                         J and rho_abs are then re-EXCHANGED so the
//                         next step's injector criteria see owner-summed
//                         values in their ghost stencil slots.
//   migrate()           — leavers to the owning world rank
//                         (rank-agnostic GLOBAL cells on the wire)
//   sort by local cell.
// =========================================================================
template <typename ExecPolicy>
class prismatic_ptc_updater : public system_t {
 public:
  static std::string name() { return "prismatic_ptc_updater"; }

  prismatic_ptc_updater(prismatic_mesh& mesh,
                        const prismatic_mesh_partition* mp = nullptr,
                        const prismatic_mpi_comm* comm = nullptr);
  ~prismatic_ptc_updater() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  nonown_ptr<prismatic_particle_data> particles() { return m_ptc; }

  // The local particle mesh (identity when single-rank).  Valid after
  // init(); shared by the injectors.
  const prismatic_ptc_mesh_local& ptc_mesh() const { return m_lmesh; }

  // Field sync point (see class comment).  Collective when distributed;
  // idempotent per step.  Public so the injector can pull it forward.
  void sync_fields(uint32_t step);

  int add_particle(Scalar x, Scalar y, Scalar z,
                   Scalar px, Scalar py, Scalar pz,
                   Scalar weight, uint32_t flag = 0);

  // Distributed only: move particles whose cell is owned by another
  // rank to that rank.  Collective on MPI_COMM_WORLD; called at the
  // end of every update().  Public for the CUDA extended-lambda rule.
  void migrate();

  // -----------------------------------------------------------------------
  // Restart support (checkpoint plan D3).
  //
  // inject_wire_particles: route particles carried with GLOBAL cells
  // (the rank-agnostic migration wire encoding, widened to uint64 on
  // disk) to their owning ranks and append them there — the reading
  // distribution is arbitrary, so a checkpoint restarts at ANY rank
  // count.  Destinations come from pure arithmetic on the global cell
  // (unit path ordering × radial slab map = migrate_dest's math on
  // global ids); the exchange and arrival unpack are the migration
  // machinery.  Collective on the logical world comm when distributed;
  // single-rank appends directly.  Component order: x1,x2,x3,p1,p2,p3,
  // E,weight.  Aborts loudly on particle-buffer overflow.
  void inject_wire_particles(const std::vector<Scalar> comps[8],
                             const std::vector<uint64_t>& gcells,
                             const std::vector<uint32_t>& flags,
                             const std::vector<uint64_t>& ids);

  // refresh_deposit_ghosts: refresh the ghost slots of the deposit
  // fields whose owned slots were loaded from a checkpoint (the
  // next-step injector criteria read J / rho_abs through their
  // stencils).  Collective; no-op single-rank.
  void refresh_deposit_ghosts();

 private:
  // Shared migration/restart wire machinery: component-wise Alltoallv
  // of the packed send arrays into the m_rcv_* staging (returns the
  // arrival count), and arrival append with GLOBAL→local cell
  // translation (m_rcv_cell holds uint64 wire cells on entry — the
  // global cell space passes 2^32 near L9).
  int exchange_wire(const Scalar* const comps[8], const uint64_t* cells,
                    const uint32_t* flags, const uint64_t* ids,
                    const std::vector<int>& snd_cnt,
                    const std::vector<int>& snd_off);
  void append_wire_arrivals(int n_recv);
  // Parse and validate the synchrotron cooling knobs, and set
  // m_sync_cool_coef.  Aborts loudly on an on-but-unset configuration.
  void init_sync_cooling();
  // Parse the hybrid-switch rate, abort on the removed per-step key, and
  // check that Boris can actually resolve gyrations at the switch.
  void init_gca_switch();
  prismatic_mesh& m_mesh;
  const prismatic_mesh_partition* m_mp = nullptr;
  const prismatic_mpi_comm* m_comm = nullptr;
  bool m_distributed = false;

  // Identity bundle for single-rank runs (built in init).
  icosphere_topology m_topo_own;
  prismatic_mesh_partition m_mp_own;

  prismatic_ptc_mesh_local m_lmesh;
  prismatic_halo_exchanger<ExecPolicy> m_ex;

  // Local field state: solver totals in, deposits out (all local-sized
  // under a partition; global-sized single-rank — same combined
  // layouts either way).
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  nonown_ptr<prismatic_vertex_field> m_gamma_wsum;

  int m_world_rank = 0, m_world_size = 1;
  uint32_t m_synced_step = uint32_t(-1);

  // Migration scratch: per-rank counts/cursors, packed send components
  // (device-packed, host-staged through MPI), receive staging.  Cells
  // travel in the 64-bit GLOBAL wire encoding.
  buffer<int> m_mig_count, m_mig_cursor;
  buffer<Scalar> m_snd_s[8];
  buffer<uint64_t> m_snd_cell;
  buffer<uint32_t> m_snd_flag;
  buffer<uint64_t> m_snd_id;
  std::vector<Scalar> m_rcv_s[8];
  std::vector<uint64_t> m_rcv_cell;
  std::vector<uint32_t> m_rcv_flag;
  std::vector<uint64_t> m_rcv_id;

  nonown_ptr<prismatic_particle_data> m_ptc;
  // Shared RNG pool (consumed by prismatic_ptc_injector, mirroring the
  // base ptc_updater's registration of "rng_states").
  nonown_ptr<rng_states_t<typename ExecPolicy::exec_tag>> m_rng_states;

  // Optional diagnostic deposits (config "deposit_diagnostics",
  // default true): |q| w  ("rho_abs", the multiplicity numerator) and
  // gamma |q| w ("gamma_wsum", for the mean Lorentz factor).
  bool m_deposit_diagnostics = true;

  // C0 second-order B-gather (see prismatic_vertex_recovery.h); the
  // primal Whitney gather pitch-angle-scatters particles off face jumps.
  // Config "use_recovery_gather" (default true) selects it; E-gather and
  // deposition always stay primal Whitney.  Weights are built globally
  // (sphere data is replicated); the per-vertex Bv lives in the LOCAL
  // vertex layout, 3 components with stride = layout size.
  prismatic_vertex_recovery m_recovery;
  buffer<Scalar> m_Bv;
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
  // Hybrid switch: Boris when the gyro-frequency omega_c / gamma falls
  // below this (B-null regions).  Config "gca_switch_omegac", a RATE in
  // inverse time units -- NOT the old per-step "gca_switch_omegac_dt",
  // whose physical switching surface moved with dt (that key now aborts
  // at init; see init_gca_switch()).
  Scalar m_gca_switch_omegac = Scalar(20);
  // Synchrotron-locking option: zero mu when (re)captured by GCA.
  bool m_gca_zero_mu = false;

  // Landau-Lifshitz synchrotron drag on the Boris branch (config
  // "use_sync_cooling").  The coefficient is a RATE coefficient in
  // inverse time units -- deliberately not per-step, so the cooling
  // physics is the same at every dt and refinement level.  Set either
  // from "sync_gamma_rad" anchored at "sync_cool_b_lc", or directly by
  // "sync_cooling_coef" which overrides it.  0 disables the drag.
  Scalar m_sync_cool_coef = Scalar(0);

  // 7E scaling harness: per-phase wall-time accumulators, reported as
  // min/mean/max across ranks every `step_timer_interval` steps
  // (config; 0 = off).  Feeds the deferred measurement campaign on any
  // machine — grep "step timing".
  int m_timer_interval = 0;
  double m_t_sync = 0, m_t_push = 0, m_t_reduce = 0, m_t_migrate = 0,
         m_t_sort = 0;
  void report_timers(uint32_t step);
};

using prismatic_ptc_updater_t = prismatic_ptc_updater<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
