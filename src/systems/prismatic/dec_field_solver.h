#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/dec_solver_dist.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_halo_exchanger.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_partition.h"
#include "systems/prismatic/prismatic_recon_hodge.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename ExecPolicy>
class dec_field_solver : public system_t {
 public:
  static std::string name() { return "dec_field_solver"; }

  // Pass a non-null, non-single-rank comm (prismatic_mpi_comm::create,
  // 20*K ranks) to run distributed: the partition is built here so that
  // register_data_components — which the framework calls at
  // register_system time — sizes the field buffers locally.  The comm
  // must outlive this system.  Under a distributed solver,
  // combined-range consumers (particles, sph output, exporter) must
  // NOT be registered — init() fatals if a "particles" data component
  // exists.
  dec_field_solver(prismatic_mesh& mesh,
                   const prismatic_mpi_comm* comm = nullptr);
  ~dec_field_solver() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Phase 5: the rank's partition bundle, for output systems that need
  // the local cochain layouts (exporter / sph output).  Null when
  // single-rank (the partition is built lazily in init() then).
  const prismatic_mesh_partition* mesh_partition() const {
    return m_distributed ? &m_mesh_part : nullptr;
  }

  // Per-rank field dump (4.1b.7 option (i)): writes this rank's OWNED
  // slots of the total E/B cochains plus the owned l2g index maps to
  // <output_dir>/rank<R>_step_<S>.h5.  Stitch with the l2g maps.
  void dump_rank_fields(uint32_t step);

  // Initial condition helpers — call from main after env.init().
  void set_initial_dipole();
  void set_initial_deutsch();

  // Spherical-cavity TE/TM eigenmode initial condition.
  //   l, m         : angular quantum numbers (l >= 1, |m| <= l)
  //   n_root       : radial root index (1 = lowest)
  //   polarization : 'E' for TE (B_r = max angular structure),
  //                  'M' for TM (E_r = max angular structure)
  //   start_with_e : if true,  E(t=0) at maximum, B(t=0) = 0;
  //                  if false, B(t=0) at maximum, E(t=0) = 0  (default)
  // The eigenvalue ω = c k is computed from r_min and r_max at runtime.
  // Reads m_resonator_amp as the overall amplitude (default 1.0).
  void set_initial_resonator_mode(int l, int m, int n_root,
                                  char polarization, bool start_with_e);

  // Public because HIP/CUDA compilers require __device__ lambdas in public methods
  void update_explicit(double dt);
  void update_semi_implicit(double dt);

  // Fill a face-flux buffer with the exact cochains of a point dipole of
  // moment (mx, my, mz) via Gauss quadrature (used for ICs and the static
  // background).
  void fill_dipole_B(buffer<Scalar>& B, Scalar mx, Scalar my, Scalar mz);

  // Recompute the total fields "E"/"B" = background + delta.  Called after
  // every update and IC; particles, sph output, and dumps consume totals.
  void refresh_total_fields();

  void compute_rhs(buffer<Scalar>& E_in, buffer<Scalar>& B_in,
                   buffer<Scalar>& dE_dt, buffer<Scalar>& dB_dt);

  void apply_damping(buffer<Scalar>& E, buffer<Scalar>& B, double dt);
  // Overwrites inner-boundary E and B with the analytic solution.  E and
  // B take separate evaluation times because the leapfrog stores B at
  // half-steps (t + dt/2 after the Faraday update, i.e. dt/2 BEHIND the
  // end-of-step time): the explicit path passes time_B = time_E - dt/2,
  // while the (co-located) semi-implicit path passes time_B = time_E.
  // Evaluating both at the integer time injects an O(dt) boundary error
  // (found by the A2.0 Deutsch dt-halving study).
  void apply_inner_bc(buffer<Scalar>& E, buffer<Scalar>& B, double time_E,
                      double time_B);

  // PEC (perfect conductor) boundary on inner and outer shells:
  // zero tangential E (horizontal edges) and normal B (triangular faces)
  // on shells k = 0 and k = N_r.
  void apply_pec_bc(buffer<Scalar>& E, buffer<Scalar>& B);

 private:

  prismatic_mesh& m_mesh;

  // 4.1b distributed core: all step kernels (Faraday, Ampere, RHS,
  // damping, PEC / inner BC) and the dipole/Deutsch ICs run through
  // dec_solver_dist over the partition's local d1/d1^T blocks.  With no
  // comm (single-rank) the partition is single_rank, whose local
  // layouts are identity maps onto the global ordering — the registered
  // field buffers pass straight through, bit-compatible with the old
  // global kernels.  With a comm the partition is the rank's slice of
  // the combined 20xK decomposition and the fields are local-sized.
  icosphere_topology m_topo;
  prismatic_partition m_part;
  prismatic_mesh_partition m_mesh_part;
  dec_solver_dist<ExecPolicy> m_dist;

  // Distributed mode (ctor comm): real partition + MPI halo exchanges
  // at the sync points.  m_ex is inactive (all no-ops) when single-rank.
  const prismatic_mpi_comm* m_mpi = nullptr;
  bool m_distributed = false;
  prismatic_halo_exchanger<ExecPolicy> m_ex;
  // Per-rank dump cadence under distributed runs (config
  // "rank_dump_interval", 0 = off).
  int m_rank_dump_interval = 0;
  std::string m_output_dir = "Data/";

  // Shared field data (owned by env, found in register_data_components).
  // Mirrors the main-code background split: the solver evolves the delta
  // fields ("Edelta"/"Bdelta"); "E"/"B" hold background + delta and are
  // what the particle updater, sph output, and exporter consume.  The
  // static background ("E0"/"B0") is exempt from the discrete curl, so
  // its quasi-static Hodge truncation (the O(h) tier) never enters the
  // evolution.  E0 is registered for symmetry but stays zero: the only
  // supported background is a static magnetic field.
  nonown_ptr<prismatic_edge_field> m_E;   // Edelta (evolved)
  nonown_ptr<prismatic_face_field> m_B;   // Bdelta (evolved)
  nonown_ptr<prismatic_edge_field> m_Etotal;
  nonown_ptr<prismatic_face_field> m_Btotal;
  nonown_ptr<prismatic_edge_field> m_E0;
  nonown_ptr<prismatic_face_field> m_B0;
  nonown_ptr<prismatic_edge_field> m_J;

  // Temporary buffers for semi-implicit iteration (owned by solver)
  buffer<Scalar> m_tmp_E, m_tmp_B;
  buffer<Scalar> m_dE_dt, m_dB_dt;
  buffer<Scalar> m_dE_dt_new, m_dB_dt_new;

  // Physics parameters
  Scalar m_Bp = 1.0;
  Scalar m_Omega = 1.0;
  Scalar m_obliquity = 0.0;

  // Damping layer: sigma(k) = damping_coef * ramp^damping_exponent, with
  // ramp rising linearly 0 -> 1 across the last damping_length shells.
  int m_damping_length = 10;
  Scalar m_damping_coef = 0.05;
  Scalar m_damping_exponent = 1.0;

  // Update toggles
  bool m_update_e = true;
  bool m_update_b = true;

  // Semi-implicit parameters
  bool m_use_implicit = false;
  Scalar m_beta = 0.55;
  int m_implicit_iters = 4;

  // Use full Deutsch retarded fields for inner BC (for convergence testing)
  bool m_use_deutsch_bc = false;

  // When false, apply_inner_bc drives tangential E only (standard
  // rotating-conductor BC) instead of also overwriting B on the ring.
  bool m_inner_bc_overwrite_b = true;

  // Reconstruction-corrected Hodge for the Ampere constitutive chain
  // (config "use_reconstruction_hodge", default false): 2nd-order
  // consistent (see prismatic_recon_hodge.h), explicit ~30-entry rows,
  // boundary shells keep the diagonal.  Explicit stepping only.
  bool m_use_recon_hodge = false;
  prismatic_recon_hodge m_recon_hodge;

  // Use PEC (perfect conductor) boundary instead of dipole/Deutsch BC.
  // When true, apply_pec_bc() is called each step instead of apply_inner_bc().
  bool m_use_pec_bc = false;

  // Resonator mode amplitude (used by set_initial_resonator_mode)
  Scalar m_resonator_amp = 1.0;

  // Static background subtraction (config "use_static_background",
  // default false = legacy behavior with zero background).  When true,
  // B0 is filled with the ALIGNED static dipole component
  // (0, 0, Bp cos(obliquity)) — the only obliquity-safe static choice;
  // time-dependent backgrounds are deliberately unsupported (a rigidly
  // rotating dipole is not a Maxwell solution and would delete
  // retardation physics from the delta equations).
  bool m_use_static_background = false;

  double m_time = 0.0;
};

using dec_field_solver_t = dec_field_solver<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
