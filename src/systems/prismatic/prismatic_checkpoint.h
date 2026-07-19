#pragma once

#include "core/typedefs_and_constants.h"
#include "data/rng_states.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_owned_runs.h"
#include "utils/nonown_ptr.hpp"
#include <string>
#include <vector>

namespace Aperture {

template <typename ExecPolicy>
class prismatic_ptc_updater;
template <typename ExecPolicy>
class dec_field_solver;

// =========================================================================
// Checkpoint / restart for the prismatic drivers (see
// CHECKPOINT_RESTART_PLAN.md; design D1–D5 agreed 2026-07-19).
//
// Format: rank-agnostic, GLOBAL-indexed, one HDF5 file per generation
// under <checkpoint_dir>/ckpt_<resume_step>/checkpoint.h5.  Cochains
// are stored as the global datasets (owned-runs collective writes, the
// exporter's snapshot pattern); particles as one concatenated dataset
// of live macros with GLOBAL cells stored as uint64 (the uint32 wall
// sits at ~L9, below the L10 target); rng streams and id counters per
// WRITING rank.  Everything derived (mesh, E0/B0, totals, layouts, Bv)
// is recomputed at restart — which is what buys restart at a DIFFERENT
// rank count than the writer.
//
// Usage (mains):
//   auto ckpt = env.register_system<prismatic_checkpointer_t>(mesh, mp, pc);
//   ...register everything else BEFORE it; the checkpointer must be
//   LAST so update() captures the end-of-step state...
//   env.init();
//   if (!ckpt->try_restart()) solver->set_initial_dipole();
//
// Config:
//   checkpoint_interval  steps between checkpoints (0 = off, default)
//   checkpoint_dir       default "<output_dir>/ckpt"
//   checkpoint_keep      generations kept (default 2)
//   restart_from         "" (fresh start), "auto" (newest complete
//                        generation in checkpoint_dir), or an explicit
//                        generation directory
//
// Crash safety (D5): a generation is written to <dir>/tmp/ and
// atomically renamed to <dir>/ckpt_<step>/ after a successful close;
// the oldest generation is deleted only after the rename.  A SIGKILL
// mid-write can never destroy the last good checkpoint.  The SIGUSR1
// graceful-stop hook (sim_environment::register_force_snapshot) is
// wired in init(): kill -USR1 writes a final checkpoint and exits.
// =========================================================================
template <typename ExecPolicy>
class prismatic_checkpointer : public system_t {
 public:
  static std::string name() { return "prismatic_checkpointer"; }

  prismatic_checkpointer(const prismatic_mesh& mesh,
                         const prismatic_mesh_partition* mp = nullptr,
                         const prismatic_mpi_comm* comm = nullptr);
  ~prismatic_checkpointer() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Write a checkpoint capturing the current state, resumable at
  // (resume_step, resume_time) — for the end of step s these are
  // (s + 1, (s + 1) * dt).  Collective.
  void write_checkpoint(uint32_t resume_step, double resume_time);

  // Restart from the generation selected by config "restart_from" (or
  // the --restart command-line flag, which takes precedence).  Call
  // from main AFTER env.init() and INSTEAD of the initial-condition
  // call; returns false when no restart was requested (fresh start).
  // Aborts loudly on a fingerprint mismatch or an invalid generation.
  bool try_restart();

 private:
  void load_generation(const std::string& gen_dir);
  // Rank-0 helpers (filesystem side of D5).
  std::string find_latest_generation() const;
  void rotate_generations(uint32_t resume_step);

  const prismatic_mesh& m_mesh;
  const prismatic_mesh_partition* m_mp = nullptr;
  const prismatic_mpi_comm* m_comm = nullptr;
  bool m_distributed = false;

  // Fetched by name in init() — the checkpointer never registers field
  // data (shared solver/updater/exporter components; re-registering
  // with different sizing is the known framework mine).
  nonown_ptr<prismatic_edge_field> m_Edelta;
  nonown_ptr<prismatic_face_field> m_Bdelta;
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  nonown_ptr<prismatic_vertex_field> m_gamma_wsum;
  nonown_ptr<prismatic_particle_data> m_ptc;
  nonown_ptr<rng_states_t<typename ExecPolicy::exec_tag>> m_rng;
  prismatic_ptc_updater<ExecPolicy>* m_updater = nullptr;
  dec_field_solver<ExecPolicy>* m_solver = nullptr;

  // Owned runs for the global cochain datasets (exporter pattern).
  prismatic_run_set m_E_runs, m_B_runs, m_V_runs;

  int m_interval = 0;
  int m_keep = 2;
  std::string m_dir;
  std::string m_restart_from;
  // Config fingerprint stored with every generation and validated
  // loudly on load.
  double m_r_min = 0, m_r_max = 0, m_dt = 0;

  int m_world_rank = 0, m_world_size = 1;
  double m_time = 0.0;
};

using prismatic_checkpointer_t =
    prismatic_checkpointer<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
