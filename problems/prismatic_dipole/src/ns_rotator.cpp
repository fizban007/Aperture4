// NS rotator with plasma: the A3 magnetosphere prototype.
//
// Aligned (or oblique) rotating dipole with surface pair injection:
//   - dec_field_solver with the rotating-conductor inner BC
//     (use_deutsch_bc = false: instantaneous dipole B + corotation E)
//     and the static aligned background subtracted
//     (use_static_background = true);
//   - prismatic_ptc_updater (Boris push, recovery B-gather, Whitney
//     deposit -> J -> Ampere) with absorption at the damping-layer
//     entrance (ptc_absorb_radius);
//   - prismatic_surface_injector: neutral pairs in the first shell(s).
//
// Update order matters: inject, then push/deposit (J at t+dt/2), then
// advance the fields with that J.
//
// MPI (Phase 7C): run under mpirun with A*K ranks (config
// "n_angular_ranks" = A, default 20) for FULLY-LOCAL distributed PIC.
// One pic-depth mesh_partition bundle is built here and shared by the
// solver and every particle system; fields AND particles are local,
// deposits fold back through halo reduce(), and particles migrate by
// cell ownership after each push.  Requires >= 2 radial shells per
// slab (N_r / K >= 2).
//
// Start: vacuum aligned dipole (set_initial_dipole).  The corotation E
// spins up the magnetosphere as injected plasma fills it.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_checkpoint.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "systems/prismatic/prismatic_surface_injector.h"
#include <mpi.h>

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);  // initializes MPI

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int L = env.params().get_as<int64_t>("subdivision_level", 5);
  int N_r = env.params().get_as<int64_t>("N_r", 102);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 45.0);

  prismatic_mesh mesh;
  mesh.sphere_optimize_iters =
      env.params().get_as<int64_t>("mesh_optimize_iters", 0);
  // 7D: distributed runs never build the global 3D mesh arrays.
  if (world_size > 1) {
    mesh.build_sphere_only(L, N_r, r_min, r_max);
  } else {
    mesh.build(L, N_r, r_min, r_max);
  }
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  // Distributed setup: ONE pic-depth partition bundle, shared by the
  // solver and every particle system (plan 7C — removes the
  // two-identical-bundles risk).  All of these must outlive env.run().
  prismatic_mpi_comm mcomm;
  icosphere_topology topo;
  prismatic_partition part;
  prismatic_mesh_partition mpart;
  const prismatic_mesh_partition* mp = nullptr;
  const prismatic_mpi_comm* pc = nullptr;
  if (world_size > 1) {
    // n_angular_ranks: explicit A, or 0 = auto (the angular-major
    // suggestion for this world size and mesh — cluster-agnostic).
    int A = env.params().get_as<int64_t>("n_angular_ranks", 20);
    if (A == 0) {
      A = prismatic_partition::suggest_angular_ranks(
          world_size, mesh.m_L, mesh.m_N_r);
      Logger::print_info("ns_rotator: auto decomposition A = {} (K = {})", A,
                         A > 0 ? world_size / A : 0);
    }
    if (A < 1 || world_size % A != 0) {
      Logger::print_err(
          "ns_rotator: world size {} has no valid angular rank count "
          "(n_angular_ranks = {})",
          world_size, A);
      return 1;
    }
    // ranks_per_node > 1 tiles consecutive rank blocks into compact
    // patches of the A x K grid (see prismatic_mpi_comm) — set it to
    // the launcher's tasks-per-node for cheap intra-node halos.
    int rpn = env.params().get_as<int64_t>("ranks_per_node", 0);
    mcomm = prismatic_mpi_comm::create(MPI_COMM_WORLD, A,
                                       world_size / A, rpn);
    topo = icosphere_topology::build_from_mesh(mesh);
    part = prismatic_partition::combined(mesh.m_L, mesh.m_N_r, A,
                                         mcomm.n_radial_ranks(),
                                         mcomm.world_rank());
    part.set_topology(&topo);
    mpart = prismatic_mesh_partition::build(part, topo, halo_depth::pic);
    mp = &mpart;
    pc = &mcomm;
  }

  env.register_system<prismatic_surface_injector_t>(mesh, mp, pc);
  env.register_system<prismatic_ptc_updater_t>(mesh, mp, pc);
  auto solver = env.register_system<dec_field_solver_t>(mesh, pc, mp);
  env.register_system<prismatic_data_exporter>(mesh, mp, pc);
  // 7D: sph output is single-rank only; distributed runs post-process
  // the exporter dumps with python/sph_from_dump.py.
  if (world_size == 1) {
    env.register_system<prismatic_sph_output>(mesh);
  }
  // LAST: captures end-of-step state (config checkpoint_interval /
  // restart_from; SIGUSR1 forces a final checkpoint).
  auto ckpt = env.register_system<prismatic_checkpointer_t>(mesh, mp, pc);

  env.init();

  if (!ckpt->try_restart()) {
    solver->set_initial_dipole();
  }

  env.run();

  if (world_size > 1) {
    // The env singleton is never destroyed, so its destructor's
    // MPI_Finalize never runs; finalize here or mpirun reports an
    // abnormal termination.  mcomm's destructor safely no-ops after
    // this.
    MPI_Finalize();
  }
  return 0;
}
