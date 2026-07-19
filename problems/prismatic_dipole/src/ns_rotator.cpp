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
// MPI (Phase 6): run under mpirun with 20*K ranks for distributed PIC.
// The field solver runs on local cochains with halo exchanges; the
// particle path runs on global field replicas (replicated at the start
// of each step by prismatic_field_replicator, which must come first)
// with particles sharded by cell ownership, deposits summed across
// ranks, and particles migrating between ranks after each push.
//
// Start: vacuum aligned dipole (set_initial_dipole).  The corotation E
// spins up the magnetosphere as injected plasma fills it.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/icosphere_topology.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_field_replicator.h"
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
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  // Distributed setup: the particle systems get their own partition
  // bundle, built here so registration order (= update order) can stay
  // physical.  The solver builds an identical bundle internally.
  // All of these must outlive env.run().
  prismatic_mpi_comm mcomm;
  icosphere_topology topo;
  prismatic_partition part;
  prismatic_mesh_partition mpart;
  const prismatic_mesh_partition* mp = nullptr;
  const prismatic_mpi_comm* pc = nullptr;
  if (world_size > 1) {
    if (world_size % 20 != 0) {
      Logger::print_err("ns_rotator: MPI runs need 20*K ranks (got {})",
                        world_size);
      return 1;
    }
    mcomm = prismatic_mpi_comm::create(MPI_COMM_WORLD, world_size / 20);
    topo = icosphere_topology::build_from_mesh(mesh);
    part = prismatic_partition::combined_ico_face(mesh.m_L, mesh.m_N_r,
                                         mcomm.n_radial_ranks(),
                                         mcomm.radial_rank(),
                                         mcomm.angular_rank());
    part.set_topology(&topo);
    mpart = prismatic_mesh_partition::build(part, topo);
    mp = &mpart;
    pc = &mcomm;
    env.register_system<prismatic_field_replicator_t>(mesh, mp, pc);
  }

  env.register_system<prismatic_surface_injector_t>(mesh, mp, pc);
  env.register_system<prismatic_ptc_updater_t>(mesh, mp, pc);
  auto solver = env.register_system<dec_field_solver_t>(mesh, pc);
  env.register_system<prismatic_data_exporter>(mesh, mp, pc);
  env.register_system<prismatic_sph_output>(mesh, mp, pc);

  env.init();

  solver->set_initial_dipole();

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
