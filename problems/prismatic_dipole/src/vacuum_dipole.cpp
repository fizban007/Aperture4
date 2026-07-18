// Vacuum rotating oblique dipole: DEC field solver only, no particles.
// Tests electromagnetic radiation from a rotating inclined magnetic dipole.
//
// MPI (4.1b B2): run under mpirun with 20*K ranks for the distributed
// solver — fields are local-sized, halo exchanges happen inside the
// solver, and output switches to per-rank dumps controlled by
// "rank_dump_interval" (the combined-range exporter / sph systems are
// single-rank only and are not registered).  Single-process runs are
// unchanged.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include <mpi.h>

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);  // initializes MPI

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int L = env.params().get_as<int64_t>("subdivision_level", 3);
  int N_r = env.params().get_as<int64_t>("N_r", 50);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 20.0);
  bool use_deutsch_ic = env.params().get_as<bool>("use_deutsch_ic", false);

  prismatic_mesh mesh;
  mesh.sphere_optimize_iters =
      env.params().get_as<int64_t>("mesh_optimize_iters", 0);
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  prismatic_mpi_comm mcomm;  // must outlive env.run()
  if (world_size > 1) {
    if (world_size % 20 != 0) {
      Logger::print_err("vacuum_dipole: MPI runs need 20*K ranks (got {})",
                        world_size);
      return 1;
    }
    mcomm = prismatic_mpi_comm::create(MPI_COMM_WORLD, world_size / 20);
  }

  auto solver = env.register_system<dec_field_solver_t>(
      mesh, world_size > 1 ? &mcomm : nullptr);
  if (world_size == 1) {
    env.register_system<prismatic_data_exporter>(mesh);
    env.register_system<prismatic_sph_output>(mesh);
  }

  env.init();

  if (use_deutsch_ic) {
    solver->set_initial_deutsch();
  } else {
    solver->set_initial_dipole();
  }

  env.run();

  if (world_size > 1) {
    // The env singleton is never destroyed, so its destructor's
    // MPI_Finalize never runs; finalize here or mpirun reports an
    // abnormal termination.  mcomm's destructor safely no-ops after
    // finalize.
    MPI_Finalize();
  }
  return 0;
}
