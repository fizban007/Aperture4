// Vacuum rotating oblique dipole: DEC field solver only, no particles.
// Tests electromagnetic radiation from a rotating inclined magnetic dipole.
//
// MPI (4.1b B2 + Phase 5): run under mpirun with 20*K ranks for the
// distributed solver — fields are local-sized and halo exchanges happen
// inside the solver.  The exporter writes single global snapshot files
// collectively (parallel HDF5), and the sph output gathers to rank 0;
// both are bit-identical to the single-process files.  Per-rank raw
// dumps remain available via "rank_dump_interval".

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
  // Phase 5: the output systems run in both modes.  Under MPI they take
  // the solver's partition (solver registered first — field components
  // must be local-sized).
  env.register_system<prismatic_data_exporter>(
      mesh, solver->mesh_partition(), world_size > 1 ? &mcomm : nullptr);
  env.register_system<prismatic_sph_output>(
      mesh, solver->mesh_partition(), world_size > 1 ? &mcomm : nullptr);

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
