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
// Start: vacuum aligned dipole (set_initial_dipole).  The corotation E
// spins up the magnetosphere as injected plasma fills it.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "systems/prismatic/prismatic_surface_injector.h"

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

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

  env.register_system<prismatic_surface_injector_t>(mesh);
  env.register_system<prismatic_ptc_updater_t>(mesh);
  auto solver = env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();

  solver->set_initial_dipole();

  env.run();
  return 0;
}
