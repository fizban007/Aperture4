// Vacuum rotating oblique dipole: DEC field solver only, no particles.
// Tests electromagnetic radiation from a rotating inclined magnetic dipole.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_sph_output.h"

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

  int L = env.params().get_as<int64_t>("subdivision_level", 3);
  int N_r = env.params().get_as<int64_t>("N_r", 50);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 20.0);

  prismatic_mesh mesh;
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();
  env.run();
  return 0;
}
