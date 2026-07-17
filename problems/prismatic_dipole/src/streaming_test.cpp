#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "utils/util_functions.h"
#include "fill_volume.hpp"

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

  auto solver = env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  auto updater = env.register_system<prismatic_ptc_updater_t>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();
  solver->set_initial_dipole();

  // Volume-fill injection (shared helper, see fill_volume.hpp)
  if (env.params().get_as<bool>("fill_volume", false)) {
    fill_volume_radial_beam(mesh, updater);
  }

  Logger::print_info("Before run: {} particles", updater->particles()->number());

  env.run();
  return 0;
}
