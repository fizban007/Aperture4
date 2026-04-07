// Spherical cavity TE/TM eigenmode test for the prismatic DEC field solver.
//
// Initializes a single (l, m, n_root) eigenmode of the spherical resonator
// formed by perfectly conducting spheres at r = r_min and r = r_max, then
// evolves under PEC boundary conditions. Used for vacuum-Maxwell convergence
// testing: at integer multiples of the period, the solution should equal
// the analytic mode at t = 0.
//
// All physics parameters are read from the config file. Particles are not
// instantiated. The default phase choice puts B at maximum and E at zero
// at t = 0; set `start_with_e = true` to swap.

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
  double r_max = env.params().get_as<double>("r_max", 5.0);

  // Mode parameters
  int mode_l = env.params().get_as<int64_t>("mode_l", 1);
  int mode_m = env.params().get_as<int64_t>("mode_m", 0);
  int mode_n_root = env.params().get_as<int64_t>("mode_n_root", 1);
  std::string mode_pol = env.params().get_as<std::string>("mode_polarization",
                                                          std::string("E"));
  bool start_with_e = env.params().get_as<bool>("start_with_e", false);

  prismatic_mesh mesh;
  mesh.build(L, N_r, r_min, r_max);
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif

  auto solver = env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();

  char polarization = mode_pol.empty() ? 'E' : mode_pol[0];
  solver->set_initial_resonator_mode(mode_l, mode_m, mode_n_root,
                                      polarization, start_with_e);

  env.run();
  return 0;
}
