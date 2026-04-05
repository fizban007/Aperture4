#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_ptc_updater.h"
#include "systems/prismatic/prismatic_sph_output.h"
#include "utils/util_functions.h"

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

  // Systems find shared data (E, B, J, rho, particles) from env
  env.register_system<dec_field_solver_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  auto updater = env.register_system<prismatic_ptc_updater>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();

  // Inject particles from config
  int n_inject = env.params().get_as<int64_t>("n_inject", 0);
  if (n_inject > 0) {
    double ptc_r = env.params().get_as<double>("ptc_r", 1.5);
    double ptc_pr = env.params().get_as<double>("ptc_pr", 0.1);
    double ptc_theta = env.params().get_as<double>("ptc_theta", 1.5708);
    double ptc_phi = env.params().get_as<double>("ptc_phi", 0.0);
    double ptc_weight = env.params().get_as<double>("ptc_weight", 1.0);

    Scalar x = ptc_r * std::sin(ptc_theta) * std::cos(ptc_phi);
    Scalar y = ptc_r * std::sin(ptc_theta) * std::sin(ptc_phi);
    Scalar z = ptc_r * std::cos(ptc_theta);
    Scalar r_hat_x = x/ptc_r, r_hat_y = y/ptc_r, r_hat_z = z/ptc_r;

    for (int i = 0; i < n_inject; i++) {
      updater->add_particle(x, y, z,
                            ptc_pr*r_hat_x, ptc_pr*r_hat_y, ptc_pr*r_hat_z,
                            ptc_weight, gen_ptc_type_flag(PtcType::electron));
    }
    Logger::print_info("Injected {} electrons at r={}, theta={}, pr={}",
                       n_inject, ptc_r, ptc_theta, ptc_pr);
  }

  env.run();
  return 0;
}
