// Wald test on a rotating black hole using the GR DEC field solver.
//
// Setup: start with the Schwarzschild Wald solution (uniform B₀·ẑ, E=0)
// on a spacetime with nonzero spin a.  In the correct rotating Wald
// equilibrium, frame dragging induces an E field and a BH charge
// Q = -2 a M B₀ (Wald 1974).  Our initial condition is NOT the
// equilibrium for a ≠ 0, so the system relaxes to equilibrium by
// radiating low-frequency EM waves both into the horizon and out to
// infinity.  First-pass diagnostic: the fields should stabilize and
// not blow up.
//
// Inner BC: r_min is placed just inside the outer horizon so causal
// disconnection naturally terminates the evolution (no explicit BC
// needed).  The horizon safety damping is disabled — we rely on
// causality.
//
// Outer BC: exponential damping layer to absorb outgoing waves.

#include "framework/environment.h"
#include "systems/prismatic/dec_field_solver_gr_ks.h"
#include "systems/prismatic/prismatic_data_exporter.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_mesh_metric.h"
#include "systems/prismatic/prismatic_sph_output.h"

using namespace Aperture;

int main(int argc, char* argv[]) {
  auto& env = sim_environment::instance(&argc, &argv);

  // --- Mesh parameters ---
  int L = env.params().get_as<int64_t>("subdivision_level", 4);
  int N_r = env.params().get_as<int64_t>("N_r", 44);
  double r_min = env.params().get_as<double>("r_min", 1.0);
  double r_max = env.params().get_as<double>("r_max", 20.0);

  // --- Physics parameters ---
  double a = env.params().get_as<double>("bh_spin", 0.998);
  double B0 = env.params().get_as<double>("B0", 1.0);

  // --- Build the metric-aware mesh ---
  prismatic_mesh_metric mesh;
  mesh.build(L, N_r, r_min, r_max);
  // Copy mesh topology to device *before* compute_metric so the
  // metric evaluation and Hodge-quadrature kernels can read the
  // vertex / edge / face tables from GPU memory.
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
  mesh.copy_to_device();
#endif
  bool use_flat = env.params().get_as<bool>("use_flat_metric", false);
  if (use_flat) {
    mesh.compute_metric(flat_spherical_metric{});
  } else {
    mesh.compute_metric(ks_spherical_metric{Scalar(a)});
  }

  // --- Register systems ---
  auto solver = env.register_system<dec_field_solver_gr_ks_t>(mesh);
  env.register_system<prismatic_data_exporter>(mesh);
  env.register_system<prismatic_sph_output>(mesh);

  env.init();

  // --- Initial condition: uniform B₀·ẑ (Schwarzschild Wald solution) ---
  solver->set_initial_wald(Scalar(B0));

  env.run();
  return 0;
}
