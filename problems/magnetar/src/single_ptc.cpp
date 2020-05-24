#include "framework/config.h"
#include "framework/environment.h"
// #include "systems/field_solver_sph.h"
#include "systems/ptc_updater_magnetar.h"
#include "systems/data_exporter.h"
#include "systems/rt_magnetar.h"
// #include "systems/boundary_condition.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_sph_t<Conf>>(env);
  auto pusher =
      env.register_system<ptc_updater_magnetar<Conf>>(env, *grid);
  auto rt =
      env.register_system<rt_magnetar<Conf>>(
          env, *grid);
  // auto solver =
  //     env.register_system<field_solver_sph<Conf>>(env, *grid);
  // auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);

  env.init();

  double Bp = 10000.0;
  env.params().get_value("Bp", Bp);

  // Set initial condition
  // set_initial_condition(env, *grid, 0, 1.0, Bp);
  vector_field<Conf> *B0, *B;
  particle_data_t *ptc;
  // env.get_data("B0", &B0);
  env.get_data("B", &B);
  env.get_data("particles", &ptc);

  // Set dipole initial magnetic field
  B->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // Add a single particle to the magnetosphere
  Scalar p0 = 100.0f;
  ptc->append_dev({0.5f, 0.5f, 0.0f}, {p0, 0.0f, 0.0f},
                  10 + 60 * grid->dims[0], 100.0);

  env.run();
  return 0;
}
