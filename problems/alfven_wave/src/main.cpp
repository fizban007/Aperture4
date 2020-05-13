#include "framework/config.h"
#include "framework/environment.h"
#include "systems/field_solver_sph.h"
#include "systems/ptc_updater_sph.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.hpp"
#include "systems/initial_condition.h"
#include "systems/ptc_injector_mult.h"
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
      env.register_system<ptc_updater_sph_cu<Conf>>(env, *grid);
  auto solver =
      env.register_system<field_solver_sph<Conf>>(env, *grid);
  auto injector =
      env.register_system<ptc_injector_mult<Conf>>(env, *grid);
  auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);

  env.init();

  set_initial_condition(env, *grid, 20, 1.0);

  env.run();
  return 0;
}
