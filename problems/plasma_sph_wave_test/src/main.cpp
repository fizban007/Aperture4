#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/field_solver_logsph.h"
#include "systems/ptc_updater_logsph.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.hpp"
#include "systems/initial_condition.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_logsph_t<Conf>>(env);
  // auto pusher =
  //     env.register_system<ptc_updater_logsph_cu<Conf>>(env, *grid);
  auto solver =
      env.register_system<field_solver_logsph<Conf>>(env, *grid);
  auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);
  // auto ic = env.register_system<initial_condition<Conf>>(env, *grid, 2, 1.0);

  env.init();

  set_initial_condition(env, *grid, 0, 1.0, 1000.0);

  env.run();
  return 0;
}
