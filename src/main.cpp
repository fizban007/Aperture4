#include "framework/config.h"
#include "framework/environment.h"
#include "systems/field_solver_default.h"
#include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env, *comm);
  auto solver =
      env.register_system<field_solver_default<Conf>>(env, *grid, comm);
  auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);

  env.init();
  env.run();
  return 0;
}
