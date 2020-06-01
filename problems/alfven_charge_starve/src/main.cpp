#include "framework/config.h"
#include "framework/environment.h"
#include "systems/field_solver.h"
#include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.h"
#include <iostream>

using namespace std;
using namespace Aperture;

template <typename Conf>
void set_initial_condition(vector_field<Conf>& B0, int mult, Scalar weight) {

}

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env);
  auto pusher =
      env.register_system<ptc_updater_cu<Conf>>(env, *grid);
  auto solver =
      env.register_system<field_solver_cu<Conf>>(env, *grid);
  auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);

  env.init();

  vector_field<Conf>* B0;
  env.get_data("B0", &B0);

  set_initial_condition(*B0, 20, 1.0);


  env.run();
  return 0;
}
