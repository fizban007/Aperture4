#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/field_solver_logsph.h"
// #include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_logsph_t<Conf>>(env, *comm);
  auto solver =
      env.register_system<field_solver_logsph<Conf>>(env, *grid, comm);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);
  auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);

  env.init();

  // Set initial condition
  vector_field<Conf> *B, *B0;
  env.get_data("B", &B);
  env.get_data("B0", &B0);
  if (B == nullptr)
    Logger::print_err("B is nullptr!!!");

  double Bp = 100.0;

  B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    return Bp / (r * r);
    // return Bp * 2.0 * cos(theta) / (r * r * r);
  });
  // B0->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
  //   Scalar r = std::exp(x);
  //   // return Bp / (r * r);
  //   return Bp * sin(theta) / (r * r * r);
  // });
  B->copy_from(*B0);

  env.run();
  return 0;
}
