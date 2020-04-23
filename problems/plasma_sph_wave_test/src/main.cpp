#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/field_solver_logsph.h"
#include "systems/ptc_updater_logsph.h"
#include "systems/data_exporter.h"
#include "systems/boundary_condition.hpp"
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
  // auto solver =
  //     env.register_system<field_solver_logsph<Conf>>(env, *grid);
  auto pusher =
      env.register_system<ptc_updater_logsph_cu<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);
  // auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);

  env.init();

  // Set initial condition
  vector_field<Conf> *B, *B0;
  particle_data_t *ptc;
  env.get_data("B", &B);
  env.get_data("particles", &ptc);
  // env.get_data("B0", &B0);

  double Bp = 10000.0;

  B->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = std::exp(x);
    // return Bp / (r * r);
    return Bp * sin(theta) / cube(r);
  });
  ptc->append_dev({0.0f, 0.0f, 0.0f},
                  {10.0f, 0.0f, 0.0f},
                  grid->get_idx(10, 100).linear);
  // B0->copy_from(*B);

  env.run();
  return 0;
}
