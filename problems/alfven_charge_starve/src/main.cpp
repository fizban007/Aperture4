#include "framework/config.h"
#include "framework/environment.h"
#include "systems/field_solver.h"
#include "systems/ptc_updater.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/boundary_condition.h"
#include <iostream>

using namespace std;
using namespace Aperture;

template <typename Conf>
void set_initial_condition(vector_field<Conf>& B0,
                           particle_data_t& ptc,
                           int mult, Scalar weight) {
  auto& grid = B0.grid();

}

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  sim_environment env(&argc, &argv);

  env.params().add("log_level", (int64_t)LogLevel::debug);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env, comm);
  auto pusher =
      env.register_system<ptc_updater_cu<Conf>>(env, *grid, comm);
  auto solver =
      env.register_system<field_solver_cu<Conf>>(env, *grid, comm);
  // auto bc = env.register_system<boundary_condition<Conf>>(env, *grid);
  auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);

  env.init();

  vector_field<Conf>* B0;
  particle_data_t* ptc;
  env.get_data("B0", &B0);
  env.get_data("particles", &ptc);

  // set_initial_condition(*B0, *ptc, 20, 1.0);
  auto Bp = env.params().get_as<double>("Bp", 1000.0);
  auto muB = env.params().get_as<double>("muB", 1.0);
  B0->set_values(0, [Bp, muB](Scalar x, Scalar y, Scalar z) {
    return Bp * muB;
  });
  B0->set_values(1, [Bp, muB](Scalar x, Scalar y, Scalar z) {
    return Bp * math::sqrt(1.0 - muB);
  });
  pusher->fill_multiplicity(10);
  // ptc->append_dev({0.0f, 0.0f, 0.0f}, {0.0f, 100.0f, 0.0f}, 200 + 258 * grid->dims[0],
  //                 100.0);

  env.run();
  return 0;
}
