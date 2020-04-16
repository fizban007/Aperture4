#include "catch.hpp"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "systems/ptc_updater.h"

using namespace Aperture;

TEST_CASE("Boris push in a uniform B field", "[boris][.]") {
  Logger::init(0, LogLevel::debug);
  typedef Config<2> Conf;
  sim_environment env;
  env.params().add("log_level", 2l);
  env.params().add("N", std::vector<int64_t>({64, 64, 64}));
  env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  env.params().add("size", std::vector<double>({1.0, 1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));

  auto ptc = env.register_data<particle_data_t>("particles", 10000,
                                                MemType::host_device);

  auto comm = env.register_system<domain_comm<Conf>>(env);
  auto grid = env.register_system<grid_t<Conf>>(env, *comm);
  auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid, *comm);

  env.init();

  std::shared_ptr<vector_field<Conf>> B;
  env.get_data("B", B);
  (*B)[2].assign(100.0);
  REQUIRE((*B)[2](20, 34) == 100.0f);

  ptc->append(vec_t<Pos_t, 3>(0.0, 0.0, 0.0), vec_t<Scalar, 3>(0.0, 10.0, 0.0),
              grid->get_idx(20, 34).linear, 0);

  pusher->push<boris_pusher>(0.01);
  pusher->move(0.01);

  Logger::print_debug("after push, p is {}, {}, {}", ptc->p1[0], ptc->p2[0],
                      ptc->p3[0]);
}
