/*
 * Copyright (c) 2023 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "data/fields.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"

using namespace Aperture;

template <typename Conf>
class boundary_condition : public system_t {
 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(
      const domain_comm<Conf, exec_policy_dynamic>* comm = nullptr)
      : m_comm(comm) {}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("Bdelta", B);
  }
  void update(double dt, uint32_t step) override {
    float time = sim_env().get_time();
    float E0 = m_E0;
    float omega = m_omega;
    float num_lambda = m_num_lambda;

    if (m_comm != nullptr &&
        m_comm->domain_info().neighbor_left[0] == MPI_PROC_NULL) {
      exec_policy_dynamic<Conf>::launch(
          [time, E0, omega, num_lambda] LAMBDA(auto e, auto b) {
            auto& grid = exec_policy_dynamic<Conf>::grid();
            auto ext = grid.extent();
            exec_policy_dynamic<Conf>::loop(
                0, grid.dims[1], [&] LAMBDA(auto n1) {
                  int n0 = grid.guard[0];
                  auto idx = Conf::idx(index_t<2>(n0, n1), ext);
                  if (omega * time < 2.0 * M_PI * num_lambda) {
                    e[1][idx] = E0 * math::sin(omega * time);
                  }
                });
          },
          E, B);
    }
  }

 private:
  const domain_comm<Conf, exec_policy_dynamic>* m_comm;
  nonown_ptr<vector_field<Conf>> E, B;

  float m_omega = 10.0;
  float m_E0 = 1.0;
  int m_num_lambda = 3;
};

int
main(int argc, char* argv[]) {
  typedef Config<2> Conf;
  auto& env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;
  using exec_policy = exec_policy_dynamic<Conf>;
  using exec_tag = typename exec_policy::exec_tag;

  // Specify config parameters
  env.params().add("log_level", (int64_t)LogLevel::debug);
  env.params().add("N", std::vector<int64_t>({128, 128}));
  env.params().add("guard", std::vector<int64_t>({3, 3}));
  env.params().add("nodes", std::vector<int64_t>({2, 2}));
  env.params().add("periodic_boundary", std::vector<bool>({false, true}));
  env.params().add("use_pml", true);
  env.params().add("pml_length", 8);
  env.params().add("damping_boundary",
                   std::vector<bool>({false, true, false, false}));
  env.params().add("size", std::vector<double>({1.0, 1.0}));
  env.params().add("lower", std::vector<double>({0.0, 0.0}));
  env.params().add("max_ptc_num", 10);
  env.params().add("fld_output_interval", 10);
  env.params().add("ptc_output_interval", 10);
  env.params().add("dt", 5.0e-3);
  env.params().add("max_steps", 1000);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto boundary = env.register_system<boundary_condition<Conf>>(&comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  env.run();

  return 0;
}
