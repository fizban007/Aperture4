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
#include "data/phase_space.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_cartesian.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/vlasov_solver.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<1> Conf;
  auto &env = sim_environment::instance(&argc, &argv, true);
  typedef typename Conf::value_t value_t;
  using exec_policy = exec_policy_dynamic<Conf>;

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_t<Conf> grid(comm);
  auto vlasov = env.register_system<
      vlasov_solver<Conf, 1, exec_policy_dynamic, coord_policy_cartesian>>(
      grid, &comm);
  auto solver = env.register_system<
      field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
                                                                       &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(
      grid, &comm);

  env.init();

  cout << "after init" << endl;

  // Prepare initial conditions
  std::vector<phase_space<Conf, 1> *> f(2);
  env.get_data("f_e", &f[0]);
  env.get_data("f_p", &f[1]);

  cout << "after get data" << endl;

  // data is the distribution function of electrons
  vec_t<int, 1> momentum_ext;
  vec_t<double, 1> momentum_lower;
  vec_t<double, 1> momentum_upper;
  sim_env().params().get_vec_t("momentum_ext", momentum_ext);
  sim_env().params().get_vec_t("momentum_lower", momentum_lower);
  sim_env().params().get_vec_t("momentum_upper", momentum_upper);
  std::cout << "momuntum_size" << momentum_ext[0];
  double dp = (momentum_upper[0] - momentum_lower[0]) / momentum_ext[0];

  cout << "before set_value" << std::endl;
  double p_stream = 0.5;
  sim_env().params().get_value("p_stream", p_stream);

  f[0]->set_value(
      [=](double p0, double p1, double p2, double x0, double x1, double x2) {
        if (math::abs(math::abs(p0) - p_stream) < 0.5 * dp) {
          // if (math::abs(x0 - 0.5) < 0.1)
          return 1.0 / dp + 1.0e-1 * std::cos(2.0 * M_PI * x0);
        }
        return 0.0;
      });

  env.run();
  return 0;
}
