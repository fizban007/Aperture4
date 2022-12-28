/*
 * Copyright (c) 2022 Alex Chen.
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

#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_frame_dragging.h"
#include "systems/gather_momentum_space.h"
// #include "systems/legacy/ptc_updater_old.h"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_gca_lite.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_base_impl.hpp"
// #include "systems/radiation/curvature_emission_scheme_polar_cap.hpp"
#include "systems/radiation/curvature_emission_scheme_gca_lite.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

namespace Aperture {

template class radiative_transfer<Config<3>, exec_policy_gpu,
                                  coord_policy_cartesian,
                                  // curvature_emission_scheme_polar_cap>;
                                  curvature_emission_scheme_gca_lite>;

template class ptc_updater<Config<3>, exec_policy_gpu, coord_policy_cartesian_gca_lite>;

}  // namespace Aperture

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;

  // auto comm = env.register_system<domain_comm<Conf>>(env);
  domain_comm<Conf> comm;
  grid_t<Conf> grid(comm);

  auto solver = env.register_system<field_solver_frame_dragging<Conf>>(grid, &comm);
  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_gpu, coord_policy_cartesian_gca_lite>>(grid, &comm);
  auto rad = env.register_system<
      radiative_transfer<Conf, exec_policy_gpu, coord_policy_cartesian,
                         curvature_emission_scheme_gca_lite>>(grid, &comm);
                         // curvature_emission_scheme_polar_cap>>(grid, &comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto bc = env.register_system<boundary_condition<Conf>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf, exec_policy_gpu>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *Bdelta, *Edelta;
  value_t Bp = sim_env().params().get_as<double>("Bp", 1.0e3);
  value_t Rpc = sim_env().params().get_as<double>("Rpc", 1.0);
  value_t R_star = sim_env().params().get_as<double>("R_star", 10.0);

  auto B1_func = [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * x * z / (r * r * r * r * r);
  };
  auto B2_func = [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * y * z / (r * r * r * r * r);
  };
  auto B3_func = [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * z * z / (r * r * r * r * r) - Bp / (r * r * r);
  };
  // Set initial dipole field
  env.get_data("B0", &B0);
  B0->set_values(0, B1_func);
  B0->set_values(1, B2_func);
  B0->set_values(2, B3_func);

  particle_data_t *ptc;
  rng_states_t<exec_tags::device> *states;

  env.run();
  return 0;
}
