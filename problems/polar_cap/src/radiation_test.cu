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

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/boundary_condition.h"
#include "systems/compute_lorentz_factor.h"
#include "systems/data_exporter.h"
#include "systems/field_solver.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_gca_lite.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/policies.h"
#include "systems/ptc_updater_base_impl.hpp"
#include "systems/radiation/curvature_emission_scheme_polar_cap.hpp"
#include "systems/radiative_transfer_impl.hpp"

namespace Aperture {

template class radiative_transfer<Config<3>, exec_policy_cuda,
                                  coord_policy_cartesian,
                                  curvature_emission_scheme_polar_cap>;

template class ptc_updater_new<Config<3>, exec_policy_cuda, coord_policy_cartesian_gca_lite>;

}

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;

  // env.params().add("dt", 3.5e-3);
  // env.params().add("N", std::vector<int64_t>({128, 128, 128}));
  // env.params().add("guard", std::vector<int64_t>({2, 2, 2}));
  // env.params().add("size", std::vector<double>({1.0, 1.0, 1.0}));
  // env.params().add("lower", std::vector<double>({0.0, 0.0, 0.0}));
  // env.params().add("periodic_boundary",
  //                  std::vector<bool>({false, false, false}));
  // env.params().add("damping_boundary",
  //                  std::vector<bool>({true, true, true, true, true, true}));
  // env.params().add("use_implicit", false);
  // env.params().add("pml_length", 8l);
  // env.params().add("fld_output_interval", 10l);

  domain_comm<Conf> comm;
  grid_t<Conf> grid(comm);

  auto solver = env.register_system<field_solver_cu<Conf>>(grid, &comm);
  auto pusher = env.register_system<
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian_gca_lite>>(grid);
  auto rad = env.register_system<
      radiative_transfer<Conf, exec_policy_cuda, coord_policy_cartesian,
                         curvature_emission_scheme_polar_cap>>(grid, &comm);
  auto lorentz = env.register_system<compute_lorentz_factor_cu<Conf>>(grid);
  auto bc = env.register_system<boundary_condition<Conf>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B0, *E0;
  env.get_data("B0", &B0);
  // env.get_data("B", &B0);
  // env.get_data("E", &E0);
  value_t Bp = sim_env().params().get_as<double>("Bp", 1.0e3);
  value_t Rpc = sim_env().params().get_as<double>("Rpc", 1.0);
  value_t R_star = sim_env().params().get_as<double>("R_star", 10.0);

  B0->set_values(0, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * x * z / (r * r * r * r * r);
  });
  B0->set_values(1, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * y * z / (r * r * r * r * r);
  });
  B0->set_values(2, [Bp, R_star](auto x, auto y, auto z) {
    z = z / R_star + 1.0;
    x /= R_star;
    y /= R_star;
    value_t r = math::sqrt(x * x + y * y + z * z);
    return 3.0f * Bp * z * z / (r * r * r * r * r) - Bp / (r * r * r);
  });
  // Bp *= -0.01;
  // E0->set_values(0, [Bp, R_star](auto x, auto y, auto z) {
  //   z = z / R_star + 1.0;
  //   x /= R_star;
  //   y /= R_star;
  //   value_t r = math::sqrt(x * x + y * y + z * z);
  //   return 3.0f * Bp * x * z / (r * r * r * r * r);
  // });
  // E0->set_values(1, [Bp, R_star](auto x, auto y, auto z) {
  //   z = z / R_star + 1.0;
  //   x /= R_star;
  //   y /= R_star;
  //   value_t r = math::sqrt(x * x + y * y + z * z);
  //   return 3.0f * Bp * y * z / (r * r * r * r * r);
  // });
  // E0->set_values(2, [Bp, R_star](auto x, auto y, auto z) {
  //   z = z / R_star + 1.0;
  //   x /= R_star;
  //   y /= R_star;
  //   value_t r = math::sqrt(x * x + y * y + z * z);
  //   return 3.0f * Bp * z * z / (r * r * r * r * r) - Bp / (r * r * r);
  // });
  particle_data_t *ptc;
  env.get_data("particles", &ptc);

  ptc->append_dev({0.0f, 0.0f, 0.0f}, {100.0f, 0.0f, 0.0f},
                  grid.dims[0] * 2 / 5 + grid.dims[1] / 2 * grid.dims[0] + 2 * grid.dims[0] * grid.dims[1],
                  1.0f, set_ptc_type_flag(0, PtcType::electron));
  cudaDeviceSynchronize();
  Logger::print_info("finished initializing a single particle");

  env.run();

  return 0;
}
