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
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_updater_base.h"
// #include "systems/policies/exec_policy_host.hpp"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/coord_policy_cartesian_gca_lite.hpp"
#include "systems/policies/ptc_physics_policy_empty.hpp"
#include "systems/ptc_updater_base_impl.hpp"
#include "systems/radiation/curvature_emission_scheme_polar_cap.hpp"
// #include "systems/radiation/curvature_emission_scheme_gca_lite.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "utils/hdf_wrapper.h"
#include <vector>

namespace Aperture {

template class radiative_transfer<Config<3>, exec_policy_cuda,
                                  coord_policy_cartesian,
                                  curvature_emission_scheme_polar_cap>;
template class ptc_updater_new<Config<3>, exec_policy_cuda,
                               coord_policy_cartesian_gca_lite>;

}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<3> Conf;
  auto &env = sim_environment::instance(&argc, &argv);
  typedef typename Conf::value_t value_t;

  domain_comm<Conf> comm;
  grid_t<Conf> grid;

  auto ptc_data = env.register_data<particle_data_t>(
      "particles", env.params().get_as("max_ptc_num", 1000000l),
      MemType::device_managed);
  auto ph_data = env.register_data<photon_data_t>(
      "photons", env.params().get_as("max_ph_num", 1000000l),
      MemType::device_managed);

  auto pusher = env.register_system<
      // ptc_updater_new<Conf, exec_policy_cuda,
      // coord_policy_cartesian_gca_lite>>(grid, comm);
      ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian>>(grid,
                                                                       comm);
  auto rad = env.register_system<
      radiative_transfer<Conf, exec_policy_cuda, coord_policy_cartesian,
                         curvature_emission_scheme_polar_cap>>(grid, &comm);
  auto exporter = env.register_system<data_exporter<Conf>>(grid, &comm);

  env.init();

  vector_field<Conf> *B, *E;
  particle_data_t *ptc;
  rng_states_t *states;

  env.get_data("B", &B);
  env.get_data("E", &E);
  env.get_data("particles", &ptc);

  // Set initial condition
  value_t Bp = env.params().get_as("Bp", 1.0e4);
  value_t R_star = 10.0;
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

  B->set_values(0, B1_func);
  B->set_values(1, B2_func);
  B->set_values(2, B3_func);
  value_t E_frac = env.params().get_as("E_frac", 0.1);
  value_t E0 = E_frac * Bp;
  E->set_values(2, [E0, R_star](auto x, auto y, auto z) {
    z = z + 1.0;
    return E0 / z / z;
  });

  auto ptc_global_x = vec_t<value_t, 3>(0.4, 0.4, 0.01);
  vec_t<value_t, 3> rel_x;
  uint32_t cell;
  grid.from_x_global(ptc_global_x, rel_x, cell);
  Logger::print_info("cell is {}, {}, {}", cell / (grid.dims[0] * grid.dims[1]),
                     (cell / grid.dims[0]) % grid.dims[1], cell % grid.dims[0]);

  value_t ptc_p_parallel = env.params().get_as("p0", 100.0);
  vec_t<value_t, 3> ptc_p(
      B1_func(ptc_global_x[0], ptc_global_x[1], ptc_global_x[2]),
      B2_func(ptc_global_x[0], ptc_global_x[1], ptc_global_x[2]),
      B3_func(ptc_global_x[0], ptc_global_x[1], ptc_global_x[2]));
  ptc_p *= ptc_p_parallel / math::sqrt(ptc_p.dot(ptc_p));

  ptc->append_dev(rel_x, ptc_p, cell, 1.0,
                  gen_ptc_type_flag(PtcType::positron));
  std::cout << "Total steps is " << env.get_max_steps() << std::endl;
  std::cout << ptc->p1[2] << std::endl;
  std::vector<value_t> x(env.get_max_steps());
  std::vector<value_t> y(env.get_max_steps());
  std::vector<value_t> z(env.get_max_steps());
  std::vector<value_t> gamma(env.get_max_steps());
  vec_t<value_t, 3> x_global;

#ifdef GPU_ENABLED
    size_t free_mem, total_mem;
    GpuSafeCall(gpuMemGetInfo(&free_mem, &total_mem));
    Logger::print_info("GPU memory: free={:.3f}GiB/{:.3f}GiB", free_mem / 1.0e9,
                       total_mem / 1.0e9);
#endif
  for (int i = 0; i < env.get_max_steps(); i++) {
    std::cout << "at step " << i << std::endl;
    env.update();
    x_global =
        grid.x_global({ptc->x1[0], ptc->x2[0], ptc->x3[0]}, ptc->cell[0]);
    x[i] = x_global[0];
    y[i] = x_global[1];
    z[i] = x_global[2];
    gamma[i] = ptc->E[0];
    Logger::print_info("Current pos ({}, {}, {}), gamma {}", x[i], y[i], z[i], gamma[i]);
  }

  auto file = hdf_create("test_gca_compare.h5");
  file.write(x.data(), x.size(), "x");
  file.write(y.data(), y.size(), "y");
  file.write(z.data(), z.size(), "z");
  file.write(gamma.data(), gamma.size(), "gamma");
  file.write(ph_data->E.dev_ptr(), ph_data->number(), "Eph");
  file.write(ptc_data->E.dev_ptr(), ptc_data->number(), "Eptc");
  file.close();

  return 0;
}
