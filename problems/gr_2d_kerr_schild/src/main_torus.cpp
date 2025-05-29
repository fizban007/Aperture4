/*
 * Copyright (c) 2020 Alex Chen.
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

#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/compute_moments_gr_ks.h"
#include "systems/grid_ks.h"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/radiation/gr_ks_ic_radiation_scheme_fido.hpp"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include "utils/util_functions.h"
#include "utils/hdf_wrapper.h"
#include "utils/interpolation.hpp"

using namespace std;

namespace Aperture {

HOST_DEVICE Scalar test_func(Scalar x, Scalar y) {
  return x * x + y * y;
}

// template class radiative_transfer<Config<2>, exec_policy_gpu,
//                                   coord_policy_gr_ks_sph,
//                                   default_radiation_scheme_gr>;

}  // namespace Aperture

using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv, true);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_ks_t<Conf> grid(comm);

  auto pusher = env.register_system<
      ptc_updater<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto moments = env.register_system<compute_moments_gr_ks<Conf, exec_policy_dynamic>>(grid);
  // auto injector = env.register_system<bh_injector<Conf>>(grid);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  // auto radiation = env.register_system<
  //   radiative_transfer<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph,
  //                     //  default_radiation_scheme_gr>>(grid, &comm);
  //                      gr_ks_ic_radiation_scheme_fido>>(grid, &comm);
  // auto solver = env.register_system<
  //     field_solver<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  // Prepare initial condition here
  // vector_field<Conf> *B, *D, *B0, *D0;
  // particle_data_t *ptc;
  // env.get_data("B0", &B0);
  // env.get_data("E0", &D0);
  // env.get_data("Bdelta", &B);
  // env.get_data("Edelta", &D);
  // env.get_data("particles", &ptc);

  // Load the torus IC hdf5 file
  auto file = H5File("collisionless_torus.h5");

  auto torus_density = file.read_multi_array<double, 2>("density");
  auto torus_r_range = file.read_vector<double>("r_range");
  auto torus_th_range = file.read_vector<double>("th_range");
  torus_density.copy_to_device();
  double r_lower = torus_r_range[0];
  double r_upper = torus_r_range[1];
  double th_lower = torus_th_range[0];
  double th_upper = torus_th_range[1];
  
  auto ext_torus = torus_density.extent();
  double torus_dr = (torus_r_range[1] - torus_r_range[0]) / ext_torus[0];
  double torus_dth = (torus_th_range[1] - torus_th_range[0]) / ext_torus[1];
  
  Logger::print_info("r_range: {}, {}", torus_r_range[0], torus_r_range[1]);
  Logger::print_info("th_range: {}, {}", torus_th_range[0], torus_th_range[1]);

  // Obtain pointers for the arrays (host or device)
  auto density_ptr = adapt(exec_tags::dynamic{}, torus_density);

  ptc_injector_dynamic<Conf> ptc_inj(grid);
  ptc_inj.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [r_lower, r_upper, th_lower, th_upper, ext_torus, torus_dr, torus_dth,
       density_ptr] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto r = grid_ks_t<Conf>::radius(
                  grid.template coord<0>(pos[0], false));
        auto th = grid_ks_t<Conf>::theta(
                   grid.template coord<1>(pos[1], false));
        // TODO: Write a condition for injection given r and th
        if (r < r_lower || r > r_upper || th < th_lower || th > th_upper) {
          return false;
        }

        // TODO: do an interpolation and do not inject when density is 0
        int nr = std::floor((r - r_lower) / torus_dr);
        int nth = std::floor((th - th_lower) / torus_dth);
        index_t<2> pos_torus(nr, nth);
        auto idx = Conf::idx(pos_torus, ext_torus);
        lerp<2> interp;
        auto density = interp(density_ptr, 
                              vec_t<value_t, 3>(r - nr * torus_dr, th - nth * torus_dth, 0.0),
                              idx);
        if (density > 0.0) 
          return true;
        else
          return false;
        
        // return true;
      },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return 100;
      },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        // TODO: do the random number drawing here to get the particle momenta
        return vec_t<value_t, 3>(0.0, 0.0, 0.0);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [r_lower, r_upper, th_lower, th_upper, ext_torus, torus_dr, torus_dth,
        density_ptr] LAMBDA(auto &x_global, PtcType type) {
        // TODO: do interpolation and obtain the particle weights
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);
        value_t th = grid_ks_t<Conf>::theta(x_global[1]);

        int nr = std::floor((r - r_lower) / torus_dr);
        int nth = std::floor((th - th_lower) / torus_dth);
        index_t<2> pos(nr, nth);
        auto idx = Conf::idx(pos, ext_torus);
        lerp<2> interp;
        auto density = interp(density_ptr, 
                              vec_t<value_t, 3>(r - nr * torus_dr, th - nth * torus_dth, 0.0),
                              idx);
        
        // TODO: Translate density to particle weight
        return math::sin(x_global[1]);
      });

  // env.run();

  return 0;
}
