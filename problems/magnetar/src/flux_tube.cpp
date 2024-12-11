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

#include "core/math.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/compute_moments.h"
#include "systems/data_exporter.h"
#include "systems/field_solver_sph.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "systems/policies/coord_policy_spherical_sync_cooling.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater_impl.hpp"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/resonant_scattering_scheme.hpp"
// #include "systems/boundary_condition.h"
#include <iostream>

namespace Aperture {
  template class radiative_transfer<Config<2>, exec_policy_dynamic,
                                  coord_policy_spherical, resonant_scattering_scheme>;
}

using namespace std;
using namespace Aperture;

int
main(int argc, char *argv[]) {
  typedef Config<2> Conf;
  using value_t = typename Config<2>::value_t;
  auto &env = sim_environment::instance(&argc, &argv);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_sph_t<Conf> grid(comm);
  auto pusher =
      env.register_system<ptc_updater<Conf, exec_policy_dynamic,
                                      coord_policy_spherical>>(
          grid, &comm);
  auto tracker =
      env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  auto rad = env.register_system<radiative_transfer<
      Conf, exec_policy_dynamic, coord_policy_spherical,
      resonant_scattering_scheme>>( grid, &comm);
  auto exporter =
      env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);
    auto moments =
    env.register_system<compute_moments<Conf, exec_policy_dynamic>>(grid);

  env.init();

  double Bp = 10000.0;
  double BQ = 1000.0;
  double tube_rmax_1 = 4.0;
  double tube_rmax_2 = 6.0;
  int ppc = 10;
  env.params().get_value("Bp", Bp);
  env.params().get_value("B_Q", BQ);
  env.params().get_value("tube_rmax_1", tube_rmax_1);
  env.params().get_value("tube_rmax_2", tube_rmax_2);
  env.params().get_value("ppc", ppc);
  // Set initial condition
  // set_initial_condition(env, *grid, 0, 1.0, Bp);
  vector_field<Conf> *B0, *B;
  particle_data_t *ptc;
  // env.get_data("B0", &B0);
  env.get_data("B", &B);
  env.get_data("particles", &ptc);

  // Set dipole initial magnetic field
  B->set_values(0, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    // return Bp / (r * r);
    return Bp * 2.0 * cos(theta) / cube(r);
  });
  B->set_values(1, [Bp](Scalar x, Scalar theta, Scalar phi) {
    Scalar r = grid_sph_t<Conf>::radius(x);
    return Bp * sin(theta) / cube(r);
  });

  // Add a distribution of particles along an entire flux tube
  //Scalar p0 = 1000.0f; //mag of momentum for all particles
  //define which flux tube we look at by rmax in config yielding iniial theta on star
  value_t tube_th1 = math::asin(math::sqrt(1.0 / tube_rmax_1));
  value_t tube_th2 = math::asin(math::sqrt(1.0 / tube_rmax_2));

  ptc_injector_dynamic<Conf> injector(grid);
  injector.inject_pairs(
    // Injection criterion (in flux tube)
      [tube_rmax_1,tube_rmax_2] LAMBDA(auto &pos, auto &grid, auto &ext) { 
        //TODO check if this is the actual r we are trying to input
        auto r = math::exp(grid.template coord<0>(pos, 0.5f));// Since r_grid = log(r)
        auto th = grid.template coord<1>(pos, 0.5f);
        auto r_max_check = r / (sin(th) * sin(th));
        if (r_max_check > tube_rmax_1 && r_max_check < tube_rmax_2) {
            //printf("r_max_check is %f\n tube_rmax_1/2 is %f,%f", r_max_check, tube_rmax_1, tube_rmax_2);
          return true;
        }else {
          return false;
        }
        },
      // Number injected per cell
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return ppc; },
        // Initialize particle momentum
      [Bp,BQ] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto &grid = static_cast<const grid_sph_t<Conf> &>(
            exec_policy_dynamic<Conf>::grid());
        auto r =grid.radius(x_global[0]);
        auto th = grid.theta(x_global[1]);
        // From Belobodorov 2013 gamma is 100b where b = B/BQ
        //TODO check if this is the correct conversion to momentum
        //value_t gamma = 100*math::sqrt(Bp/ cube(r) * (4*cos(th)*cos(th) + sin(th)*sin(th)))/BQ; // i.e Sqrt(B1^2+B2^2)/BQ
        value_t gamma = 100 * Bp/BQ / cube(r) * math::sqrt( 1 + 3*cos(th)*cos(th));
        value_t p0 = math::sqrt(gamma*gamma - 1.0); // gamma = sqrt(1+p^2)
        p0 = p0 / math::sqrt( 1.0 + 3.0 * cos(th)*cos(th) ); // Normalization
        if (th <= M_PI / 2.0) {
          return vec_t<value_t, 3>(p0 * 2.0 * cos(th), p0 * sin(th), 0);
        } else {
          return vec_t<value_t, 3>(-p0 * 2.0 * cos(th), -p0 * sin(th), 0);
        }
        //return vec_t<value_t, 3>(p0 * 2.0 * cos(th), p0 * sin(th), 0);
      },
        // Initialize particle weight (i.e makes each particle worth a different amount of "actual" particles)
      [] LAMBDA(auto &x_global, PtcType type) {
        // auto &grid = static_cast<const grid_sph_t<Conf> &>(
        //     exec_policy_dynamic<Conf>::grid());
        //auto r = grid_sph_t<Conf>::radius(x_global[0]);
        auto th = grid_sph_t<Conf>::theta(x_global[1]);
        return 1.0 * math::sin(th);
      }
      );
  


  //Scalar r0 = 1.1f;
  //Scalar th0 = 0.338f;
//   int N = 5000;
//   for (int i = 0; i < N; i++) {
//     // ptc->append(exec_tags::device{}, {0.5f, 0.5f, 0.0f}, {p0, 0.0f, 0.0f}, 10 + 60 * grid->dims[0],
//     //                 100.0);
//     ptc_append_global(exec_policy_dynamic<Conf>::exec_tag{}, *ptc, grid,
//                       {grid_sph_t<Conf>::from_radius(r0), th0, 0.0f},
//                       {p0 * 2.0 * std::cos(th0), p0 * std::sin(th0), 0.0f}, 1.0f, flag_or(PtcFlag::tracked));
//   }

  env.run();
  return 0;
}
